"""mesh_api.py -- DEM -> 3 Plotly interactive figures (JSON)

Endpoint
--------
POST /mesh  { lat, lon, bbox_deg, wind_speed?, wind_dir? }
Returns     { plotly_dem, plotly_pts, plotly_tri, wind_grid, meta }

Plotly figures are interactive (rotate, zoom, pan, hover).
Wind arrows use terrain-corrected vectors from air/air.py when
wind_speed + wind_dir are supplied; uniform fallback otherwise.
"""

import json
import math
import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import config
from core.landscape import Landscape, decode_corine_tif, decode_corine_png

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds as _rio_window
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False

app  = Flask("mesh_api")
CORS(app)

M_PER_DEG     = 111_320.0
BG            = "#0a0a14"
N_CAP         = 130          # DEM grid cells per side
N_PTS_CAP     = 4_000        # scatter slide max points
N_MESH_CAP    = 20_000       # > rows*cols forces stride=1 → full DEM resolution texture
N_ARROW       = 10           # wind-arrow grid: N_ARROW x N_ARROW
POPUP_EXTENT_M = 2_000.0     # physical square side in metres (always 2 km × 2 km)
SAT_ZOOM      = 17           # ESRI satellite zoom: 17 → ~1.2 m/px
DEM_ZOOM      = 14           # Terrarium DEM zoom:  14 → ~9.5 m/px


# ── Tile / bounds helpers ──────────────────────────────────────────────────

def _deg_to_tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple:
    n     = 1 << zoom
    x     = int((lon_deg + 180.0) / 360.0 * n)
    lat_r = math.radians(lat_deg)
    y     = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile_nw_deg(x: int, y: int, zoom: int) -> tuple:
    """NW corner (lat, lon) of slippy-map tile (x, y)."""
    n   = 1 << zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def _square_bounds(lat: float, lon: float, extent_m: float = POPUP_EXTENT_M) -> tuple:
    """
    Return (west, south, east, north) for a *physically* square region
    centred at (lat, lon).  Corrects for cos(lat) so width_m == height_m.
    """
    half    = extent_m / 2.0
    lat_buf = half / M_PER_DEG
    lon_buf = half / (M_PER_DEG * math.cos(math.radians(lat)))
    return lon - lon_buf, lat - lat_buf, lon + lon_buf, lat + lat_buf


# ── Plotly style helpers ───────────────────────────────────────────────────

def _scene(z_label: str = "Z (m)") -> dict:
    ax = {"gridcolor": "#2e2e3e", "color": "#6c7086",
          "backgroundcolor": BG, "showbackground": True}
    return {
        "xaxis": {**ax, "title": "X (m)"},
        "yaxis": {**ax, "title": "Y (m)"},
        "zaxis": {**ax, "title": z_label},
        "bgcolor": BG,
        "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.0}},
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35},
    }


def _colorbar(title: str, thickness: int = 12) -> dict:
    return {
        "title": {"text": title, "font": {"color": "#a6adc8", "size": 10}},
        "tickfont": {"color": "#a6adc8", "size": 9},
        "outlinecolor": "#313244",
        "thickness": thickness,
    }


def _layout_base() -> dict:
    return {
        "paper_bgcolor": BG,
        "plot_bgcolor":  BG,
        "font":    {"color": "#a6adc8", "size": 10},
        "margin":  {"l": 0, "r": 0, "t": 38, "b": 0},
        "modebar": {"bgcolor": "transparent", "color": "#6c7086",
                    "activecolor": "#cdd6f4"},
    }


# ── DEM generation ─────────────────────────────────────────────────────────

def _generate_dem(n: int) -> tuple[np.ndarray, Landscape]:
    """Synthetic fallback — used only when the real DEM fetch fails."""
    big = max(64, n * 2)
    land = Landscape(config)
    land.shape = (big, big)
    land.fuel_names = getattr(land, "fuel_names", ["Grass", "Shrub", "Forest"])
    land.generate_random_terrain(num_patches=8)
    r0 = (big - n) // 2
    c0 = (big - n) // 2
    dem = land.elevation[r0:r0 + n, c0:c0 + n].astype(np.float32)
    land.elevation = dem
    land.fuel_map  = land.fuel_map[r0:r0 + n, c0:c0 + n]
    land.shape     = (n, n)
    return dem, land


def _fetch_terrarium_dem(west: float, south: float,
                         east: float, north: float,
                         n: int, zoom: int = DEM_ZOOM) -> "np.ndarray | None":
    """
    Fetch Terrarium elevation tiles and return a south-up float32 (n, n) grid.
    Uses only PIL + numpy — no rasterio required.
    Terrarium encoding: elevation = R*256 + G + B/256 - 32768 (metres).
    Falls back to None on failure.
    """
    try:
        from PIL import Image as _PImg
        import requests as _req
        from concurrent.futures import ThreadPoolExecutor, as_completed as _asc
        import io as _io

        TILE    = 256
        TER_URL = ("https://s3.amazonaws.com/elevation-tiles-prod/"
                   "terrarium/{z}/{x}/{y}.png")

        x0, y0 = _deg_to_tile(north, west,  zoom)
        x1, y1 = _deg_to_tile(south, east,  zoom)
        x1 = max(x1, x0);  y1 = max(y1, y0)
        nx, ny  = x1 - x0 + 1, y1 - y0 + 1

        print(f"[mesh_api] Terrarium z{zoom}: {nx}x{ny}={nx*ny} tiles ...")

        def _get(tx_ty):
            tx, ty = tx_ty
            url    = TER_URL.format(z=zoom, x=tx, y=ty)
            for _ in range(3):
                try:
                    r   = _req.get(url, timeout=20)
                    r.raise_for_status()
                    img = _PImg.open(_io.BytesIO(r.content)).convert("RGBA")
                    arr = np.array(img, dtype=np.float32)
                    elev = arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0
                    return tx, ty, elev
                except Exception:
                    pass
            return tx, ty, None

        canvas = np.zeros((ny * TILE, nx * TILE), dtype=np.float32)
        jobs   = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
        failed = 0
        with ThreadPoolExecutor(max_workers=min(16, len(jobs))) as pool:
            for fut in _asc({pool.submit(_get, j): j for j in jobs}):
                tx, ty, tile_elev = fut.result()
                if tile_elev is not None:
                    row = (ty - y0) * TILE;  col = (tx - x0) * TILE
                    canvas[row:row + TILE, col:col + TILE] = tile_elev
                else:
                    failed += 1

        if failed == len(jobs):
            print("[mesh_api] Terrarium: all tiles failed")
            return None

        # Canvas is north-up (row 0 = north).  Crop to exact bbox.
        nw_lat, nw_lon = _tile_nw_deg(x0,     y0,     zoom)
        se_lat, se_lon = _tile_nw_deg(x1 + 1, y1 + 1, zoom)
        H_c, W_c       = canvas.shape[:2]
        lat_span        = nw_lat - se_lat
        lon_span        = se_lon - nw_lon

        r0c = max(0, int((nw_lat - north) / lat_span * H_c))
        r1c = min(H_c, int((nw_lat - south) / lat_span * H_c))
        c0c = max(0, int((west   - nw_lon) / lon_span * W_c))
        c1c = min(W_c, int((east  - nw_lon) / lon_span * W_c))
        cropped = canvas[r0c:r1c, c0c:c1c]

        # Resample to n×n using PIL (float32 mode)
        img_elev   = _PImg.fromarray(cropped, mode="F")
        img_scaled = img_elev.resize((n, n), _PImg.BILINEAR)
        dem        = np.array(img_scaled, dtype=np.float32)

        # Flip to south-up (row 0 = south), clamp negatives
        dem = np.flipud(dem)
        dem = np.where(dem < 0, 0.0, dem)

        print(f"[mesh_api] Terrarium DEM {n}x{n} ({cropped.shape[1]}x{cropped.shape[0]} raw px, "
              f"{failed}/{len(jobs)} failed)")
        return dem

    except Exception as exc:
        print(f"[mesh_api] Terrarium DEM error: {exc}")
        return None


def _load_real_dem(lat: float, lon: float, n: int,
                    extent_m: float = POPUP_EXTENT_M):
    """
    Fetch Terrarium terrain for a physically square *extent_m* × *extent_m*
    area centred at (lat, lon) and resample to n×n.

    Primary path  : fetch_terrain_from_api  → rasterio crop/resample (if rasterio is installed)
    Fallback path : _fetch_terrarium_dem    → direct tile fetch with PIL (no rasterio needed)
    Last resort   : synthetic random terrain

    Returns (dem, land, cell_m, bounds=(west, south, east, north)).
    """
    west, south, east, north = _square_bounds(lat, lon, extent_m=extent_m)
    cell_m = extent_m / n          # always square: same in x and y

    if not _RASTERIO_OK:
        # No rasterio — fetch Terrarium tiles directly with PIL
        dem = _fetch_terrarium_dem(west, south, east, north, n)
        if dem is not None:
            land            = Landscape(config)
            land.shape      = (n, n)
            land.elevation  = dem
            land.fuel_names = getattr(land, "fuel_names", ["Grass", "Shrub", "Forest"])
            land.fuel_map   = np.zeros((n, n), dtype=int)
            return dem, land, cell_m, (west, south, east, north)
        print("[mesh_api] Terrarium fallback failed — using synthetic terrain")
        dem, land = _generate_dem(n)
        return dem, land, cell_m, (west, south, east, north)

    try:
        from pipeline.auto_fetcher import fetch_terrain_from_api

        # lon_buf > lat_buf for lat≠0; pass it so the fetcher covers full width
        lon_buf = (east - west) / 2.0
        tif_path = fetch_terrain_from_api(lat, lon, buffer=lon_buf, output_dir=ROOT_DIR)

        if tif_path is None or not os.path.exists(tif_path):
            raise RuntimeError("fetch_terrain_from_api returned no file")

        with rasterio.open(tif_path) as src:
            # Crop to the exact physically-square bounds before resampling
            win = _rio_window(west, south, east, north, src.transform)
            dem_raw = src.read(
                1,
                window=win,
                out_shape=(n, n),
                resampling=Resampling.bilinear,
                boundless=True,      # pad with nodata if window slightly exceeds file
                fill_value=0,
            ).astype(np.float32)

        # GeoTIFF is north-up; flip so row 0 = south
        dem_raw = np.flipud(dem_raw)
        dem_raw = np.where(dem_raw < 0, 0.0, dem_raw)

        land = Landscape(config)
        land.shape      = (n, n)
        land.elevation  = dem_raw
        land.fuel_names = getattr(land, "fuel_names", ["Grass", "Shrub", "Forest"])
        land.fuel_map   = np.zeros((n, n), dtype=int)

        print(f"[mesh_api] Real DEM — {n}x{n}  "
              f"{extent_m/1000:.2f}km x {extent_m/1000:.2f}km  "
              f"cell {cell_m:.1f} m")
        return dem_raw, land, cell_m, (west, south, east, north)

    except Exception as exc:
        print(f"[mesh_api] Real DEM fetch failed ({exc}); using synthetic terrain")
        dem, land = _generate_dem(n)
        return dem, land, cell_m, (west, south, east, north)


def _load_real_fuel_map(land: Landscape, west: float, south: float,
                        east: float, north: float, n: int) -> bool:
    """
    Fetch CORINE Land Cover for the popup bbox and decode it into
    ``land.fuel_map`` so the wind field (air.py Layer 1 — WAF) sees real
    per-cell vegetation instead of a uniform placeholder.

    Without this, every popup cell defaults to the same fuel index (0),
    the Wind Adjustment Factor is spatially uniform, and the corrected
    wind field ends up pointing almost the same direction everywhere
    (only the small upslope-draft term varies cell to cell).

    Returns True on success; land.fuel_map is left untouched (all-zero)
    on any failure so the caller can fall back gracefully.
    """
    try:
        from pipeline.auto_fetcher import fetch_corine_land_cover
        corine_path = fetch_corine_land_cover(
            west, south, east, north, width=n, height=n, output_dir=ROOT_DIR)
        if not corine_path:
            return False

        ext = str(corine_path).lower()
        if ext.endswith((".tif", ".tiff")):
            if not _RASTERIO_OK:
                return False
            land.fuel_map = decode_corine_tif(corine_path, land.fuel_names, n, n)
        else:
            land.fuel_map = decode_corine_png(corine_path, land.fuel_names, n, n)
        return True
    except Exception as exc:
        print(f"[mesh_api] CORINE fuel-map load failed ({exc}); wind uses uniform WAF")
        return False


# ── Satellite texture ──────────────────────────────────────────────────────

def _bilinear_sample(arr: np.ndarray,
                     row_f: np.ndarray,
                     col_f: np.ndarray) -> np.ndarray:
    """Bilinear-interpolate float (row, col) positions in a (H, W, 3) uint8 array."""
    H, W = arr.shape[:2]
    r0   = np.floor(row_f).astype(int)
    c0   = np.floor(col_f).astype(int)
    r1   = np.clip(r0 + 1, 0, H - 1);  r0 = np.clip(r0, 0, H - 1)
    c1   = np.clip(c0 + 1, 0, W - 1);  c0 = np.clip(c0, 0, W - 1)
    dr   = (row_f - np.floor(row_f))[:, None]
    dc   = (col_f - np.floor(col_f))[:, None]
    return np.clip(
        arr[r0, c0] * (1 - dr) * (1 - dc) +
        arr[r0, c1] * (1 - dr) *      dc  +
        arr[r1, c0] *      dr  * (1 - dc) +
        arr[r1, c1] *      dr  *      dc,
        0, 255,
    ).astype(np.uint8)


def _fetch_sat_image(west: float, south: float,
                     east: float, north: float,
                     cache_path: str,
                     zoom: int = SAT_ZOOM) -> "np.ndarray | None":
    """
    Fetch ESRI World Imagery tiles at `zoom` and stitch/crop to (west,south,east,north).
    Saves the result as a JPEG at cache_path and returns a north-up uint8 (H,W,3) array.
    Returns None on complete failure.
    """
    from PIL import Image as _PImage
    import requests as _req
    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc
    import io as _io

    TILE    = 256
    SAT_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}")

    # Tile-index range  (y_min = northernmost tile, y_max = southernmost)
    x0, y0 = _deg_to_tile(north, west,  zoom)
    x1, y1 = _deg_to_tile(south, east,  zoom)
    x1 = max(x1, x0);  y1 = max(y1, y0)
    nx, ny  = x1 - x0 + 1, y1 - y0 + 1

    print(f"[mesh_api] Fetching satellite z{zoom}: {nx}×{ny}={nx*ny} tiles …")

    def _get(tx_ty):
        tx, ty = tx_ty
        url = SAT_URL.format(z=zoom, y=ty, x=tx)
        for _ in range(3):
            try:
                r = _req.get(url, timeout=20)
                r.raise_for_status()
                img = _PImage.open(_io.BytesIO(r.content)).convert("RGB")
                return tx, ty, np.array(img, dtype=np.uint8)
            except Exception:
                pass
        return tx, ty, None

    canvas = np.zeros((ny * TILE, nx * TILE, 3), dtype=np.uint8)
    jobs   = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    failed = 0
    with ThreadPoolExecutor(max_workers=min(32, len(jobs))) as pool:
        for fut in _asc({pool.submit(_get, j): j for j in jobs}):
            tx, ty, tile = fut.result()
            if tile is not None:
                row = (ty - y0) * TILE;  col = (tx - x0) * TILE
                canvas[row:row + TILE, col:col + TILE] = tile
            else:
                failed += 1

    if failed == len(jobs):
        print("[mesh_api] Satellite fetch: all tiles failed")
        return None

    # Canvas NW and SE corners in degrees
    nw_lat, nw_lon = _tile_nw_deg(x0,     y0,     zoom)
    se_lat, se_lon = _tile_nw_deg(x1 + 1, y1 + 1, zoom)
    H_c, W_c       = canvas.shape[:2]
    lat_span        = nw_lat - se_lat
    lon_span        = se_lon - nw_lon

    # Crop to exact requested bbox (still north-up)
    r0c = max(0, int((nw_lat - north) / lat_span * H_c))
    r1c = min(H_c, int((nw_lat - south) / lat_span * H_c))
    c0c = max(0, int((west   - nw_lon) / lon_span * W_c))
    c1c = min(W_c, int((east  - nw_lon) / lon_span * W_c))
    cropped = canvas[r0c:r1c, c0c:c1c]

    # Save cache
    _PImage.fromarray(cropped).save(cache_path, "JPEG", quality=95)
    H_f, W_f = cropped.shape[:2]
    print(f"[mesh_api] Satellite {W_f}×{H_f} px saved  (zoom {zoom}, {failed} tiles failed)")
    return cropped


def _fetch_sat_colors(lat: float, lon: float,
                      west: float, south: float,
                      east: float, north: float,
                      mx: np.ndarray, my_s: np.ndarray) -> "list | None":
    """
    Return a list of 'rgb(r,g,b)' strings for each mesh vertex, sampled from the
    high-res satellite image via geographic-coordinate bilinear interpolation.
    Only used for the mesh slide.
    """
    try:
        # Include bounds in cache key so different extents get different files
        tag        = f"{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}_z{SAT_ZOOM}"
        cache_path = os.path.join(ROOT_DIR, f"sat_{tag}.jpg")

        if os.path.exists(cache_path):
            from PIL import Image as _PImage
            arr_north_up = np.array(_PImage.open(cache_path).convert("RGB"), dtype=np.uint8)
        else:
            arr_north_up = _fetch_sat_image(west, south, east, north, cache_path)

        if arr_north_up is None:
            return None

        # Flip to south-up so row 0 = south (matches DEM and xs/ys convention)
        arr = np.flipud(arr_north_up)
        H, W = arr.shape[:2]

        # Geographic position of each mesh vertex
        cos_lat = math.cos(math.radians(lat))
        lon_v   = lon + mx   / (M_PER_DEG * cos_lat)
        lat_v   = lat + my_s / M_PER_DEG

        row_f = (lat_v - south) / (north - south) * H
        col_f = (lon_v - west)  / (east  - west ) * W

        px = _bilinear_sample(arr, row_f, col_f)  # (N_verts, 3)
        return [f"rgb({r},{g},{b})" for r, g, b in px]

    except Exception as exc:
        print(f"[mesh_api] Satellite texture failed: {exc}")
        return None


# ── Endpoint ───────────────────────────────────────────────────────────────

@app.route("/mesh", methods=["POST"])
def make_mesh():
    pay      = request.get_json(force=True)
    lat      = float(pay.get("lat"))
    lon      = float(pay.get("lon"))
    bbox_deg = float(pay.get("bbox_deg", 0.025))

    # Optional extent override (metres, square side).  Default = 2 km popup.
    # Passed by the 3D-View tab when it uses the AABB of ignition points.
    extent_m = float(pay.get("extent_m", POPUP_EXTENT_M))
    extent_m = max(500.0, min(extent_m, 50_000.0))   # clamp 0.5 km – 50 km

    # Optional wind input for air-corrected arrows
    wind_speed = pay.get("wind_speed")
    wind_dir   = pay.get("wind_dir")        # FROM direction (meteorological)

    n = N_CAP
    dem, land, cell_m, bounds = _load_real_dem(lat, lon, n, extent_m=extent_m)
    rows, cols = dem.shape
    b_west, b_south, b_east, b_north = bounds

    fuel_loaded = False
    if wind_speed is not None and wind_dir is not None:
        # Real vegetation cover drives spatial WAF variation in air.py — only
        # needed when we're actually going to compute a wind field.
        fuel_loaded = _load_real_fuel_map(
            land, b_west, b_south, b_east, b_north, n)

    xs_1d = (np.arange(cols) - cols // 2) * cell_m
    ys_1d = (np.arange(rows) - rows // 2) * cell_m

    # Vertical exaggeration
    extent  = max(xs_1d[-1] - xs_1d[0], ys_1d[-1] - ys_1d[0])
    z_all   = dem.ravel()
    z_range = max(float(z_all.max() - z_all.min()), 1.0)
    VEX     = max(2.0, min(8.0, extent / z_range * 0.20))

    # Sub-sample for scatter slide
    XX, YY  = np.meshgrid(xs_1d, ys_1d)
    pts_all = np.column_stack((XX.ravel(), YY.ravel()))
    idxs    = np.arange(len(pts_all)).reshape(rows, cols)

    stride_pts  = max(1, int(math.sqrt(len(pts_all) / N_PTS_CAP)))
    sel_pts     = np.unique(idxs[::stride_pts,  ::stride_pts].ravel())
    px, py_s, pz = pts_all[sel_pts, 0], pts_all[sel_pts, 1], z_all[sel_pts]

    # Mesh sub-sample — keep grid structure for explicit triangulation
    stride_mesh = max(1, int(math.sqrt(len(pts_all) / N_MESH_CAP)))
    row_sub = np.arange(rows)[::stride_mesh]
    col_sub = np.arange(cols)[::stride_mesh]
    n_rows_m, n_cols_m = len(row_sub), len(col_sub)
    sel_mesh    = np.unique(idxs[::stride_mesh, ::stride_mesh].ravel())
    mx, my_s, mz = pts_all[sel_mesh, 0], pts_all[sel_mesh, 1], z_all[sel_mesh]
    mz_vex = (mz - mz.mean()) * VEX

    # Explicit grid-aligned triangles — avoids Delaunay connecting distant vertices
    R_sub, C_sub = np.meshgrid(np.arange(n_rows_m - 1),
                                np.arange(n_cols_m - 1), indexing="ij")
    R_sub = R_sub.ravel();  C_sub = C_sub.ravel()
    v00 = R_sub       * n_cols_m + C_sub
    v01 = R_sub       * n_cols_m + (C_sub + 1)
    v10 = (R_sub + 1) * n_cols_m + C_sub
    v11 = (R_sub + 1) * n_cols_m + (C_sub + 1)
    tri_i = np.concatenate([v00, v01]).tolist()
    tri_j = np.concatenate([v10, v11]).tolist()
    tri_k = np.concatenate([v01, v10]).tolist()

    # Satellite texture — mesh only, via geographic coordinate sampling
    mesh_colors = _fetch_sat_colors(
        lat, lon, b_west, b_south, b_east, b_north, mx, my_s
    )

    base = _layout_base()

    # ── Slide 0: DEM heatmap ──────────────────────────────────────────────
    fig_dem = {
        "data": [{
            "type": "heatmap",
            "z":    dem.tolist(),
            "x":    xs_1d.tolist(),
            "y":    ys_1d.tolist(),
            "colorscale":  "Earth",
            "showscale":   True,
            "hovertemplate": "X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Elev: %{z:.0f} m<extra></extra>",
            "colorbar": _colorbar("m"),
        }],
        "layout": {
            **base,
            "margin": {"l": 55, "r": 20, "t": 40, "b": 45},
            "title": {
                "text": (f"DEM  {rows}×{cols}  "
                         f"({POPUP_EXTENT_M/1000:.1f}km × {POPUP_EXTENT_M/1000:.1f}km)  |  "
                         f"{lat:.4f}N  {lon:.4f}E"),
                "font": {"color": "#f9e2af", "size": 12}, "x": 0.5,
            },
            "xaxis": {"title": "West-East (m)",   "gridcolor": "#2e2e3e",
                      "color": "#6c7086", "zeroline": False},
            "yaxis": {"title": "South-North (m)", "gridcolor": "#2e2e3e",
                      "color": "#6c7086", "zeroline": False},
        },
    }

    # ── Slide 1: Point cloud (elevation colouring only) ──────────────────
    _pts_marker = {
        "size": 2.2,
        "color": pz.tolist(), "colorscale": "Earth",
        "showscale": True, "colorbar": _colorbar("m", thickness=10),
    }
    fig_pts = {
        "data": [{
            "type": "scatter3d",
            "x": px.tolist(), "y": py_s.tolist(), "z": pz.tolist(),
            "mode": "markers",
            "marker": _pts_marker,
            "hovertemplate": "X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Z: %{z:.0f} m<extra></extra>",
            "name": "Point Cloud",
        }],
        "layout": {
            **base,
            "title": {
                "text": f"Point Cloud  {len(px):,} pts",
                "font": {"color": "#89b4fa", "size": 12}, "x": 0.5,
            },
            "scene": _scene("Z (m)"),
        },
    }

    # ── Slide 2: Triangulated mesh with explicit grid connectivity ────────
    # i,j,k are pre-computed from the regular subgrid — no Delaunay re-triangulation,
    # so no spurious long-distance edges.
    # ambient=1 / diffuse=0 → satellite colours shown without directional shadows.
    n_tris = len(tri_i) // 2
    _tri_trace = {
        "type": "mesh3d",
        "x": mx.tolist(), "y": my_s.tolist(), "z": mz_vex.tolist(),
        "i": tri_i, "j": tri_j, "k": tri_k,
        "flatshading":  False,
        "lighting": {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0},
        "hovertemplate": "X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Elev: %{customdata:.0f} m<extra></extra>",
        "customdata": mz.tolist(),
        "name": "Terrain",
    }
    if mesh_colors:
        _tri_trace["vertexcolor"] = mesh_colors
        _tri_trace["showscale"]   = False
    else:
        _tri_trace.update({"intensity": mz.tolist(), "colorscale": "Earth",
                           "showscale": True, "colorbar": _colorbar("Elev (m)", thickness=10)})
    fig_tri = {
        "data": [_tri_trace],
        "layout": {
            **base,
            "title": {
                "text": f"Terrain Mesh  {len(mx):,} verts  {n_tris*2:,} tris  (VEX {VEX:.1f}x)",
                "font": {"color": "#a6e3a1", "size": 12}, "x": 0.5,
            },
            "scene": _scene(f"Z x{VEX:.1f} (m)"),
        },
    }

    # ── 2. Air-corrected wind field ───────────────────────────────────────
    step_r = max(1, rows // N_ARROW)
    step_c = max(1, cols // N_ARROW)
    YY_s, XX_s = np.meshgrid(ys_1d[::step_r], xs_1d[::step_c], indexing="ij")
    xs_arr = XX_s.ravel().tolist()
    ys_arr = YY_s.ravel().tolist()

    # Per-arrow elevation for correct 3D cone placement
    row_idxs = np.arange(rows)[::step_r]
    col_idxs = np.arange(cols)[::step_c]
    RR, CC   = np.meshgrid(row_idxs, col_idxs, indexing="ij")
    arrow_elev = dem[RR.ravel(), CC.ravel()].astype(float)
    arrow_elev_vex = (arrow_elev - float(dem.mean())) * VEX

    wind_grid: dict = {
        "xs":       xs_arr,
        "ys":       ys_arr,
        "zs_real":  arrow_elev.tolist(),        # real elevation for slide-1 cones
        "zs_vex":   arrow_elev_vex.tolist(),    # VEX-adjusted z for slide-2 cones
        "dem_mean": float(dem.mean()),
        "vex":      round(VEX, 2),
        "has_air":  False,
        "has_fuel": fuel_loaded,
    }

    if wind_speed is not None and wind_dir is not None:
        try:
            from air.air import compute_wind_field, compute_vertical_wind
            spd      = float(wind_speed)
            from_deg = float(wind_dir)
            push_deg = (from_deg + 180.0) % 360.0
            land.set_wind(spd, push_deg)
            # Pass a config proxy with the popup's actual cell size
            import types as _t
            mesh_cfg = _t.SimpleNamespace(**vars(config))
            mesh_cfg.CELL_SIZE_METERS = cell_m
            U, V = compute_wind_field(land, mesh_cfg)
            # Vertical component (anabatic/katabatic + divergence compensation)
            # for true-3D cones; no fire state in the static popup.
            W = compute_vertical_wind(land, U, V, cell_size_m=cell_m)
            # Subsample to arrow grid
            U_s = U[::step_r, ::step_c].ravel()
            V_s = V[::step_r, ::step_c].ravel()
            W_s = W[::step_r, ::step_c].ravel()
            wind_grid["us"]      = U_s.tolist()
            wind_grid["vs"]      = V_s.tolist()
            wind_grid["ws"]      = W_s.tolist()
            wind_grid["has_air"] = True
        except Exception as exc:
            wind_grid["air_error"] = str(exc)

    return jsonify({
        "plotly_dem": json.dumps(fig_dem),
        "plotly_pts": json.dumps(fig_pts),
        "plotly_tri": json.dumps(fig_tri),
        "wind_grid":  wind_grid,
        "meta": {
            "center":      [lat, lon],
            "bounds":      {"west": b_west, "south": b_south,
                            "east": b_east, "north": b_north},
            "dem_shape":   [rows, cols],
            "n_pts":       int(len(px)),
            "n_mesh":      int(len(mx)),
            "n_mesh_rows": n_rows_m,
            "n_mesh_cols": n_cols_m,
            "cell_m":      round(cell_m, 2),
            "vex":         round(VEX, 2),
        },
    })


if __name__ == "__main__":
    print("mesh_api: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

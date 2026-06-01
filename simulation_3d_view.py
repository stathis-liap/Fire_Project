"""
simulation_3d_view.py
=====================
Live 3D terrain + fire-simulation overlay for Project WILSON.

Importable API  (called from server.py):
    compute_aabb(ignition_latlons, offset_km)  → bbox dict
    build_sim3d_npz(bbox, elevation, fuel_map, fuel_names,
                    geo_grid_bounds, state, wind_u, wind_v, cell_m,
                    ignition_latlon_list, n, live_state_path, output_dir)
                                                → (npz_path, extract_info)
    launch_viewer(npz_path)                    → subprocess.Popen
    write_live_state(path, state, r0, r1, c0, c1)

Script mode  (launched as subprocess by launch_viewer):
    python simulation_3d_view.py  payload.npz

Visual layers (bottom → top):
    1. Terrain mesh   – satellite texture (zoom 14) or elevation colormap
    2. Fuel overlay   – semi-transparent per-fuel colour mesh (static)
    3. Fire overlay   – live-updated burn state RGBA
    4. Wind arrows    – cyan glyphs at coarse resolution
    5. Ignition stars – yellow spheres

Live updates:
    server.py writes the current fire-state subset to a .npy file once per
    second.  The viewer polls that file's mtime and redraws the fire overlay
    whenever it changes.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
M_PER_DEG  = 111_320.0
SAT_ZOOM   = 14          # ~9.5 m/px at equator (14 tiles → fast fetch)
NPZ_NAME   = "_sim3d_payload.npz"
LIVE_NAME  = "_sim3d_live_state.npy"
Z_EXG      = 1.5         # vertical exaggeration (matches fire_viewer_3d)


# ── AABB computation ───────────────────────────────────────────────────────────

def compute_aabb(ignition_latlons: list[tuple[float, float]],
                 offset_km: float = 5.0) -> dict | None:
    """
    Return the Axis-Aligned Bounding Box of *ignition_latlons* expanded by
    *offset_km* on every side.

    The offset is at least 3 km so the viewer always shows meaningful terrain
    around the ignition cluster.

    Returns a dict with:
        lat_min, lat_max, lon_min, lon_max,
        center_lat, center_lon, width_km, height_km
    or None when *ignition_latlons* is empty.
    """
    if not ignition_latlons:
        return None

    lats = [p[0] for p in ignition_latlons]
    lons = [p[1] for p in ignition_latlons]

    lat_c   = (min(lats) + max(lats)) / 2.0
    lon_c   = (min(lons) + max(lons)) / 2.0
    off     = max(offset_km, 3.0)

    lat_d   = off / 111.32
    lon_d   = off / (111.32 * math.cos(math.radians(lat_c)) + 1e-9)

    lat_min = min(lats) - lat_d
    lat_max = max(lats) + lat_d
    lon_min = min(lons) - lon_d
    lon_max = max(lons) + lon_d

    h_km = (lat_max - lat_min) * 111.32
    w_km = (lon_max - lon_min) * 111.32 * math.cos(math.radians(lat_c))

    return {
        "lat_min":    lat_min,    "lat_max":    lat_max,
        "lon_min":    lon_min,    "lon_max":    lon_max,
        "center_lat": lat_c,      "center_lon": lon_c,
        "width_km":   round(w_km, 2),
        "height_km":  round(h_km, 2),
    }


# ── Satellite tile helpers (self-contained copy from mesh_api.py) ──────────────

def _deg_to_tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    n     = 1 << zoom
    x     = int((lon_deg + 180.0) / 360.0 * n)
    lat_r = math.radians(lat_deg)
    y     = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile_nw_deg(x: int, y: int, zoom: int) -> tuple[float, float]:
    n   = 1 << zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def _fetch_sat_for_bbox(west: float, south: float,
                        east: float, north: float,
                        cache_path: str,
                        zoom: int = SAT_ZOOM) -> "np.ndarray | None":
    """
    Stitch ESRI World Imagery tiles for (west, south, east, north) at *zoom*.
    Saves a JPEG to *cache_path* and returns a north-up uint8 (H, W, 3) array.
    Returns None on total failure.
    """
    try:
        from PIL import Image as _PImg
        import requests as _req
        from concurrent.futures import ThreadPoolExecutor, as_completed as _asc
        import io as _io

        TILE    = 256
        SAT_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                   "World_Imagery/MapServer/tile/{z}/{y}/{x}")

        x0, y0 = _deg_to_tile(north, west,  zoom)
        x1, y1 = _deg_to_tile(south, east,  zoom)
        x1 = max(x1, x0);  y1 = max(y1, y0)
        nx, ny  = x1 - x0 + 1, y1 - y0 + 1

        print(f"[sim3d] Satellite z{zoom}: {nx}×{ny}={nx*ny} tiles …")

        def _get(tx_ty: tuple) -> tuple:
            tx, ty = tx_ty
            url    = SAT_URL.format(z=zoom, y=ty, x=tx)
            for _ in range(3):
                try:
                    r = _req.get(url, timeout=20)
                    r.raise_for_status()
                    img = _PImg.open(_io.BytesIO(r.content)).convert("RGB")
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
            print("[sim3d] Satellite: all tiles failed")
            return None

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

        _PImg.fromarray(cropped).save(cache_path, "JPEG", quality=90)
        print(f"[sim3d] Satellite {cropped.shape[1]}×{cropped.shape[0]} px  "
              f"(failed {failed}/{len(jobs)} tiles)")
        return cropped

    except Exception as exc:
        print(f"[sim3d] Satellite fetch error: {exc}")
        return None


# ── NPZ builder ────────────────────────────────────────────────────────────────

def build_sim3d_npz(
    bbox:                 dict,
    elevation:            "np.ndarray",   # (rows, cols) float32 — full sim DEM
    fuel_map:             "np.ndarray",   # (rows, cols) int     — full sim fuel
    fuel_names:           list[str],
    geo_grid_bounds:      tuple[float, float, float, float],  # lat_min,lat_max,lon_min,lon_max
    state_array:          "np.ndarray",   # (rows, cols) uint8   — current fire state
    wind_u:               "np.ndarray",   # (rows, cols) float32 or scalar
    wind_v:               "np.ndarray",   # (rows, cols) float32 or scalar
    cell_m:               float,
    ignition_latlon_list: list[tuple[float, float]],
    n:                    int  = 150,
    live_state_path:      "str | None" = None,
    output_dir:           "str | None" = None,
) -> tuple["str | None", "tuple | None"]:
    """
    Extract the AABB subset from the simulation arrays, resize to *n*×*n*,
    fetch satellite imagery, and write a self-contained NPZ.

    All input arrays are thread-safe copies (caller's responsibility).

    Returns ``(npz_path, extract_info)`` where
    ``extract_info = (r0, r1, c0, c1, n)`` are the row/col slice indices
    in the original simulation grid — needed by ``write_live_state``.

    Returns ``(None, None)`` on failure.
    """
    if output_dir is None:
        output_dir = ROOT_DIR

    rows, cols = elevation.shape
    lat_min_s, lat_max_s, lon_min_s, lon_max_s = geo_grid_bounds

    # ── Map AABB corners to simulation grid rows/cols ─────────────────────────
    # Row 0 = lat_min (south-up), consistent with _state_to_geojson in server.py
    def _frac(lat: float, lon: float) -> tuple[float, float]:
        fr = (lat - lat_min_s) / max(lat_max_s - lat_min_s, 1e-9)
        fc = (lon - lon_min_s) / max(lon_max_s - lon_min_s, 1e-9)
        return fr, fc

    fr0, fc0 = _frac(bbox["lat_min"], bbox["lon_min"])
    fr1, fc1 = _frac(bbox["lat_max"], bbox["lon_max"])

    r0 = int(max(0,    min(rows, fr0 * rows)))
    r1 = int(max(0,    min(rows, fr1 * rows)))
    c0 = int(max(0,    min(cols, fc0 * cols)))
    c1 = int(max(0,    min(cols, fc1 * cols)))

    # Fallback: use entire grid when AABB is outside simulation bounds
    if r1 - r0 < 4 or c1 - c0 < 4:
        r0, r1, c0, c1 = 0, rows, 0, cols
        print("[sim3d] AABB outside sim bounds — using full grid")

    print(f"[sim3d] Sim-grid slice [{r0}:{r1}, {c0}:{c1}]  "
          f"({r1-r0}×{c1-c0}) → {n}×{n} display")

    # ── Extract and resize ────────────────────────────────────────────────────
    try:
        from scipy.ndimage import zoom as _sz
    except ImportError:
        print("[sim3d] scipy not found — cannot resize arrays")
        return None, None

    sub_r, sub_c = r1 - r0, c1 - c0
    zr, zc = n / sub_r, n / sub_c

    elev_n  = _sz(elevation[r0:r1, c0:c1].astype(np.float32), (zr, zc), order=1)
    fuel_n  = np.round(_sz(fuel_map[r0:r1, c0:c1].astype(float), (zr, zc), order=0)).astype(np.int16)
    state_n = np.round(_sz(state_array[r0:r1, c0:c1].astype(float), (zr, zc), order=0)).astype(np.uint8)

    wu_sub = wind_u[r0:r1, c0:c1] if wind_u.ndim == 2 else np.full((sub_r, sub_c), float(wind_u))
    wv_sub = wind_v[r0:r1, c0:c1] if wind_v.ndim == 2 else np.full((sub_r, sub_c), float(wind_v))
    wu_n   = _sz(wu_sub.astype(np.float32), (zr, zc), order=1)
    wv_n   = _sz(wv_sub.astype(np.float32), (zr, zc), order=1)

    # ── Physical cell size for the 3D display grid ────────────────────────────
    lat_ext_m = (bbox["lat_max"] - bbox["lat_min"]) * M_PER_DEG
    lon_ext_m = (bbox["lon_max"] - bbox["lon_min"]) * M_PER_DEG * math.cos(
                    math.radians(bbox["center_lat"]))
    cell_3d = max(lat_ext_m, lon_ext_m) / n

    # ── Ignition markers mapped to 3D display grid ────────────────────────────
    ign_rcs_3d: list[list[int]] = []
    for lat_i, lon_i in ignition_latlon_list:
        lat_f = (lat_i - bbox["lat_min"]) / max(bbox["lat_max"] - bbox["lat_min"], 1e-9)
        lon_f = (lon_i - bbox["lon_min"]) / max(bbox["lon_max"] - bbox["lon_min"], 1e-9)
        if -0.05 <= lat_f <= 1.05 and -0.05 <= lon_f <= 1.05:
            r3d = int(min(n - 1, max(0, lat_f * n)))
            c3d = int(min(n - 1, max(0, lon_f * n)))
            ign_rcs_3d.append([r3d, c3d])
    if not ign_rcs_3d:
        ign_rcs_3d = [[n // 2, n // 2]]

    # ── Satellite texture ──────────────────────────────────────────────────────
    sat_tag  = f"sim3d_{bbox['center_lat']:.4f}_{bbox['center_lon']:.4f}_z{SAT_ZOOM}"
    sat_path = os.path.join(output_dir, f"sat_{sat_tag}.jpg")

    if not os.path.exists(sat_path):
        _fetch_sat_for_bbox(
            bbox["lon_min"], bbox["lat_min"],
            bbox["lon_max"], bbox["lat_max"],
            sat_path, zoom=SAT_ZOOM,
        )

    # ── Write initial live state ───────────────────────────────────────────────
    if live_state_path:
        np.save(live_state_path, state_array[r0:r1, c0:c1].astype(np.uint8))

    # ── Pack NPZ ──────────────────────────────────────────────────────────────
    npz_path = os.path.join(output_dir, NPZ_NAME)

    payload: dict[str, object] = dict(
        elevation    = elev_n.astype(np.float32),
        snapshots    = state_n[np.newaxis],           # (1, n, n) — initial frame
        ignition_rc  = np.array([n // 2, n // 2], dtype=int),
        ignition_rcs = np.array(ign_rcs_3d,       dtype=int),
        cell_size_m  = np.float32(cell_3d),
        wind_u       = wu_n,
        wind_v       = wv_n,
        fuel_map     = fuel_n,
        fuel_names   = np.array(fuel_names, dtype=object),
        frame_dt_min = np.float32(1.0),
    )

    if sat_path and os.path.exists(sat_path):
        payload["texture_path"] = np.frombuffer(sat_path.encode(), dtype=np.uint8)

    if live_state_path:
        payload["live_state_path"] = np.frombuffer(live_state_path.encode(), dtype=np.uint8)
        payload["live_r0"] = np.int32(r0)
        payload["live_r1"] = np.int32(r1)
        payload["live_c0"] = np.int32(c0)
        payload["live_c1"] = np.int32(c1)
        payload["live_n"]  = np.int32(n)

    np.savez_compressed(npz_path, **payload)
    print(f"[sim3d] NPZ → {npz_path}  "
          f"({n}×{n} cells, cell={cell_3d:.0f} m, "
          f"{bbox['width_km']:.1f}×{bbox['height_km']:.1f} km)")

    return npz_path, (r0, r1, c0, c1, n)


# ── Launcher ───────────────────────────────────────────────────────────────────

def launch_viewer(npz_path: str) -> "subprocess.Popen":
    """Launch this script as a subprocess (the viewer entry point)."""
    script = os.path.abspath(__file__)
    # Force UTF-8 output so emoji/Unicode in print statements don't crash on
    # Windows terminals that use cp1252/cp1253.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-u", script, npz_path],
        cwd = ROOT_DIR,
        env = env,
    )
    print(f"[sim3d] Viewer PID {proc.pid}")
    return proc


# ── Live-state writer (called from server.py on every broadcast frame) ─────────

def write_live_state(path: str,
                     state: "np.ndarray",
                     r0: int, r1: int,
                     c0: int, c1: int) -> None:
    """Write the simulation-state subset relevant to the 3D view."""
    try:
        np.save(path, state[r0:r1, c0:c1].astype(np.uint8))
    except Exception as exc:
        print(f"[sim3d] write_live_state: {exc}")


# ── Script / viewer entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulation_3d_view.py payload.npz")
        sys.exit(1)

    try:
        import pyvista as pv
        from PIL import Image as _PImage
        from scipy.ndimage import zoom as _sz, binary_erosion as _erode
    except ImportError as exc:
        print(f"[sim3d] Missing dependency: {exc}")
        sys.exit(1)

    sys.path.insert(0, ROOT_DIR)
    from core.fuels import fuel_color_rgba as _fcr

    # ── Load NPZ ──────────────────────────────────────────────────────────────
    data       = np.load(sys.argv[1], allow_pickle=True)
    elevation  = data["elevation"].astype(float)                  # (n, n)
    snapshots  = data["snapshots"].astype(np.uint8)               # (1, n, n)
    cell_size_m = float(data["cell_size_m"])

    ign_rc  = data.get("ignition_rc", np.array([0, 0]))
    ign_rcs = [tuple(int(x) for x in rc) for rc in data["ignition_rcs"]] \
              if "ignition_rcs" in data else [(int(ign_rc[0]), int(ign_rc[1]))]

    fuel_map_raw   = data["fuel_map"].astype(int)  if "fuel_map"   in data else None
    fuel_names_raw = data["fuel_names"].tolist()   if "fuel_names" in data else []

    wu = data["wind_u"] if "wind_u" in data else np.zeros_like(elevation)
    wv = data["wind_v"] if "wind_v" in data else np.zeros_like(elevation)
    if wu.ndim == 0: wu = np.full_like(elevation, float(wu))
    if wv.ndim == 0: wv = np.full_like(elevation, float(wv))

    texture_path = data["texture_path"].tobytes().decode() \
                   if "texture_path" in data and len(data["texture_path"]) > 0 \
                   else ""

    live_state_path = data["live_state_path"].tobytes().decode() \
                      if "live_state_path" in data and len(data["live_state_path"]) > 0 \
                      else ""
    live_n = int(data["live_n"]) if "live_n" in data else elevation.shape[0]

    rows, cols = elevation.shape
    print(f"[sim3d] Terrain {rows}x{cols}  |  cell={cell_size_m:.0f} m  |  "
          f"live={'yes' if live_state_path else 'no'}")

    # ── Build coordinate grids ─────────────────────────────────────────────────
    xx, yy = np.meshgrid(
        np.arange(cols, dtype=float) * cell_size_m,
        np.arange(rows, dtype=float) * cell_size_m,
    )
    zz = elevation * Z_EXG

    # ── PyVista meshes ─────────────────────────────────────────────────────────
    terrain_mesh = pv.StructuredGrid(xx, yy, zz)
    fuel_mesh    = pv.StructuredGrid(xx, yy, zz + 1.0)
    fire_mesh    = pv.StructuredGrid(xx, yy, zz + 3.0)

    # ── Satellite texture ──────────────────────────────────────────────────────
    texture_loaded = False
    if texture_path and os.path.exists(texture_path):
        try:
            img = _PImage.open(texture_path).convert("RGB")
            img = img.resize((cols, rows), _PImage.LANCZOS)
            rgb = np.flipud(np.array(img)).reshape(-1, 3).astype(np.uint8)
            terrain_mesh.point_data["RGB"] = rgb
            texture_loaded = True
            print(f"[sim3d] Satellite texture loaded ({cols}×{rows})")
        except Exception as e:
            print(f"[sim3d] Texture failed: {e}")

    # ── Fuel overlay ───────────────────────────────────────────────────────────
    if fuel_map_raw is not None:
        flat_fuel = fuel_map_raw.ravel()
        fuel_rgba = np.zeros((len(flat_fuel), 4), dtype=np.uint8)
        for fi, fname in enumerate(fuel_names_raw):
            mask = flat_fuel == fi
            if mask.any():
                fuel_rgba[mask] = _fcr(fname)
        fuel_mesh.point_data["fuel_rgba"] = fuel_rgba
    else:
        fuel_mesh.point_data["fuel_rgba"] = np.zeros((fuel_mesh.n_points, 4), dtype=np.uint8)

    # ── Fire overlay helpers ───────────────────────────────────────────────────
    def _state_to_rgba(state2d: "np.ndarray") -> "np.ndarray":
        """
        Convert a (R, C) fire-state array (0/1/2) to (R*C, 4) RGBA.
        Mirrors the rendering logic of fire_viewer_3d._render_frame.
        """
        c = np.zeros((state2d.size, 4), dtype=np.uint8)

        active2d   = state2d == 1
        consumed2d = state2d > 0
        if active2d.any():
            interior2d = active2d & _erode(consumed2d, structure=np.ones((3, 3)))
            front2d    = active2d & ~interior2d
            c[front2d.ravel()]    = [255, 160, 10,  250]   # bright orange front
            c[interior2d.ravel()] = [60,  15,  5,   230]   # dark ember interior

        burned2d = state2d == 2
        if burned2d.any():
            c[burned2d.ravel()] = [55, 25, 10, 210]        # charcoal scar

        return c

    # Initial fire state from snapshot
    fire_mesh.point_data["fire_rgba"] = _state_to_rgba(snapshots[0])

    # ── Wind arrows ────────────────────────────────────────────────────────────
    q_step = max(1, min(rows, cols) // 20)
    qi = np.arange(0, rows, q_step)
    qj = np.arange(0, cols, q_step)
    qI, qJ = np.meshgrid(qi, qj, indexing="ij")

    vec_pts = np.column_stack((
        (qJ * cell_size_m).ravel(),
        (qI * cell_size_m).ravel(),
        (zz[qI, qJ] + 80.0).ravel(),
    ))
    q_u = wu[qI, qJ].ravel()
    q_v = wv[qI, qJ].ravel()
    vec_dir = np.column_stack((q_u, q_v, np.zeros_like(q_u)))
    vec_mag = np.linalg.norm(vec_dir[:, :2], axis=1) + 0.1

    wind_cloud = pv.PolyData(vec_pts)
    wind_cloud["wind_vectors"]   = vec_dir
    wind_cloud["wind_magnitude"] = vec_mag
    arrows = wind_cloud.glyph(
        orient="wind_vectors",
        factor=cell_size_m * 2,
        scale="wind_magnitude",
    )

    # ── Ignition markers ───────────────────────────────────────────────────────
    ign_star_pts = []
    for r_i, c_i in ign_rcs:
        r_i = min(r_i, rows - 1);  c_i = min(c_i, cols - 1)
        ign_star_pts.append([c_i * cell_size_m, r_i * cell_size_m, zz[r_i, c_i] + 120.0])

    # ── Plotter ────────────────────────────────────────────────────────────────
    pl = pv.Plotter(title="Project WILSON — 3D Simulation View",
                    window_size=[1400, 900])
    pl.set_background("#0d0d1a")

    if texture_loaded:
        pl.add_mesh(terrain_mesh, scalars="RGB", rgb=True,
                    lighting=True, smooth_shading=True)
    else:
        pl.add_mesh(terrain_mesh, cmap="terrain",
                    show_scalar_bar=False, lighting=True, smooth_shading=True)

    fuel_alpha = 0.55 if texture_loaded else 0.85
    pl.add_mesh(fuel_mesh, scalars="fuel_rgba", rgba=True,
                opacity=fuel_alpha, show_scalar_bar=False,
                name="fuel_overlay", lighting=False)

    pl.add_mesh(fire_mesh, scalars="fire_rgba", rgba=True,
                show_scalar_bar=False, name="fire", lighting=False)

    pl.add_mesh(arrows, color="cyan", opacity=0.55,
                show_scalar_bar=False, name="wind")

    for idx, pt in enumerate(ign_star_pts):
        sphere = pv.Sphere(radius=cell_size_m * 1.8, center=pt)
        pl.add_mesh(sphere, color="yellow", name=f"ign_{idx}")
    if ign_star_pts:
        label_pts = np.array(ign_star_pts)
        label_pts[:, 2] += cell_size_m * 3
        pl.add_point_labels(
            label_pts,
            [f"★ Seed {i+1}" if len(ign_star_pts) > 1 else "★ Ignition"
             for i in range(len(ign_star_pts))],
            font_size=11, text_color="yellow",
            shape_opacity=0.0, always_visible=True,
        )

    # ── Camera ─────────────────────────────────────────────────────────────────
    pl.camera_position = "iso"
    pl.camera.elevation = 35
    pl.camera.azimuth   = -45

    # ── HUD text ───────────────────────────────────────────────────────────────
    pl.add_text(
        "Left-drag: rotate  |  Scroll: zoom  |  Q: quit",
        position="upper_left", font_size=9, color="#a6adc8",
    )
    live_label_color = "#a6e3a1" if live_state_path else "#585b70"
    live_label_text  = "[ LIVE ] updating from simulation" \
                       if live_state_path else "[ static snapshot ]"
    pl.add_text(live_label_text, position="lower_left",
                font_size=9, color=live_label_color, name="live_status")

    # ── Live-update state ──────────────────────────────────────────────────────
    _lv = {
        "mtime":      0.0,
        "last_check": 0.0,
        "active_ha":  0.0,
        "burned_ha":  0.0,
    }
    _cell_ha = cell_size_m ** 2 / 10_000.0

    def _refresh_fire() -> None:
        """Reload live state file when it has changed, update fire overlay."""
        if not live_state_path or not os.path.exists(live_state_path):
            return
        try:
            mtime = os.path.getmtime(live_state_path)
            if mtime <= _lv["mtime"]:
                return
            _lv["mtime"] = mtime

            state_sub = np.load(live_state_path)          # (sub_r, sub_c) uint8
            sub_r, sub_c = state_sub.shape
            if sub_r < 1 or sub_c < 1:
                return

            # Resize to display grid
            zr = rows / sub_r;  zc = cols / sub_c
            state_disp = np.round(_sz(state_sub.astype(float), (zr, zc),
                                       order=0)).astype(np.uint8)

            fire_mesh.point_data["fire_rgba"] = _state_to_rgba(state_disp)

            # Update HUD
            _lv["active_ha"] = float((state_sub == 1).sum()) * _cell_ha
            _lv["burned_ha"] = float((state_sub == 2).sum()) * _cell_ha
            pl.add_text(
                f"FIRE {_lv['active_ha']:.1f} ha active  |  "
                f"BURNED {_lv['burned_ha']:.1f} ha",
                position="lower_edge", font_size=11, color="white",
                name="burn_stats",
            )
        except Exception as exc:
            print(f"[sim3d] Refresh error: {exc}")

    # ── Main loop ──────────────────────────────────────────────────────────────
    pl.show(interactive_update=True, auto_close=False)
    _refresh_fire()   # draw initial state immediately

    while pl.iren.initialized:
        now = time.time()
        if now - _lv["last_check"] >= 0.5:
            _lv["last_check"] = now
            _refresh_fire()
        pl.update()

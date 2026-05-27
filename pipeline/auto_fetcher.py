"""auto_fetcher.py — Real-World Data Acquisition
================================================
Fetches all geospatial inputs needed to run the Hellenic Wildfire Digital Twin
on real terrain instead of synthetic hills:

  1. Drone telemetry     → GPS fix (Lat, Lon) from a drone log file
  2. High-res terrain DEM  (two sources, in priority order):
       a) AWS/Mapzen Terrarium elevation tiles — ~9.5 m @ zoom 14, free, no key
       b) Copernicus COP30 via OpenTopography  — ~30 m, requires API key (fallback)
  3. ESRI World Imagery  → High-res satellite texture (for visualizer_3d.py)
  4. CORINE Land Cover    → EU vegetation/land-use map for fuel-type assignment

API key (OpenTopography / Copernicus COP30):  57314bc7ed85882904a7485d77c0dbe5
Register or renew at:  https://portal.opentopography.org/requestApiKey
"""

import io
import math
import os
import requests
import numpy as np

# Default Copernicus/OpenTopography API key
_DEFAULT_API_KEY = "57314bc7ed85882904a7485d77c0dbe5"


# ──────────────────────────────────────────────────────────────────────────────
# Tile helpers (Web-Mercator slippy map convention)
# ──────────────────────────────────────────────────────────────────────────────

def _deg_to_tile(lat_deg: float, lon_deg: float, zoom: int):
    """Return the (x, y) Slippy-map tile index for a given lat/lon."""
    lat_r = math.radians(lat_deg)
    n = 1 << zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_deg(x: int, y: int, zoom: int):
    """Return the NW corner (lat, lon) of tile (x, y)."""
    n = 1 << zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Drone telemetry parser


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Drone telemetry parser
# ──────────────────────────────────────────────────────────────────────────────

def read_drone_telemetry(filepath="drone_telemetry.txt"):
    """
    Parse a drone telemetry text file with the format:
        Lat: 38.894939
        Lon: 23.401405

    Returns (lat, lon) as floats.
    """
    telemetry = {}
    print(f"[Fetcher] Reading telemetry from: {filepath} ...")
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) == 2:
                telemetry[parts[0].strip()] = float(parts[1].strip())
    return telemetry["Lat"], telemetry["Lon"]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  High-res terrain — AWS/Mapzen Terrarium elevation tiles  (~19 m @ z=13)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_terrain_tiles(lat: float, lon: float, buffer: float = 0.25,
                        zoom: int = 14, output_dir: str = ".") -> "str | None":
    """
    Fetch a high-resolution DEM from the AWS/Mapzen Terrarium elevation tile
    service and save it as a GeoTIFF.  Uses parallel HTTP fetching for speed.

    Source  : https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
    Encoding: elevation = (R×256 + G + B/256) − 32 768  (metres)

    Practical zoom choices for a ±0.25° bbox (~55 km):
        zoom 13 → ~19 m/pixel,  ~130 tiles, ~10 s
        zoom 14 → ~10 m/pixel,  ~530 tiles, ~35 s  ← default (best balance)
        zoom 15 → ~5 m/pixel,  ~2100 tiles, ~80 s  ← high detail, slower

    No API key required.  Result is cached as terrain_<lat>_<lon>_<buf>_z<zoom>.tif.
    The GeoTIFF is EPSG:4326 south-up (row 0 = lat_min) to match rasterio.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image
    import rasterio
    from rasterio.transform import from_bounds

    tag = f"{lat:.3f}_{lon:.3f}_{buffer:.3f}_z{zoom}"
    filename = os.path.join(output_dir, f"terrain_{tag}.tif")

    if os.path.exists(filename):
        print(f"[Fetcher] Using cached high-res DEM '{filename}'.")
        return filename

    south = lat - buffer
    north = lat + buffer
    west  = lon - buffer
    east  = lon + buffer

    # Tile index range — y increases southward in slippy-map tiles
    x_min, y_min = _deg_to_tile(north, west, zoom)
    x_max, y_max = _deg_to_tile(south, east, zoom)

    n_tx = x_max - x_min + 1
    n_ty = y_max - y_min + 1
    total_tiles = n_tx * n_ty
    res_m = 111_320 * 360 / (1 << zoom) / 256

    print(f"\n[Fetcher] Terrarium tiles (zoom={zoom}, ~{res_m:.0f} m/px, "
          f"{n_tx}×{n_ty}={total_tiles} tiles)")
    print(f"[Fetcher] Bbox: S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")

    tile_size = 256
    canvas = np.zeros((n_ty * tile_size, n_tx * tile_size, 3), dtype=np.uint8)
    base_url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"

    def _fetch_tile(args):
        tx, ty = args
        url = f"{base_url}/{zoom}/{tx}/{ty}.png"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return tx, ty, np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"),
                                 dtype=np.uint8)

    jobs = [(tx, ty)
            for ty in range(y_min, y_max + 1)
            for tx in range(x_min, x_max + 1)]

    n_workers = min(32, len(jobs))
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fetch_tile, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                tx, ty, img = fut.result()
                row = (ty - y_min) * tile_size
                col = (tx - x_min) * tile_size
                canvas[row:row + tile_size, col:col + tile_size] = img
                done += 1
                if done % 20 == 0:
                    print(f"[Fetcher]  {done}/{total_tiles} tiles …")
            except Exception as exc:
                tx, ty = futures[fut]
                print(f"[Fetcher] WARNING tile {zoom}/{tx}/{ty}: {exc}")

    # Decode Terrarium: elevation = R*256 + G + B/256 - 32768
    R = canvas[:, :, 0].astype(np.float32)
    G = canvas[:, :, 1].astype(np.float32)
    B = canvas[:, :, 2].astype(np.float32)
    elevation = R * 256.0 + G + B / 256.0 - 32768.0

    # Canvas extents
    nw_lat, nw_lon = _tile_to_deg(x_min,     y_min,     zoom)
    se_lat, se_lon = _tile_to_deg(x_max + 1, y_max + 1, zoom)
    total_h, total_w = elevation.shape

    # Crop to exact requested bbox
    r0 = max(0, int((nw_lat - north) / (nw_lat - se_lat) * total_h))
    r1 = min(total_h, int((nw_lat - south) / (nw_lat - se_lat) * total_h))
    c0 = max(0, int((west - nw_lon) / (se_lon - nw_lon) * total_w))
    c1 = min(total_w, int((east - nw_lon) / (se_lon - nw_lon) * total_w))
    elevation = elevation[r0:r1, c0:c1]

    # Save as standard north-up GeoTIFF (row 0 = north).
    # landscape.py applies np.flipud when loading, converting to south-up.
    # Do NOT pre-flip here — that would cause a double-flip = upside-down terrain.
    h, w = elevation.shape
    transform = from_bounds(west, south, east, north, w, h)

    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(
        filename, "w", driver="GTiff",
        height=h, width=w, count=1,
        dtype=np.float32, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(elevation.astype(np.float32), 1)

    actual_res = (east - west) * 111_320 / w
    print(f"[Fetcher] SUCCESS — terrain saved: '{filename}' "
          f"({w}×{h} px, ~{actual_res:.0f} m/cell)")
    return filename


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Copernicus COP30 DEM  (OpenTopography API) — fallback if tiles fail
# ──────────────────────────────────────────────────────────────────────────────

def fetch_terrain_from_api(lat, lon, api_key=_DEFAULT_API_KEY,
                            output_dir=".", buffer=0.05, force=False):
    """
    Fetch a high-resolution DEM GeoTIFF for a bounding box centred on (lat, lon).

    Strategy (in priority order):
      1. AWS/Mapzen Terrarium tiles at zoom 14  → ~9.5 m/pixel, free, no key.
      2. Copernicus COP30 via OpenTopography    → ~30 m/pixel, API key required.
      3. Any cached .tif already on disk.

    ``force=True`` re-downloads even if a cached file exists.
    """
    tag      = f"{lat:.3f}_{lon:.3f}_{buffer:.3f}"
    filename = os.path.join(output_dir, f"terrain_{tag}.tif")

    if not force and os.path.exists(filename):
        print(f"[Fetcher] Using cached DEM '{filename}' (pass force=True to re-fetch).")
        return filename

    print(f"\n[Fetcher] Target: Lat={lat:.5f}, Lon={lon:.5f}")

    # ── 1. Try Terrarium high-res tiles (~9.5 m) ──────────────────────────────
    try:
        result = fetch_terrain_tiles(lat, lon, buffer=buffer, zoom=14,
                                     output_dir=output_dir)
        if result and os.path.exists(result):
            return result
    except Exception as exc:
        print(f"[Fetcher] Terrarium tiles failed: {exc}  — falling back to COP30")

    # ── 2. Fall back to Copernicus COP30 via OpenTopography (~30 m) ───────────
    west, east   = lon - buffer, lon + buffer
    south, north = lat - buffer, lat + buffer

    print(f"[Fetcher] Bounding box: S={south:.4f} N={north:.4f} "
          f"W={west:.4f} E={east:.4f}")

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype":      "COP30",
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": "GTiff",
        "API_Key":      api_key,
    }

    print("[Fetcher] Requesting Copernicus COP30 DEM from OpenTopography ...")

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        with open(filename, "wb") as fh:
            fh.write(resp.content)
        print(f"[Fetcher] SUCCESS — COP30 terrain saved as '{filename}'")
        return filename
    except Exception as exc:
        print(f"[Fetcher] WARNING — COP30 download failed: {exc}")
        if os.path.exists(filename):
            print(f"[Fetcher] OFFLINE MODE — using cached '{filename}'")
            return filename
        import glob as _glob
        fallbacks = _glob.glob(os.path.join(output_dir, "terrain_*.tif"))
        if fallbacks:
            print(f"[Fetcher] FALLBACK — using nearest cached terrain: {fallbacks[0]}")
            return fallbacks[0]
        print("[Fetcher] CRITICAL — no cached DEM found. Aborting.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ESRI World Imagery satellite texture
# ──────────────────────────────────────────────────────────────────────────────

def fetch_satellite_image_by_bounds(west, south, east, north,
                                     width, height, output_dir="."):
    """
    Fetch high-resolution satellite imagery by stitching ESRI World Imagery
    XYZ tiles (256×256 each).  XYZ tiles respond in <0.5s; the old export-API
    approach timed out on large bboxes.

    Zoom 14  → ~10m/px native ESRI resolution, 256×256 tiles.
    The final image is cropped to the exact bbox and saved as texture.jpg.
    No API key required.
    """
    import math
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image as _Image

    XYZ_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}")
    ZOOM = 14
    TILE_PX = 256

    def _deg2tile(lat_deg, lon_deg, zoom):
        lat_r = math.radians(lat_deg)
        n = 2 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
        return x, y

    def _tile2deg(x, y, zoom):
        """Return the NW corner (lat, lon) of tile (x, y) at zoom."""
        n = 2 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_r)
        return lat, lon

    # Tile range covering the bbox
    x0, y1 = _deg2tile(south, west, ZOOM)   # y increases southward
    x1, y0 = _deg2tile(north, east, ZOOM)
    x1 = max(x1, x0)
    y1 = max(y1, y0)

    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    total = nx * ny
    print(f"\n[Fetcher] Fetching satellite texture (ESRI XYZ z{ZOOM}, {nx}×{ny}={total} tiles) ...")

    def _fetch_tile(args):
        tx, ty = args
        url = XYZ_URL.format(z=ZOOM, y=ty, x=tx)
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                return (tx, ty, _Image.open(io.BytesIO(resp.content)).convert("RGB"))
            except Exception:
                if attempt == 2:
                    return (tx, ty, None)

    # Fetch all tiles in parallel
    tile_jobs = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    tile_map = {}
    workers = min(32, total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tx, ty, img in pool.map(_fetch_tile, tile_jobs):
            tile_map[(tx, ty)] = img

    failed = sum(1 for v in tile_map.values() if v is None)
    print(f"[Fetcher]   tiles fetched: {total - failed}/{total}")

    if failed == total:
        print("[Fetcher] ERROR — all tiles failed; satellite texture unavailable.")
        return None

    # Stitch canvas
    canvas_w = nx * TILE_PX
    canvas_h = ny * TILE_PX
    canvas = _Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            img = tile_map.get((tx, ty))
            if img is not None:
                px = (tx - x0) * TILE_PX
                py = (ty - y0) * TILE_PX
                canvas.paste(img, (px, py))

    # Crop canvas to exact bbox using pixel-fraction of the tile grid extent
    nw_lat, nw_lon = _tile2deg(x0, y0, ZOOM)
    se_lat, se_lon = _tile2deg(x1 + 1, y1 + 1, ZOOM)
    lon_span = se_lon - nw_lon
    lat_span = nw_lat - se_lat  # positive
    crop_left  = int((west  - nw_lon) / lon_span * canvas_w)
    crop_right = int((east  - nw_lon) / lon_span * canvas_w)
    crop_top   = int((nw_lat - north) / lat_span * canvas_h)
    crop_bot   = int((nw_lat - south) / lat_span * canvas_h)
    crop_left  = max(0, crop_left);  crop_top = max(0, crop_top)
    crop_right = min(canvas_w, crop_right); crop_bot = min(canvas_h, crop_bot)
    if crop_right > crop_left and crop_bot > crop_top:
        canvas = canvas.crop((crop_left, crop_top, crop_right, crop_bot))

    filename = os.path.join(output_dir, "texture.jpg")
    os.makedirs(output_dir, exist_ok=True)
    canvas.save(filename, "JPEG", quality=92)
    W, H = canvas.size
    print(f"[Fetcher] SUCCESS — satellite texture saved as '{filename}' ({W}×{H})")
    return filename


# ──────────────────────────────────────────────────────────────────────────────
# 4.  EEA CORINE Land Cover  (vegetation / fuel type map)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_corine_land_cover(west, south, east, north,
                             width, height, output_dir=".", force=False):
    """
    Fetch CORINE Land Cover 2018 as a raw-code GeoTIFF (integer CLC class codes).

    Primary:  EEA Discomap ArcGIS ImageServer — returns a single-band GeoTIFF
              where each pixel value is the CLC class code (1–44 sequential OR
              3-digit 111–523 depending on server version).  landscape.py uses
              a direct integer lookup table on this file — no colour matching.

    Fallback: EEA Discomap MapServer PNG rendered export — original behaviour;
              landscape.py falls back to nearest-RGB colour matching for PNGs.

    Cached as  corine_<tag>.tif  (primary)  or  corine_<tag>.png  (fallback).
    Returns the path to whichever file was saved / found in cache.
    """
    tag      = f"{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}"
    tif_path = os.path.join(output_dir, f"corine_{tag}.tif")
    png_path = os.path.join(output_dir, f"corine_{tag}.png")

    if not force:
        if os.path.exists(tif_path):
            print(f"[Fetcher] Using cached CLC GeoTIFF '{tif_path}'")
            return tif_path
        if os.path.exists(png_path):
            print(f"[Fetcher] Using cached CORINE PNG '{png_path}'")
            return png_path

    print(f"\n[Fetcher] Fetching CORINE Land Cover 2018 (CLC integer codes → GeoTIFF) ...")

    # ── Primary: ArcGIS ImageServer — raw pixel values (no rendering) ─────────
    # The ImageServer returns the raw DN = CLC class code rather than a rendered
    # RGB image.  pixelType=U8 + NearestNeighbor preserves exact class boundaries.
    _img_url = ("https://image.discomap.eea.europa.eu/arcgis/rest/services/"
                "Corine/CLC2018_WM/ImageServer/exportImage")
    try:
        resp = requests.get(_img_url, params={
            "bbox":         f"{west},{south},{east},{north}",
            "bboxSR":       "4326",
            "size":         f"{width},{height}",
            "imageSR":      "4326",
            "format":       "tiff",
            "pixelType":    "U8",
            "noDataInterpretation": "esriNoDataMatchAny",
            "interpolation": "RSP_NearestNeighbor",
            "f":            "image",
        }, timeout=60)
        resp.raise_for_status()
        content = resp.content
        # Validate: little-endian TIFF magic  II*\x00  or big-endian  MM\x00*
        if len(content) > 100 and content[:4] in (b'II*\x00', b'MM\x00*'):
            with open(tif_path, "wb") as fh:
                fh.write(content)
            print(f"[Fetcher] SUCCESS — CLC GeoTIFF saved: '{tif_path}'")
            return tif_path
        else:
            preview = content[:120].decode("utf-8", errors="replace")
            print(f"[Fetcher] ImageServer returned non-TIFF ({len(content)} B): "
                  f"{preview!r} — falling back to PNG")
    except Exception as exc:
        print(f"[Fetcher] ImageServer request failed ({exc}) — falling back to PNG")

    # ── Fallback: MapServer rendered PNG export ────────────────────────────────
    print(f"[Fetcher] Fetching CORINE PNG fallback (colour-matched) ...")
    _map_url = ("https://image.discomap.eea.europa.eu/arcgis/rest/services/"
                "Corine/CLC2018_WM/MapServer/export")
    try:
        resp = requests.get(_map_url, params={
            "bbox":        f"{west},{south},{east},{north}",
            "bboxSR":      "4326",
            "size":        f"{width},{height}",
            "imageSR":     "4326",
            "format":      "png",
            "transparent": "false",
            "f":           "image",
        }, timeout=30)
        resp.raise_for_status()
        with open(png_path, "wb") as fh:
            fh.write(resp.content)
        print(f"[Fetcher] PNG fallback saved: '{png_path}'")
        return png_path
    except Exception as exc2:
        print(f"[Fetcher] WARNING — CORINE PNG also failed: {exc2}")
        if os.path.exists(png_path):
            print(f"[Fetcher] OFFLINE MODE — using cached '{png_path}'")
            return png_path
        if os.path.exists(tif_path):
            print(f"[Fetcher] OFFLINE MODE — using cached '{tif_path}'")
            return tif_path
        raise RuntimeError(
            "CORINE fetch failed and no cached file exists. "
            "Check network or place corine_cover.png/tif manually."
        ) from exc2


# ──────────────────────────────────────────────────────────────────────────────
# 4b.  OpenStreetMap roads and urban areas  (via Overpass API)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_osm_features(west, south, east, north, output_dir=".", force=False):
    """
    Fetch roads and urban-area polygons from OpenStreetMap via the Overpass API.

    Roads (highway=motorway/trunk/primary/secondary/tertiary/residential/
    unclassified/service) → feature_type "road"

    Urban landuse (residential/commercial/industrial/retail/construction)
    → feature_type "urban"

    Saves result as a GeoJSON FeatureCollection cached by bbox.
    Returns the path to the GeoJSON file, or None if all endpoints fail.
    """
    import json

    tag          = f"{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}"
    geojson_path = os.path.join(output_dir, f"osm_features_{tag}.geojson")

    if not force and os.path.exists(geojson_path):
        print(f"[Fetcher] Using cached OSM features '{geojson_path}'")
        return geojson_path

    print(f"\n[Fetcher] Fetching OSM roads + urban areas (Overpass API) ...")

    # Overpass QL — bbox filter applied globally; we request geometry inline so
    # we get lat/lon of every node without a second lookup.
    # "out geom" on ways gives us the full node geometry in one response.
    query = (
        f"[out:json][timeout:90][bbox:{south},{west},{north},{east}];\n"
        "(\n"
        "  way[highway~\"^(motorway|trunk|primary|secondary|tertiary"
        "|residential|unclassified|service)$\"];\n"
        "  way[landuse~\"^(residential|commercial|industrial|retail"
        "|construction)$\"];\n"
        "  relation[landuse~\"^(residential|commercial|industrial|retail)$\"]"
        "[\"type\"=\"multipolygon\"];\n"
        ");\n"
        "out geom;"
    )

    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    # Overpass requires Content-Type: application/x-www-form-urlencoded.
    # A plain requests.post(data=...) usually sends this, but some servers
    # return 406 if the Accept header includes anything they can't satisfy.
    _headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    raw_data = None
    for endpoint in overpass_endpoints:
        try:
            resp = requests.post(endpoint, data=query.encode("utf-8"),
                                 headers=_headers, timeout=120)
            resp.raise_for_status()
            raw_data = resp.json()
            print(f"[Fetcher] OSM data received ({len(raw_data.get('elements', []))} "
                  f"elements) from {endpoint}")
            break
        except Exception as exc:
            print(f"[Fetcher]   {endpoint} → {exc}")

    if raw_data is None:
        print("[Fetcher] WARNING — all Overpass endpoints unreachable; OSM overlay skipped.")
        return None

    # Convert Overpass JSON → GeoJSON FeatureCollection
    # Handle both "way" (simple polygon/road) and "relation" (multipolygon urban area).
    features = []
    for elem in raw_data.get("elements", []):
        etype = elem.get("type")
        tags  = elem.get("tags", {})
        hw    = tags.get("highway", "")
        lu    = tags.get("landuse", "")

        if etype == "way":
            geom_nodes = elem.get("geometry", [])
            if len(geom_nodes) < 2:
                continue
            coords = [[n["lon"], n["lat"]] for n in geom_nodes]

            if hw:
                geom  = {"type": "LineString", "coordinates": coords}
                ftype = "road"
            elif lu:
                if coords[0] != coords[-1] and len(coords) >= 4:
                    coords.append(coords[0])
                geom  = {"type": "Polygon", "coordinates": [coords]}
                ftype = "urban"
            else:
                continue

        elif etype == "relation":
            # Collect outer-ring member coordinates to form the urban polygon.
            # Overpass "out geom" embeds each member way's geometry inline.
            if not lu:
                continue
            outer_coords = []
            for member in elem.get("members", []):
                if member.get("type") == "way" and member.get("role") == "outer":
                    nodes = member.get("geometry", [])
                    outer_coords.extend([[n["lon"], n["lat"]] for n in nodes])
            if len(outer_coords) < 4:
                continue
            if outer_coords[0] != outer_coords[-1]:
                outer_coords.append(outer_coords[0])
            geom  = {"type": "Polygon", "coordinates": [outer_coords]}
            ftype = "urban"

        else:
            continue

        features.append({
            "type":       "Feature",
            "geometry":   geom,
            "properties": {"feature_type": ftype, "highway": hw, "landuse": lu},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(geojson_path, "w") as fh:
        json.dump(geojson, fh)

    n_roads = sum(1 for f in features if f["properties"]["feature_type"] == "road")
    n_urban = sum(1 for f in features if f["properties"]["feature_type"] == "urban")
    print(f"[Fetcher] OSM: {n_roads} road segments, {n_urban} urban polygons "
          f"→ '{geojson_path}'")
    return geojson_path


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Master Orchestrator  — single call to prepare a full simulation
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_wildfire_data(
    lat:        float,
    lon:        float,
    start_date: str,
    end_date:   str,
    firms_map_key: str,
    radius_km:  float = 15.0,
    output_dir: str   = ".",
    force:      bool  = False,
) -> dict:
    """
    Master API Orchestrator.

    Given a GPS centre point and a date range, fetches ALL data required to
    run a hindcast simulation and returns them as a single ready-to-use dict.

    Orchestrates
    ------------
    1. Bounding box calculation from (lat, lon, radius_km)
    2. Open-Meteo ERA5 weather for start_date
    3. NASA FIRMS VIIRS active-fire detections
    4. Copernicus COP30 DEM via OpenTopography
    5. CORINE Land Cover 2018 PNG

    Parameters
    ----------
    lat, lon       : centre of the study area (decimal degrees)
    start_date     : ISO date string "YYYY-MM-DD" — first day of FIRMS window
    end_date       : ISO date string "YYYY-MM-DD" — last day of FIRMS window
    firms_map_key  : NASA FIRMS map key (get one at firms.modaps.eosdis.nasa.gov)
    radius_km      : approximate search radius in km (default 15 km ≈ 0.135°)
    output_dir     : directory to write all downloaded files (default ".")
    force          : if True, re-download even if cached files exist

    Returns
    -------
    dict with keys
      "bbox"         : (lat_min, lat_max, lon_min, lon_max)
      "weather"      : dict from fetch_weather()
      "df_firms"     : pandas DataFrame of VIIRS detections (may be empty)
      "dem_file"     : path to the GeoTIFF, or None on failure
      "corine_file"  : path to the CORINE PNG, or None on failure
      "status"       : dict of per-step success flags
    """
    import math as _math

    # ── 1.  Bounding box ─────────────────────────────────────────────────────
    # 1° latitude ≈ 111 km; 1° longitude ≈ 111 km × cos(lat)
    lat_buf = radius_km / 111.0
    lon_buf = radius_km / (111.0 * max(_math.cos(_math.radians(lat)), 0.01))
    lat_min, lat_max = lat - lat_buf, lat + lat_buf
    lon_min, lon_max = lon - lon_buf, lon + lon_buf
    bbox = (lat_min, lat_max, lon_min, lon_max)

    print("\n" + "=" * 60)
    print("  MASTER ORCHESTRATOR — fetch_all_wildfire_data")
    print("=" * 60)
    print(f"  Centre    : {lat:.5f}°N  {lon:.5f}°E")
    print(f"  Radius    : {radius_km} km")
    print(f"  Bbox      : S={lat_min:.4f} N={lat_max:.4f} W={lon_min:.4f} E={lon_max:.4f}")
    print(f"  Dates     : {start_date} → {end_date}")

    status = {
        "weather": False,
        "firms":   False,
        "dem":     False,
        "corine":  False,
    }

    # ── 2.  Weather ──────────────────────────────────────────────────────────
    weather = None
    try:
        weather = _fetch_weather_orchestrator(lat, lon, start_date)
        status["weather"] = True
        print(f"  [1/4] Weather OK  — {weather['wind_speed_ms']:.1f} m/s @ "
              f"{weather['wind_direction']:.0f}°  "
              f"T={weather['temperature_c']:.1f}°C  RH={weather['relative_humidity']:.0f}%")
    except Exception as exc:
        print(f"  [1/4] Weather FAILED: {exc}")

    # ── 3.  NASA FIRMS ───────────────────────────────────────────────────────
    df_firms = None
    try:
        df_firms = _fetch_firms_orchestrator(
            firms_map_key, lat_min, lat_max, lon_min, lon_max, start_date, end_date
        )
        status["firms"] = True
        print(f"  [2/4] FIRMS OK    — {len(df_firms)} detections")
    except Exception as exc:
        print(f"  [2/4] FIRMS FAILED: {exc}")
        import pandas as _pd
        df_firms = _pd.DataFrame()

    # ── 4.  COP30 DEM ────────────────────────────────────────────────────────
    dem_file = None
    try:
        dem_file = fetch_terrain_from_api(
            lat, lon, output_dir=output_dir,
            buffer=max(lat_buf, lon_buf), force=force
        )
        status["dem"] = dem_file is not None
        print(f"  [3/4] DEM OK      — {dem_file}")
    except Exception as exc:
        print(f"  [3/4] DEM FAILED: {exc}")

    # ── 5.  CORINE ───────────────────────────────────────────────────────────
    corine_file = None
    if dem_file:
        try:
            corine_file = fetch_corine_land_cover(
                lon_min, lat_min, lon_max, lat_max, 400, 400,
                output_dir=output_dir, force=force
            )
            status["corine"] = corine_file is not None
            print(f"  [4/4] CORINE OK   — {corine_file}")
        except Exception as exc:
            print(f"  [4/4] CORINE FAILED: {exc}")

    n_ok = sum(status.values())
    print(f"\n  Orchestrator complete: {n_ok}/4 sources ready")
    print("=" * 60)

    return {
        "bbox":        bbox,
        "lat_min":     lat_min,
        "lat_max":     lat_max,
        "lon_min":     lon_min,
        "lon_max":     lon_max,
        "weather":     weather,
        "df_firms":    df_firms,
        "dem_file":    dem_file,
        "corine_file": corine_file,
        "status":      status,
    }


# ── Internal helpers used only by the orchestrator ────────────────────────────

def _fetch_weather_orchestrator(lat: float, lon: float, date: str) -> dict:
    """Calls Open-Meteo ERA5 hourly for the given date; picks ignition-hour conditions.

    Delegates to hindcast_optimizer.fetch_weather_hourly for consistency,
    then returns the noon (12:00 UTC) snapshot as the representative dict
    (orchestrator doesn't know the exact ignition time at this stage).
    """
    try:
        from pipeline.hindcast_optimizer import fetch_weather_hourly as _fwh
        df = _fwh(lat, lon, date_start=date, date_end=date)
        idx  = min(12, len(df) - 1)
        row  = df.iloc[idx]
        return {
            "temperature_c":     float(row["temperature_c"]),
            "relative_humidity": float(row["relative_humidity"]),
            "wind_speed_ms":     float(row["wind_speed_ms"]),
            "wind_direction":    float(row["wind_direction"]),
        }
    except Exception:
        # Minimal fallback in case of import issues
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
            "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
            "&wind_speed_unit=ms&timezone=UTC"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        h    = resp.json().get("hourly", {})
        idx  = min(12, len(h.get("temperature_2m", [0])) - 1)
        return {
            "temperature_c":     float(h["temperature_2m"][idx]),
            "relative_humidity": float(h["relative_humidity_2m"][idx]),
            "wind_speed_ms":     float(h["wind_speed_10m"][idx]),
            "wind_direction":    float(h["wind_direction_10m"][idx]),
        }


def _fetch_firms_orchestrator(map_key, lat_min, lat_max, lon_min, lon_max,
                               date_start, date_end):
    """
    Lightweight FIRMS fetch used by the orchestrator.
    Returns a DataFrame (may be empty — caller handles that).
    """
    import pandas as _pd
    import datetime as _dt

    _SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]
    _COLS    = ["latitude", "longitude", "acq_date", "acq_time", "confidence"]

    start = _dt.date.fromisoformat(date_start)
    end   = _dt.date.fromisoformat(date_end)
    days  = max(1, (end - start).days + 1)

    for source in _SOURCES:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            f"/{map_key}/{source}/{lon_min},{lat_min},{lon_max},{lat_max}/{days}/{date_start}"
        )
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = _pd.read_csv(__import__("io").StringIO(resp.text))
            if df.empty or "latitude" not in df.columns:
                continue
            df["acq_datetime"] = _pd.to_datetime(
                df["acq_date"].astype(str) + " " +
                df["acq_time"].astype(str).str.zfill(4).str[:2] + ":" +
                df["acq_time"].astype(str).str.zfill(4).str[2:],
                utc=True
            )
            conf_ok = df["confidence"].astype(str).str.lower().isin(
                ["high", "nominal", "h", "n"]
            ) | df["confidence"].apply(
                lambda c: str(c).lstrip("0").isdigit() and int(str(c)) >= 50
            )
            df = df[conf_ok].copy()
            if not df.empty:
                print(f"  [FIRMS] {len(df)} detections from {source}")
                return df
        except Exception:
            continue

    return _pd.DataFrame()


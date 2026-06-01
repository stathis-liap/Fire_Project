"""
Project  WebSocket Simulation ServerWILSON 
=============================================
Runs a WebSocket server on ws://localhost:8765.

Protocol
--------
 Server
  {"type": "init",         "lat_center": float, "lon_center": float,
   "bbox_degrees": float,  "ignition_points": [{"lat": f, "lon": f}, ...],
   "wind_speed_ms": float, "wind_dir_deg": float,
   "temperature_c": float, "relative_humidity": float,
   "ignition_time_utc": "HH:MM"}

  {"type": "intervention", "action": "firebreak",
   "points": [{"lat": f, "lon": f}, ...]}
  {"type": "intervention", "action": "water_drop",
   "lat": float, "lon": float, "radius_m": float}
  {"type": "intervention", "action": "pause"}
  {"type": "intervention", "action": "resume"}
  {"type": "scrub", "index": int}

 Client
  {"type": "frame", "step": int,
   "geojson_burning": FeatureCollection,   #  active cellsMultiPolygon 
   "geojson_burned":  FeatureCollection,   #  burned cellsMultiPolygon 
   "burned_ha": float, "active_ha": float,
   "wall_clock_minutes": float}

  {"type": "history_ready", "count": int,
   "frames": [{"step", "geojson_burning", "geojson_burned",
               "burned_ha", "active_ha", "wall_clock_minutes"}, ...]}

  {"type": "error",  "message": str}
  {"type": "status", "message": str}

Usage
-----
  pip install websockets
  python server.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
import traceback
from typing import Any

import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("websockets is not installed.  Run:  pip install websockets")

import config as wilson_config
from core.landscape import Landscape
from core.fire_model import CellularAutomataFire
from pipeline.hindcast_optimizer import (
    apply_weather_to_landscape,
    GeoGrid,
    fetch_weather_hourly,
)
from fire_info_panel import FireInfoPanel

HOST = "localhost"
PORT = 8765
DEFAULT_STEPS_PER_SEND = 5
STEP_SLEEP_S           = 0.01
MIN_FRAME_INTERVAL_S   = 0.09   # 90 ms minimum between frame sends (~11 FPS)

logging.basicConfig(
    level=logging.INFO,
    format="[WILSON-WS] %(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wilson_ws")


# 
# Helpers
# 

def _ha(cells: int, cell_size_m: float) -> float:
    return cells * cell_size_m ** 2 / 10_000.0


def _b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).ravel().tobytes()).decode("ascii")


def _quadrant_ros(sim, state: np.ndarray,
                  rows: int, cols: int, cell_m: float) -> tuple:
    """Return average ROS (m/min) for N/E/S/W quadrants of the active fire front."""
    active = (state == 1)
    if not active.any() or not hasattr(sim, "p_spread"):
        return 0.0, 0.0, 0.0, 0.0

    fr, fc = np.where(active)
    cr, cc = fr.mean(), fc.mean()

    row_g = np.arange(rows)[:, None]
    col_g = np.arange(cols)[None, :]

    def _qros(mask):
        if not mask.any():
            return 0.0
        max_p = sim.p_spread[:, mask].max(axis=0)
        return float(max_p.mean() * cell_m / max(sim.dt, 1e-9))

    return (
        _qros(active & (row_g <  cr)),   # N
        _qros(active & (col_g >  cc)),   # E
        _qros(active & (row_g >  cr)),   # S
        _qros(active & (col_g <  cc)),   # W
    )


def _state_to_geojson(state: np.ndarray, geo_grid: GeoGrid,
                      rows: int, cols: int) -> tuple[dict, dict]:
    """
    Convert fire state array to two GeoJSON FeatureCollections:
    geojson_burning (state==1) and geojson_burned (state==2).

    Uses run-length encoding per  each contiguous run of matching cellsrow 
    becomes a single rectangle.  Reduces polygon count from O(cells) to O(runs).
    Fully vectorised within each row using numpy diff.
    """
    lon_min = geo_grid.lon_min
    lat_min = geo_grid.lat_min
    dlon = (geo_grid.lon_max - lon_min) / cols
    dlat = (geo_grid.lat_max - lat_min) / rows

    def _fc_for_val(val: int) -> dict:
        polygons = []
        for r in range(rows):
            row_mask = (state[r] == val)
            if not row_mask.any():
                continue
            # Pad with False on each side; diff finds run boundaries
            padded       = np.empty(cols + 2, dtype=np.bool_)
            padded[0]    = False
            padded[-1]   = False
            padded[1:-1] = row_mask
            diff     = np.diff(padded.view(np.int8))
            c_starts = np.where(diff  == 1)[0]   # inclusive start col
            c_ends   = np.where(diff == -1)[0]   # exclusive end col

            # Grid is south-up (row 0 = lat_min), so geometry must follow that.
            lat_s = lat_min + r * dlat
            lat_n = lat_s + dlat

            for cs, ce in zip(c_starts.tolist(), c_ends.tolist()):
                lon_w = lon_min + cs * dlon
                lon_e = lon_min + ce * dlon
                polygons.append([[
                    [lon_w, lat_s], [lon_e, lat_s],
                    [lon_e, lat_n], [lon_w, lat_n],
                    [lon_w, lat_s],
                ]])

        if not polygons:
            return {"type": "FeatureCollection", "features": []}
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
                "properties": {},
            }],
        }

    return _fc_for_val(1), _fc_for_val(2)


def _fuel_map_to_geojson(land: "Landscape", geo_grid: "GeoGrid",
                         rows: int, cols: int) -> dict:
    """
    Convert the fuel_map integer array to a GeoJSON FeatureCollection.
    Uses RLE per row per fuel type — one rectangle per contiguous run.
    Each feature carries a 'color' property (hex) and 'fuel' (name string).
    """
    from core.fuels import FUEL_DISPLAY_COLORS, FUEL_DISPLAY_COLOR_DEFAULT

    lon_min = geo_grid.lon_min
    lat_min = geo_grid.lat_min
    dlon    = (geo_grid.lon_max - lon_min) / cols
    dlat    = (geo_grid.lat_max - lat_min) / rows

    def _rgba_to_hex(rgba: tuple) -> str:
        r, g, b = rgba[0], rgba[1], rgba[2]
        return f"#{r:02x}{g:02x}{b:02x}"

    features = []
    fuel_map = land.fuel_map
    fuel_names = land.fuel_names

    # Build a per-index color lookup
    color_lookup: dict[int, str] = {}
    for idx, name in enumerate(fuel_names):
        rgba = FUEL_DISPLAY_COLORS.get(name, FUEL_DISPLAY_COLOR_DEFAULT)
        color_lookup[idx] = _rgba_to_hex(rgba)

    for r in range(rows):
        row = fuel_map[r]
        # Grid is south-up (row 0 = lat_min), so geometry must follow that.
        lat_s = lat_min + r * dlat
        lat_n = lat_s + dlat

        # Find runs of same value using diff
        padded = np.empty(cols + 2, dtype=np.int32)
        padded[0]    = -1
        padded[-1]   = -1
        padded[1:-1] = row
        diff      = np.diff(padded)
        starts    = np.where(diff != 0)[0]   # positions in padded where value changes
        # Each entry cs in starts: run of new value begins at row[cs] (= padded[cs+1])
        for si in range(len(starts)):
            cs = int(starts[si])
            if cs >= cols:
                break   # trailing sentinel transition — no real data beyond here
            ce = min(int(starts[si + 1]) if si + 1 < len(starts) else cols, cols)
            fuel_idx = int(row[cs])
            if fuel_idx < 0 or fuel_idx >= len(fuel_names):
                continue
            name  = fuel_names[fuel_idx]
            color = color_lookup.get(fuel_idx, "#808080")
            lon_w = lon_min + cs  * dlon
            lon_e = lon_min + ce  * dlon
            features.append({
                "type": "Feature",
                "properties": {"fuel": name, "color": color},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_w, lat_s], [lon_e, lat_s],
                        [lon_e, lat_n], [lon_w, lat_n],
                        [lon_w, lat_s],
                    ]],
                },
            })

    return {"type": "FeatureCollection", "features": features}


def _snap_to_land(rc: tuple[int, int], land: Landscape) -> tuple[int, int] | None:
    """Walk outward from rc until a burnable cell is found."""
    from core.fuels import NON_BURNING_FUELS
    rows, cols = land.shape
    r0, c0 = rc
    for radius in range(0, 6):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < rows and 0 <= c < cols:
                    fname = (land.fuel_names[land.fuel_map[r, c]]
                             if land.fuel_map[r, c] < len(land.fuel_names) else "")
                    if fname not in NON_BURNING_FUELS:
                        return r, c
    return None


def _build_landscape(msg: dict) -> tuple[Landscape, GeoGrid, float]:
    """
    Construct (landscape, geo_grid, cell_size_m) from an 'init' message.
    Uses 800800 grid for organic perimeter; falls back to synthetic terrain.
    """
    lat_c = float(msg["lat_center"])
    lon_c = float(msg["lon_center"])
    half  = float(msg["bbox_degrees"])

    lat_min = lat_c - half
    lat_max = lat_c + half
    lon_min = lon_c - half
    lon_max = lon_c + half

    import types as _types
    cfg = _types.SimpleNamespace(**{
        k: v for k, v in vars(wilson_config).items()
        if not k.startswith("__") and not isinstance(v, _types.ModuleType)
    })
    cfg.GRID_SIZE         = (800, 800)   # doubled for organic perimeter shape
    cfg.TEMPERATURE_C     = float(msg.get("temperature_c", 35.0))
    cfg.RELATIVE_HUMIDITY = float(msg.get("relative_humidity", 30.0))

    land = Landscape(cfg)

    real_loaded = False
    try:
        from pipeline.auto_fetcher import (
            fetch_terrain_from_api,
            fetch_corine_land_cover,
            fetch_osm_features,
        )

        dem_path    = fetch_terrain_from_api(lat_c, lon_c, buffer=half)
        corine_path = fetch_corine_land_cover(
            lon_min, lat_min, lon_max, lat_max, width=800, height=800
        )
        osm_path    = fetch_osm_features(lon_min, lat_min, lon_max, lat_max)

        if dem_path:
            land.load_real_terrain(
                dem_file     = dem_path,
                corine_file  = corine_path or "",
                target_shape = (800, 800),
                osm_file     = osm_path,
            )
            real_loaded = True
            log.info("Real terrain loaded (%dx%d cells, cell=%.0f m)",
                     *land.shape, cfg.CELL_SIZE_METERS)
    except Exception as exc:
        log.warning("Real terrain fetch failed (%s) — using synthetic fallback", exc)

    if not real_loaded:
        land.generate_random_terrain()
        import math
        lat_span_m = (lat_max - lat_min) * 111_320
        lon_span_m = (lon_max - lon_min) * 111_320 * math.cos(math.radians(lat_c))
        cfg.CELL_SIZE_METERS = float(max(lat_span_m, lon_span_m) / 800)
        log.info("Synthetic terrain  (cell=%.0f m)", cfg.CELL_SIZE_METERS)

    # Apply weather
    weather = {
        "wind_speed_ms":     float(msg["wind_speed_ms"]),
        "wind_direction":    float(msg["wind_dir_deg"]),
        "temperature_c":     float(msg.get("temperature_c", 35.0)),
        "relative_humidity": float(msg.get("relative_humidity", 30.0)),
        "_dir_is_from":      True,
    }
    apply_weather_to_landscape(land, weather)

    # Force dry fuel moisture on synthetic terrain (after weather to avoid overwrite)
    if not real_loaded:
        land.moisture[:] = 0.05

    geo_grid = GeoGrid(lat_min, lat_max, lon_min, lon_max, *land.shape)
    return land, geo_grid, float(cfg.CELL_SIZE_METERS)


# 
# Intervention helpers
# 

def _apply_firebreak(land: Landscape, sim: CellularAutomataFire,
                     geo_grid: GeoGrid, points: list[dict]) -> int:
    if "Non_Combustible" not in land.fuel_names:
        land.fuel_names.append("Non_Combustible")
    nc_idx = land.fuel_names.index("Non_Combustible")
    rows, cols = land.shape

    line_cells: set[tuple[int, int]] = set()
    for i in range(len(points) - 1):
        r0, c0 = geo_grid.latlon_to_rc(points[i]["lat"],   points[i]["lon"])
        r1, c1 = geo_grid.latlon_to_rc(points[i+1]["lat"], points[i+1]["lon"])
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
        err = dr - dc
        r, c = r0, c0
        while True:
            line_cells.add((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc: err -= dc; r += sr
            if e2 <  dr: err += dr; c += sc

    modified = 0
    fb_mask = np.zeros(land.shape, dtype=np.bool_)
    for lr, lc in line_cells:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr * dr + dc * dc > 4:
                    continue
                r, c = lr + dr, lc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if land.fuel_map[r, c] != nc_idx:
                        land.fuel_map[r, c] = nc_idx
                        modified += 1
                    fb_mask[r, c] = True

    if fb_mask.any():
        if hasattr(sim, "apply_firebreak_mask"):
            sim.apply_firebreak_mask(fb_mask)
        else:
            sim.state[fb_mask] = 0
            sim.burn_timer[fb_mask] = 0
            sim.ignition_fraction[fb_mask] = 0.0

    if modified and hasattr(sim, "_precompute_ros_grid"):
        sim._precompute_ros_grid()

    return modified


def _apply_water_drop(land: Landscape, sim: CellularAutomataFire,
                      geo_grid: GeoGrid,
                      lat: float, lon: float,
                      radius_m: float, cell_m: float) -> int:
    rows, cols = land.shape
    rc, cc = geo_grid.latlon_to_rc(lat, lon)
    radius_cells = max(1, int(radius_m / cell_m))
    affected = 0
    wet_mask = np.zeros(land.shape, dtype=np.bool_)

    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells:
                continue
            r, c = rc + dr, cc + dc
            if 0 <= r < rows and 0 <= c < cols:
                # Saturate moisture to near-extinction level so the cell can't re-ignite
                land.moisture[r, c] = min(land.config.RELATIVE_HUMIDITY / 100.0 * 0.35 + 0.20, 0.35)
                wet_mask[r, c] = True
                affected += 1

    if wet_mask.any():
        if hasattr(sim, "apply_water_mask"):
            sim.apply_water_mask(wet_mask, wetness=0.92)
        else:
            active = wet_mask & (sim.state == 1)
            sim.state[active] = 0
            sim.burn_timer[active] = 0
            sim.ignition_fraction[wet_mask & (sim.state != 2)] = 0.0

    # Re-compute ROS so saturated cells have near-zero p_spread
    if affected and hasattr(sim, "_precompute_ros_grid"):
        sim._precompute_ros_grid()

    return affected


# ─────────────────────────────────────────────────────────────────────────────
# Particle-swarm optimizer task (runs in a background thread via asyncio)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_optimizer_task(send, msg_init: dict, opt_msg: dict,
                               ctrl: dict) -> None:
    """
    Launch the PSO optimizer in a thread-pool thread, stream progress
    updates back over WebSocket, then send the final result.
    """
    try:
        from pipeline.particle_optimizer import run_particle_swarm
    except Exception as exc:
        await send({"type": "error",
                    "message": f"Optimizer import failed: {exc}"})
        ctrl["optimizer_running"] = False
        return

    n_particles   = max(4,  min(100, int(opt_msg.get("n_particles", 20))))
    n_iterations  = max(3,  min(50,  int(opt_msg.get("n_iterations", 12))))
    elapsed_hours = max(0.1, float(opt_msg.get("elapsed_hours", 6.0)))
    truth_mask_geo = opt_msg.get("truth_mask_geojson",
                                  {"type": "FeatureCollection", "features": []})

    await send({"type": "optimizer_status",
                "message": (
                    f"Optimizer starting: {n_particles} particles × {n_iterations} iterations"
                    f" — fire age {elapsed_hours:.2f} h…"
                )})

    loop           = asyncio.get_event_loop()
    progress_queue: asyncio.Queue = asyncio.Queue()

    def _progress_cb(iteration, n_done, n_total, best_iou, eta_s):
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            {
                "type":      "optimizer_progress",
                "iteration": iteration,
                "n_done":    n_done,
                "n_total":   n_total,
                "best_iou":  round(float(best_iou), 4),
                "eta_s":     max(0, int(eta_s)),
            },
        )

    async def _drain():
        while True:
            msg = await progress_queue.get()
            if msg is None:
                break
            try:
                await send(msg)
            except Exception:
                pass

    drain_task = asyncio.create_task(_drain())

    try:
        result = await asyncio.to_thread(
            run_particle_swarm,
            msg_init, truth_mask_geo, n_particles, n_iterations,
            elapsed_hours=elapsed_hours,
            progress_cb=_progress_cb,
        )
    except Exception as exc:
        log.error("Optimizer failed: %s", exc)
        traceback.print_exc()
        await progress_queue.put(None)
        await drain_task
        await send({"type": "error",
                    "message": f"Optimizer failed: {exc}"})
        ctrl["optimizer_running"] = False
        return

    await progress_queue.put(None)   # signal drain to stop
    await drain_task

    geo = result["geo_grid"]
    await send({
        "type":            "optimizer_result",
        "best_params":     result["best_params"],
        "best_iou":        round(result["best_iou"], 4),
        "heatmap_png_b64": result.get("heatmap_png_b64", ""),
        "lat_min":         geo.lat_min,
        "lat_max":         geo.lat_max,
        "lon_min":         geo.lon_min,
        "lon_max":         geo.lon_max,
    })
    log.info("Optimizer complete — best IoU %.3f", result["best_iou"])
    ctrl["optimizer_running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer-only session handler (opened via "init_optimizer" message)
# Completely isolated from the live simulation — no fire CA is created.
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_optimizer_session(websocket, send, msg_init: dict) -> None:
    """
    Handles a WebSocket connection that arrives with type='init_optimizer'.
    No fire simulation is ever spawned — this session exists purely to run
    the PSO calibration and return ensemble results.
    """
    log.info("Optimizer session opened  %s", websocket.remote_address)
    await send({"type": "init_optimizer_ack",
                "message": "Optimizer session ready."})

    ctrl = {"optimizer_running": False}

    async for raw_msg in websocket:
        try:
            iv       = json.loads(raw_msg)
            msg_type = iv.get("type", "")

            if msg_type == "run_optimizer":
                if ctrl["optimizer_running"]:
                    await send({"type": "optimizer_status",
                                "message": "Optimizer already running — please wait."})
                else:
                    ctrl["optimizer_running"] = True
                    asyncio.create_task(
                        _run_optimizer_task(send, msg_init, iv, ctrl)
                    )
            else:
                log.debug("Optimizer session: unexpected msg type '%s'", msg_type)

        except Exception as exc:
            log.warning("Optimizer session: bad message: %s", exc)

    log.info("Optimizer session closed  %s", websocket.remote_address)


#
# Per-connection handler
#

async def _handle(websocket):
    log.info("Client connected  %s", websocket.remote_address)

    async def send(obj: dict):
        await websocket.send(json.dumps(obj))

    # ── Wait for 'init' ───────────────────────────────────────────────────────
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=60.0)
        msg: dict[str, Any] = json.loads(raw)
    except asyncio.TimeoutError:
        await send({"type": "error", "message": "Timed out waiting for 'init' message."})
        return
    except Exception as exc:
        await send({"type": "error", "message": f"Bad message: {exc}"})
        return

    msg_type_recv = msg.get("type")

    # Route optimizer-only connections to their own dedicated handler.
    if msg_type_recv == "init_optimizer":
        await _handle_optimizer_session(websocket, send, msg)
        return

    if msg_type_recv != "init":
        await send({"type": "error",
                    "message": f"Expected 'init', got '{msg_type_recv}'."})
        return

    await send({"type": "status", "message": "Building landscape..."})

    try:
        land, geo_grid, cell_m = _build_landscape(msg)
    except Exception as exc:
        log.error("Landscape build failed: %s", exc)
        traceback.print_exc()
        await send({"type": "error", "message": f"Landscape build failed: {exc}"})
        return

    # ── CFL-correct dt scaling for 800-cell grid (half dt of 400-cell) ────────
    half        = float(msg["bbox_degrees"])
    lat_c       = float(msg["lat_center"])
    lon_c       = float(msg["lon_center"])
    lat_min     = lat_c - half
    lat_max     = lat_c + half
    base_dt     = 0.1
    coarse_cell = (lat_max - lat_min) * 111_320 / 400
    fine_cell   = (lat_max - lat_min) * 111_320 / 800
    dt_fine     = base_dt * (fine_cell / coarse_cell)   # = 0.05 min

    try:
        sim = CellularAutomataFire(land, land.config, dt=dt_fine)
    except TypeError:
        sim = CellularAutomataFire(land, land.config)
        sim.dt = dt_fine

    # ── Parse ignition wall-clock time ────────────────────────────────────────
    ign_time_str = msg.get("ignition_time_utc", "12:00")
    try:
        ign_h, ign_m = [int(x) for x in ign_time_str.split(":")[:2]]
    except Exception:
        ign_h, ign_m = 12, 0
    ignition_wall_hour = ign_h + ign_m / 60.0

    # Ignite supplied points
    ignition_points = msg.get("ignition_points", [])
    if not ignition_points:
        ignition_points = [{"lat": msg["lat_center"], "lon": msg["lon_center"]}]

    ignited_count = 0
    skipped_ignitions = 0
    for pt in ignition_points:
        rc = geo_grid.latlon_to_rc(float(pt["lat"]), float(pt["lon"]))
        rc = _snap_to_land(rc, land)
        if rc is None:
            skipped_ignitions += 1
            continue
        r0, c0 = rc
        # Smaller initial ignition footprint for tighter scenario control.
        radius = max(0, int(90.0 / max(cell_m, 1e-6)))
        rows_g, cols_g = land.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < rows_g and 0 <= cc < cols_g:
                    sim.ignite(rr, cc)
        ignited_count += 1

    rows, cols = land.shape

    # ── Wind grid subsampling (20×20) ─────────────────────────────────────────
    _ri  = np.linspace(0, rows - 1, 20, dtype=int)
    _ci  = np.linspace(0, cols - 1, 20, dtype=int)
    wu_20 = sim._wind_u_grid[np.ix_(_ri, _ci)]
    wv_20 = sim._wind_v_grid[np.ix_(_ri, _ci)]

    # Optional hourly weather
    df_hourly = None
    hourly_records: list[dict] = []
    _date_start = msg.get("date_start", "")
    if _date_start:
        try:
            _date_end = msg.get("date_end", _date_start)
            df_hourly = fetch_weather_hourly(lat_c, lon_c,
                                             date_start=_date_start,
                                             date_end=_date_end)
            for ts, row in df_hourly.iterrows():
                hourly_records.append({
                    "time":              ts.isoformat(),
                    "wind_speed_ms":     round(float(row.wind_speed_ms),     2),
                    "wind_direction":    round(float(row.wind_direction),     1),
                    "temperature_c":     round(float(row.temperature_c),     1),
                    "relative_humidity": round(float(row.relative_humidity), 1),
                })
            log.info("Hourly weather: %d records loaded", len(hourly_records))
        except Exception as exc:
            log.warning("Hourly weather fetch failed (%s)", exc) 

    # Build fuel map GeoJSON once for the vegetation tab
    log.info("Building vegetation GeoJSON ...")
    geojson_veg = _fuel_map_to_geojson(land, geo_grid, rows, cols)
    log.info("Vegetation GeoJSON: %d features", len(geojson_veg["features"]))

    fire_info = FireInfoPanel()

    # Send init_ack
    await send({
        "type":             "init_ack",
        "rows":             rows,
        "cols":             cols,
        "cell_m":           cell_m,
        "dt_minutes":       sim.dt,
        "lat_min":          geo_grid.lat_min,
        "lat_max":          geo_grid.lat_max,
        "lon_min":          geo_grid.lon_min,
        "lon_max":          geo_grid.lon_max,
        "wind_u_grid":      _b64_float32(wu_20),
        "wind_v_grid":      _b64_float32(wv_20),
        "wind_grid_rows":   20,
        "wind_grid_cols":   20,
        "hourly_weather":   hourly_records,
        "fired_timeline":   None,
        "osm_overlay_stats": getattr(land, "osm_overlay_stats", None),
        "geojson_fuel_map": geojson_veg,
        "message":          (
            f"Simulation ready -- {ignited_count} ignition point(s). Starting..."
            if skipped_ignitions == 0 else
            f"Simulation ready -- {ignited_count} ignition point(s), "
            f"{skipped_ignitions} skipped (non-burnable urban/water). Starting..."
        ),
    })
    await send(fire_info.init_started(rows, cols, cell_m))

    # Shared state for concurrent tasks
    ctrl = {
        "paused":         False,
        "stop":           False,
        "steps_per_send": DEFAULT_STEPS_PER_SEND,
    }
    intervention_queue: asyncio.Queue = asyncio.Queue()
    # Each entry: (step, gjson_burning, gjson_burned, burned_ha, active_ha, wall_clock_min)
    sim_history: list = []

    # Simulation task
    async def run_simulation():
        step           = 0
        last_hour      = -1
        last_send_time = 0.0

        while not ctrl["stop"]:
            # Drain interventions
            while not intervention_queue.empty():
                iv      = await intervention_queue.get()
                iv_type = iv.get("type", "")
                action  = iv_type if iv_type in ("pause", "resume", "set_speed") \
                          else iv.get("action", iv.get("intervention_type", ""))

                if action == "pause":
                    ctrl["paused"] = True
                    await send({"type": "status", "message": "Simulation paused."})

                elif action == "resume":
                    ctrl["paused"] = False
                    await send({"type": "status", "message": "Simulation resumed."})

                elif action == "set_speed":
                    n = max(1, min(100, int(iv.get("steps_per_send",
                                               DEFAULT_STEPS_PER_SEND))))
                    ctrl["steps_per_send"] = n
                    await send({"type": "status",
                                "message": f"Speed set: {n} steps/frame."})

                elif action == "firebreak":
                    n = _apply_firebreak(land, sim, geo_grid, iv.get("points", []))
                    await send({"type": "status",
                                "message": f"Firebreak applied: {n} cells cleared."})
                    await send(fire_info.intervention("firebreak", n))

                elif action == "water_drop":
                    n = _apply_water_drop(land, sim, geo_grid,
                                          float(iv["lat"]), float(iv["lon"]),
                                          float(iv.get("radius_m", 435.0)), cell_m)
                    await send({"type": "status",
                                "message": f"Water drop: {n} cells affected."})
                    await send(fire_info.intervention("water_drop", n))

            if ctrl["paused"]:
                await asyncio.sleep(0.1)
                continue

            # Hourly weather update
            simulated_minutes = step * sim.dt
            current_hour      = int(simulated_minutes // 60)
            if current_hour != last_hour and df_hourly is not None and len(df_hourly) > 0:
                last_hour = current_hour
                h_idx = min(current_hour, len(df_hourly) - 1)
                hrow  = df_hourly.iloc[h_idx]
                w_upd = {
                    "wind_speed_ms":    float(hrow.wind_speed_ms),
                    "wind_direction":   float(hrow.wind_direction),
                    "temperature_c":    float(hrow.temperature_c),
                    "relative_humidity":float(hrow.relative_humidity),
                    "_dir_is_from":     True,
                }
                apply_weather_to_landscape(land, w_upd)
                sim._precompute_ros_grid()
                log.info("Hour %d weather update applied", current_hour)
                weather_msg = {
                    "type":             "weather_update",
                    "hour":             current_hour,
                    "wind_speed_ms":    round(float(hrow.wind_speed_ms),     2),
                    "wind_direction":   round(float(hrow.wind_direction),     1),
                    "temperature_c":    round(float(hrow.temperature_c),     1),
                    "relative_humidity":round(float(hrow.relative_humidity), 1),
                }
                await send(weather_msg)
                await send(fire_info.weather_update(
                    weather_msg["wind_speed_ms"],
                    weather_msg["wind_direction"],
                    weather_msg["temperature_c"],
                    weather_msg["relative_humidity"],
                ))

            # Run simulation steps
            for _ in range(ctrl["steps_per_send"]):
                sim.step()
                step += 1

            burned             = int((sim.state == 2).sum())
            active             = int((sim.state == 1).sum())
            burned_ha          = round(_ha(burned, cell_m), 2)
            active_ha          = round(_ha(active, cell_m), 2)
            simulated_minutes  = step * sim.dt
            wall_clock_minutes = round(ignition_wall_hour * 60 + simulated_minutes, 1)

            # Store history every 10 steps (max 400 entries)
            if step % 10 == 0 and len(sim_history) < 400:
                gjb, gjd = _state_to_geojson(sim.state, geo_grid, rows, cols)
                sim_history.append((
                    step, gjb, gjd,
                    round(_ha(burned, cell_m), 2),
                    round(_ha(active, cell_m), 2),
                    wall_clock_minutes,
                ))

            # Throttle: only send frame if >= 50 ms since last send
            now = time.monotonic()
            if now - last_send_time >= MIN_FRAME_INTERVAL_S:
                ros_n, ros_e, ros_s, ros_w = _quadrant_ros(
                    sim, sim.state, rows, cols, cell_m)
                gjb, gjd = _state_to_geojson(sim.state, geo_grid, rows, cols)
                await send({
                    "type":                   "frame",
                    "step":                   step,
                    "geojson_burning":        gjb,
                    "geojson_burned":         gjd,
                    "burned_ha":              burned_ha,
                    "active_ha":              active_ha,
                    "minutes_since_ignition": round(simulated_minutes, 1),
                    "wall_clock_minutes":     wall_clock_minutes,
                    "ros_N":                  round(ros_n, 2),
                    "ros_E":                  round(ros_e, 2),
                    "ros_S":                  round(ros_s, 2),
                    "ros_W":                  round(ros_w, 2),
                })
                for ev in fire_info.process_frame(
                    burned_ha=burned_ha,
                    active_ha=active_ha,
                    minutes_since_ignition=round(simulated_minutes, 1),
                    ros={"N": ros_n, "E": ros_e, "S": ros_s, "W": ros_w},
                ):
                    await send(ev)
                last_send_time = now

            await asyncio.sleep(STEP_SLEEP_S)

            if step > 200 and active == 0:
                log.info("Fire extinguished at step %d", step)
                await send({"type": "status",
                            "message": f"Fire extinguished at step {step}."})
                break

        ctrl["stop"] = True

        # Build and send complete history for client-side playback
        history_frames = []
        for h_step, h_gjb, h_gjd, h_b_ha, h_a_ha, h_wcm in sim_history:
            history_frames.append({
                "step":               h_step,
                "geojson_burning":    h_gjb,
                "geojson_burned":     h_gjd,
                "burned_ha":          h_b_ha,
                "active_ha":          h_a_ha,
                "wall_clock_minutes": h_wcm,
            })
        await send({
            "type":   "history_ready",
            "count":  len(history_frames),
            "frames": history_frames,
        })
        await send(fire_info.simulation_completed(len(history_frames)))

    # Listener task
    async def listen_interventions():
        async for raw_msg in websocket:
            try:
                iv       = json.loads(raw_msg)
                msg_type = iv.get("type", "")

                if msg_type == "scrub":
                    # Auto-pause the sim so the next frame doesn't overwrite the scrub
                    if not ctrl["paused"]:
                        ctrl["paused"] = True
                        await send({"type": "status", "message": "Scrubbing - paused."})

                    idx = max(0, min(int(iv.get("index", 0)), len(sim_history) - 1))
                    if sim_history:
                        h_step, h_gjb, h_gjd, h_b_ha, h_a_ha, h_wcm = sim_history[idx]
                        await send({
                            "type":               "scrub_frame",
                            "index":              idx,
                            "step":               h_step,
                            "geojson_burning":    h_gjb,
                            "geojson_burned":     h_gjd,
                            "burned_ha":          h_b_ha,
                            "active_ha":          h_a_ha,
                            "wall_clock_minutes": h_wcm,
                            "count":              len(sim_history),
                        })
                    else:
                        await send({"type": "status",
                                    "message": "No history snapshots yet."})

                elif msg_type in ("pause", "resume", "set_speed", "intervention"):
                    await intervention_queue.put(iv)
                else:
                    log.debug("Unexpected message type: %s", msg_type)

            except Exception as exc:
                log.warning("Bad message: %s", exc)

            if ctrl["stop"]:
                break

    # Run both tasks concurrently
    sim_task    = asyncio.create_task(run_simulation())
    listen_task = asyncio.create_task(listen_interventions())

    done, pending = await asyncio.wait(
        [sim_task, listen_task],
        return_when=asyncio.ALL_COMPLETED,
    )

    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    for t in done:
        if t.exception():
            log.error("Simulation task raised: %s", t.exception())
            await send({"type": "error", "message": str(t.exception())})

    log.info("Connection closed  %s", websocket.remote_address)


# 
# Entry point
# 

async def main():
    log.info("Project WILSON WebSocket server starting on ws://%s:%d", HOST, PORT)
    async with websockets.serve(_handle, HOST, PORT):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped.")

"""
Project WILSON — WebSocket Simulation Server
=============================================
Runs a WebSocket server on ws://localhost:8765.

Protocol
--------
 Client → Server
  {"type": "init",         "lat_center": float, "lon_center": float,
   "bbox_degrees": float,  "ignition_points": [{"lat": f, "lon": f}, ...],
   "wind_speed_ms": float, "wind_dir_deg": float,
   "temperature_c": float, "relative_humidity": float,
   "ignition_time_utc": "HH:MM"}

  {"type": "intervention", "action": "containment_line",
   "points": [...], "strength": f, "effect_radius_m": f,
   "water_application": f, "humidity_boost": f, "decay_hours": f}
  {"type": "intervention", "action": "water_drop",
   "lat": f, "lon": f, "radius_m": f}
  {"type": "intervention", "action": "water_brush",
   "lat": f, "lon": f, "radius_m": f, "intensity": f, "falloff": f, "hardness": f}
  {"type": "intervention", "action": "pause" | "resume"}
  {"type": "scrub", "index": int}
  {"type": "cell_click", "lat": f, "lon": f}
  {"type": "rect_analysis", "lat_min": f, "lat_max": f, "lon_min": f, "lon_max": f}

 Server → Client
  {"type": "frame", ...}
  {"type": "cell_explanation", ...}
  {"type": "rect_analysis_result", ...}
  {"type": "microclimate_update", ...}
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
from core.microclimate import MicroclimateLearner
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
MIN_FRAME_INTERVAL_S   = 0.09

logging.basicConfig(
    level=logging.INFO,
    format="[WILSON-WS] %(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wilson_ws")

# Module-level microclimate singleton — persists across WebSocket connections
_microclimate: MicroclimateLearner | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ha(cells: int, cell_size_m: float) -> float:
    return cells * cell_size_m ** 2 / 10_000.0


def _b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).ravel().tobytes()).decode("ascii")


def _quadrant_ros(sim, state: np.ndarray,
                  rows: int, cols: int, cell_m: float) -> tuple:
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
        _qros(active & (row_g <  cr)),
        _qros(active & (col_g >  cc)),
        _qros(active & (row_g >  cr)),
        _qros(active & (col_g <  cc)),
    )


def _state_to_geojson(state: np.ndarray, geo_grid: GeoGrid,
                      rows: int, cols: int) -> tuple[dict, dict]:
    """RLE-based GeoJSON for burning and burned cells."""
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
            padded       = np.empty(cols + 2, dtype=np.bool_)
            padded[0]    = False
            padded[-1]   = False
            padded[1:-1] = row_mask
            diff     = np.diff(padded.view(np.int8))
            c_starts = np.where(diff  == 1)[0]
            c_ends   = np.where(diff == -1)[0]

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
    from core.fuels import FUEL_DISPLAY_COLORS, FUEL_DISPLAY_COLOR_DEFAULT

    lon_min = geo_grid.lon_min
    lat_min = geo_grid.lat_min
    dlon    = (geo_grid.lon_max - lon_min) / cols
    dlat    = (geo_grid.lat_max - lat_min) / rows

    def _rgba_to_hex(rgba: tuple) -> str:
        return "#{:02x}{:02x}{:02x}".format(rgba[0], rgba[1], rgba[2])

    features = []
    fuel_map   = land.fuel_map
    fuel_names = land.fuel_names
    color_lookup: dict[int, str] = {}
    for idx, name in enumerate(fuel_names):
        rgba = FUEL_DISPLAY_COLORS.get(name, FUEL_DISPLAY_COLOR_DEFAULT)
        color_lookup[idx] = _rgba_to_hex(rgba)

    for r in range(rows):
        row = fuel_map[r]
        lat_s = lat_min + r * dlat
        lat_n = lat_s + dlat
        padded = np.empty(cols + 2, dtype=np.int32)
        padded[0] = -1; padded[-1] = -1; padded[1:-1] = row
        diff   = np.diff(padded)
        starts = np.where(diff != 0)[0]
        for si in range(len(starts)):
            cs = int(starts[si])
            if cs >= cols:
                break
            ce = min(int(starts[si + 1]) if si + 1 < len(starts) else cols, cols)
            fuel_idx = int(row[cs])
            if fuel_idx < 0 or fuel_idx >= len(fuel_names):
                continue
            name  = fuel_names[fuel_idx]
            color = color_lookup.get(fuel_idx, "#808080")
            lon_w = lon_min + cs * dlon
            lon_e = lon_min + ce * dlon
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


def _apply_optimizer_params(land: Landscape, sim: "CellularAutomataFire",
                             best_params: dict) -> None:
    """
    Apply PSO best-params dict to an already-built sim + landscape.

    Called after CellularAutomataFire.__init__ so that per-fuel ROS multipliers,
    canopy WAF, spotting rate, and ignition threshold are all patched in.
    p_spread is rebuilt via _precompute_ros_grid() whenever wind changes.
    """
    from pipeline.particle_optimizer import _ros_mult_for_fuel, PARAM_NAMES, PARAM_BOUNDS

    wind_mult        = float(best_params.get("wind_multiplier",       1.0))
    moisture_offset  = float(best_params.get("moisture_offset",       0.0))
    wind_dir_offset  = float(best_params.get("wind_direction_offset", 0.0))
    canopy_wf        = float(best_params.get("canopy_wind_factor",    1.0))
    slope_mult       = float(best_params.get("slope_factor_mult",     1.0))
    spotting_mult    = float(best_params.get("spotting_rate_mult",    1.0))
    ign_threshold    = float(best_params.get("ignition_threshold",    1.0))

    # Landscape-level changes (wind + moisture)
    base_speed = float(getattr(land, "wind_speed", 5.0))
    base_dir   = float(getattr(land, "wind_dir",   0.0))
    land.set_wind(base_speed * wind_mult, (base_dir + wind_dir_offset) % 360.0)
    land.moisture = np.clip(land.moisture + moisture_offset, 0.01, 0.35).astype(np.float32)

    # Rebuild p_spread with modified wind/moisture
    sim._precompute_ros_grid()

    # Canopy wind factor: scale midflame wind field and rebuild p_spread
    if abs(canopy_wf - 1.0) > 1e-4 and hasattr(sim, "_wind_u_grid"):
        sim._wind_u_grid = np.clip(sim._wind_u_grid * canopy_wf, -100.0, 100.0).astype(np.float32)
        sim._wind_v_grid = np.clip(sim._wind_v_grid * canopy_wf, -100.0, 100.0).astype(np.float32)
        sim._precompute_ros_grid()

    # Slope factor: scale upslope p_spread directions
    if abs(slope_mult - 1.0) > 1e-4 and hasattr(sim, "p_spread"):
        import math as _math
        elev       = land.elevation
        cell_m_loc = float(getattr(land.config, "CELL_SIZE_METERS", 50.0))
        _nb_dirs   = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for i_nb, (dr_nb, dc_nb) in enumerate(_nb_dirs):
            src_elev = np.roll(np.roll(elev, dr_nb, axis=0), dc_nb, axis=1)
            dz       = elev - src_elev
            dist_m   = cell_m_loc * _math.sqrt(dr_nb**2 + dc_nb**2)
            tan_phi  = np.clip(dz / max(dist_m, 1e-3), 0.0, None)
            boost    = 1.0 + (slope_mult - 1.0) * (tan_phi > 0.01).astype(np.float32)
            sim.p_spread[i_nb] = np.clip(sim.p_spread[i_nb] * boost, 0.0, 1.0)

    # Per-fuel ROS multipliers
    params_arr = np.array(
        [best_params.get(name, (lo + hi) * 0.5)
         for name, (lo, hi) in zip(PARAM_NAMES, PARAM_BOUNDS)],
        dtype=np.float64
    )
    for fuel_idx, fname in enumerate(land.fuel_names):
        mult = _ros_mult_for_fuel(fname, params_arr)
        if abs(mult - 1.0) > 1e-6:
            mask = (land.fuel_map == fuel_idx)
            if mask.any():
                sim.p_spread[:, mask] = np.clip(sim.p_spread[:, mask] * mult, 0.0, 1.0)

    # Fire physics
    sim._spotting_rate_mult = spotting_mult
    sim._ignition_threshold = ign_threshold


def _init_sim_from_truth_mask(sim: "CellularAutomataFire",
                               truth_mask_geo: dict,
                               geo_grid: "GeoGrid",
                               rows: int, cols: int) -> tuple[int, int]:
    """
    Initialise sim state from a user-drawn ground-truth mask:
      • interior cells  → state 2 (burned)
      • perimeter cells → state 1 (burning)

    Returns (n_burning, n_burned).
    """
    from pipeline.particle_optimizer import _rasterize_geojson
    truth_mask = _rasterize_geojson(truth_mask_geo, geo_grid, rows, cols)
    if not truth_mask.any():
        return 0, 0

    # Erode by 2 cells to find robust interior vs perimeter
    try:
        from scipy.ndimage import binary_erosion
        interior  = binary_erosion(truth_mask, iterations=2)
    except ImportError:
        interior = truth_mask.copy()

    perimeter = truth_mask & ~interior

    # Interior → burned
    sim.state[interior]        = 2
    sim.burn_timer[interior]   = 0

    # Perimeter → burning (only combustible, unblocked cells)
    combustible_perim = perimeter & sim._combustible_mask & ~sim.blocked_mask
    sim.state[combustible_perim]       = 1
    sim.burn_timer[combustible_perim]  = sim._burn_time_steps
    sim._ignition_step[combustible_perim] = 0
    sim.ignition_fraction[truth_mask]  = 0.0

    return int(combustible_perim.sum()), int(interior.sum())


def _build_landscape(msg: dict) -> tuple[Landscape, GeoGrid, float]:
    lat_c = float(msg["lat_center"])
    lon_c = float(msg["lon_center"])
    half  = float(msg["bbox_degrees"])

    lat_min = lat_c - half;  lat_max = lat_c + half
    lon_min = lon_c - half;  lon_max = lon_c + half

    import types as _types
    cfg = _types.SimpleNamespace(**{
        k: v for k, v in vars(wilson_config).items()
        if not k.startswith("__") and not isinstance(v, _types.ModuleType)
    })
    cfg.GRID_SIZE         = (800, 800)
    cfg.TEMPERATURE_C     = float(msg.get("temperature_c",     35.0))
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
            lon_min, lat_min, lon_max, lat_max, width=800, height=800)
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
        log.info("Synthetic terrain (cell=%.0f m)", cfg.CELL_SIZE_METERS)

    weather = {
        "wind_speed_ms":     float(msg["wind_speed_ms"]),
        "wind_direction":    float(msg["wind_dir_deg"]),
        "temperature_c":     float(msg.get("temperature_c", 35.0)),
        "relative_humidity": float(msg.get("relative_humidity", 30.0)),
        "_dir_is_from":      True,
    }
    apply_weather_to_landscape(land, weather)

    if not real_loaded:
        land.moisture[:] = 0.05

    geo_grid = GeoGrid(lat_min, lat_max, lon_min, lon_max, *land.shape)
    return land, geo_grid, float(cfg.CELL_SIZE_METERS)


# ─────────────────────────────────────────────────────────────────────────────
# Intervention helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bresenham_line(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    """Draw a 1-pixel-wide Bresenham line on a boolean grid in-place."""
    rows, cols = grid.shape
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = True
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc


def _apply_water_drop(land: Landscape, sim: CellularAutomataFire,
                      geo_grid: GeoGrid,
                      lat: float, lon: float,
                      radius_m: float, cell_m: float) -> int:
    """Vectorised circular water drop."""
    rows, cols = land.shape
    rc, cc = geo_grid.latlon_to_rc(lat, lon)
    radius_cells = max(1.0, radius_m / cell_m)

    rr     = np.arange(rows, dtype=np.float32) - rc
    cc_arr = np.arange(cols, dtype=np.float32) - cc
    dist   = np.hypot(rr[:, None], cc_arr[None, :])
    wet_mask = dist <= radius_cells

    sat_moisture = min(
        land.config.RELATIVE_HUMIDITY / 100.0 * 0.35 + 0.20, 0.35)
    land.moisture[wet_mask] = np.maximum(land.moisture[wet_mask], sat_moisture)

    if wet_mask.any():
        if hasattr(sim, "apply_water_mask"):
            sim.apply_water_mask(wet_mask, wetness=0.92)
        else:
            active = wet_mask & (sim.state == 1)
            sim.state[active] = 0
            sim.burn_timer[active] = 0
            sim.ignition_fraction[wet_mask & (sim.state != 2)] = 0.0

    if wet_mask.any() and hasattr(sim, "_precompute_ros_grid"):
        sim._precompute_ros_grid()

    return int(wet_mask.sum())


def _apply_water_brush(land: Landscape, sim: CellularAutomataFire,
                       geo_grid: GeoGrid,
                       lat: float, lon: float,
                       radius_m: float, cell_m: float,
                       intensity: float = 1.0,
                       falloff: float = 0.8,
                       hardness: float = 0.5) -> int:
    rc, cc = geo_grid.latlon_to_rc(lat, lon)
    radius_cells = max(1.0, radius_m / cell_m)
    if hasattr(sim, "apply_water_brush"):
        return sim.apply_water_brush(
            int(rc), int(cc), radius_cells,
            intensity=float(intensity),
            falloff=float(falloff),
            hardness=float(hardness),
        )
    return _apply_water_drop(land, sim, geo_grid, lat, lon, radius_m, cell_m)


def _apply_containment_line(land: Landscape, sim: CellularAutomataFire,
                             geo_grid: GeoGrid,
                             points: list[dict],
                             strength: float,
                             effect_radius_m: float,
                             cell_m: float,
                             water_application: float = 0.0,
                             humidity_boost: float = 0.0,
                             decay_hours: float = 0.0) -> int:
    """
    Rasterise a containment polyline and apply graduated damping to the model.

    Replaces both the old _apply_firebreak and _apply_suppression_line.
    The effect is purely physics-based (heat-transfer reduction, moisture boost,
    optional water) — no fuel cells are cleared.
    """
    rows, cols = land.shape
    line_grid = np.zeros((rows, cols), dtype=bool)

    for i in range(len(points) - 1):
        r0, c0 = geo_grid.latlon_to_rc(points[i]["lat"],     points[i]["lon"])
        r1, c1 = geo_grid.latlon_to_rc(points[i+1]["lat"],   points[i+1]["lon"])
        _bresenham_line(line_grid, r0, c0, r1, c1)

    effect_radius_cells = max(1, int(round(effect_radius_m / cell_m)))

    # Convert decay_hours → decay per simulation step
    # sim.dt is in minutes; decay over the full lifetime means:
    #   steps_lifetime = decay_hours * 60 / sim.dt
    #   decay_per_step = strength / steps_lifetime (linear to zero)
    dt_minutes = float(getattr(sim, 'dt', 0.05))
    if decay_hours > 0.0 and dt_minutes > 0.0:
        steps_lifetime = max(1.0, decay_hours * 60.0 / dt_minutes)
        decay_rate = float(strength) / steps_lifetime
    else:
        decay_rate = 0.0

    if hasattr(sim, "apply_containment_line"):
        return sim.apply_containment_line(
            line_grid,
            strength=float(strength),
            effect_radius_cells=effect_radius_cells,
            water_application=float(water_application),
            humidity_boost=float(humidity_boost),
            decay_rate=decay_rate,
        )
    # Fallback for older model
    if hasattr(sim, "apply_suppression_line"):
        return sim.apply_suppression_line(
            line_grid,
            strength=float(strength),
            effect_radius_cells=effect_radius_cells,
        )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Particle-swarm optimizer task
# ─────────────────────────────────────────────────────────────────────────────

async def _run_optimizer_task(send, msg_init: dict, opt_msg: dict,
                               ctrl: dict) -> None:
    global _microclimate
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

    # Serialize current microclimate so the optimizer thread can use it.
    # The module-level singleton may have been built by previous simulation runs.
    mc_json = None
    mc_runs = 0
    if _microclimate is not None:
        try:
            mc_json = _microclimate.to_json()
            mc_runs = _microclimate._runs
        except Exception as exc:
            log.warning("Microclimate serialization for optimizer failed: %s", exc)

    mc_note = f" · 🧠 microclimate: {mc_runs} runs" if mc_runs >= 1 else " · no microclimate yet"
    await send({"type": "optimizer_status",
                "message": (
                    f"Optimizer starting: {n_particles} particles × {n_iterations} iterations"
                    f" — fire age {elapsed_hours:.2f} h{mc_note}…"
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
            microclimate_json=mc_json,
        )
    except Exception as exc:
        log.error("Optimizer failed: %s", exc)
        traceback.print_exc()
        await progress_queue.put(None)
        await drain_task
        await send({"type": "error", "message": f"Optimizer failed: {exc}"})
        ctrl["optimizer_running"] = False
        return

    await progress_queue.put(None)
    await drain_task

    # ── Merge optimizer-derived microclimate into the global singleton ────────
    opt_mc_json = result.get("optimizer_microclimate_json")
    if opt_mc_json:
        try:
            opt_mc = MicroclimateLearner.from_json(opt_mc_json)
            if _microclimate is None:
                _microclimate = opt_mc
                log.info("Microclimate initialised from optimizer (%d obs)", opt_mc._runs)
            else:
                # Blend: resize to match existing learner, then rolling-average ignition_freq
                mc_current = _microclimate
                opt_resized = opt_mc.resize_to(mc_current.rows, mc_current.cols)
                alpha = float(opt_resized._runs) / max(mc_current._runs + opt_resized._runs, 1)
                mc_current.ignition_freq = np.clip(
                    (1.0 - alpha) * mc_current.ignition_freq
                    + alpha * opt_resized.ignition_freq,
                    0.0, 1.0,
                ).astype(np.float32)
                mc_current._runs += opt_resized._runs
                _microclimate = mc_current
                log.info("Microclimate updated from optimizer (%d new + %d existing = %d total)",
                         opt_resized._runs, mc_current._runs - opt_resized._runs, mc_current._runs)
            mc_stats = _microclimate.get_region_stats(
                0, _microclimate.rows - 1, 0, _microclimate.cols - 1
            )
            await send({
                "type":    "microclimate_update",
                "runs":    _microclimate._runs,
                "stats":   mc_stats,
                "message": (
                    f"Microclimate updated from optimizer "
                    f"({opt_mc._runs} PSO sims, total {_microclimate._runs} observations)."
                ),
            })
        except Exception as exc:
            log.warning("Optimizer microclimate merge failed: %s", exc)

    geo = result["geo_grid"]
    await send({
        "type":              "optimizer_result",
        "best_params":       result["best_params"],
        "best_iou":          round(result["best_iou"], 4),
        "heatmap_png_b64":   result.get("heatmap_png_b64", ""),
        "lat_min":           geo.lat_min,
        "lat_max":           geo.lat_max,
        "lon_min":           geo.lon_min,
        "lon_max":           geo.lon_max,
        "microclimate_used": result.get("microclimate_used", False),
        "microclimate_runs": result.get("microclimate_runs", 0),
    })
    log.info("Optimizer complete — best IoU %.3f  (microclimate: %d runs used)",
             result["best_iou"], result.get("microclimate_runs", 0))
    ctrl["optimizer_running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer-only session
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_optimizer_session(websocket, send, msg_init: dict) -> None:
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
                        _run_optimizer_task(send, msg_init, iv, ctrl))
            else:
                log.debug("Optimizer session: unexpected msg type '%s'", msg_type)
        except Exception as exc:
            log.warning("Optimizer session: bad message: %s", exc)

    log.info("Optimizer session closed  %s", websocket.remote_address)


# ─────────────────────────────────────────────────────────────────────────────
# Per-connection handler
# ─────────────────────────────────────────────────────────────────────────────

async def _handle(websocket):
    global _microclimate
    log.info("Client connected  %s", websocket.remote_address)

    async def send(obj: dict):
        await websocket.send(json.dumps(obj))

    # ── Wait for 'init' ───────────────────────────────────────────────────────
    try:
        raw  = await asyncio.wait_for(websocket.recv(), timeout=60.0)
        msg: dict[str, Any] = json.loads(raw)
    except asyncio.TimeoutError:
        await send({"type": "error", "message": "Timed out waiting for 'init' message."})
        return
    except Exception as exc:
        await send({"type": "error", "message": f"Bad message: {exc}"})
        return

    msg_type_recv = msg.get("type")

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

    # ── CFL-correct dt ────────────────────────────────────────────────────────
    half    = float(msg["bbox_degrees"])
    lat_c   = float(msg["lat_center"])
    lat_min = lat_c - half
    lat_max = lat_c + half
    base_dt     = 0.1
    coarse_cell = (lat_max - lat_min) * 111_320 / 400
    fine_cell   = (lat_max - lat_min) * 111_320 / 800
    dt_fine     = base_dt * (fine_cell / coarse_cell)

    try:
        sim = CellularAutomataFire(land, land.config, dt=dt_fine)
    except TypeError:
        sim = CellularAutomataFire(land, land.config)
        sim.dt = dt_fine

    rows, cols = land.shape

    # ── Apply optimizer best-params before microclimate (overwrites wind/moisture) ─
    optimizer_mode     = msg.get("optimizer_mode")        # "ground_truth" | "best_params_sim" | None
    best_params        = msg.get("best_params")            # dict from PSO result
    use_opt_params     = bool(msg.get("use_opt_params", False))
    elapsed_hours_opt  = float(msg.get("elapsed_hours", 6.0))
    truth_mask_geojson = msg.get("truth_mask_geojson")    # GeoJSON FeatureCollection

    if use_opt_params and best_params:
        try:
            _apply_optimizer_params(land, sim, best_params)
            log.info("Optimizer params applied (wind×%.2f  Δm=%.3f  dir%+.1f°)",
                     best_params.get("wind_multiplier", 1.0),
                     best_params.get("moisture_offset", 0.0),
                     best_params.get("wind_direction_offset", 0.0))
        except Exception as exc:
            log.warning("Optimizer params apply failed: %s", exc)

    # ── Apply microclimate corrections if available ───────────────────────────
    if _microclimate is not None:
        mc = _microclimate.resize_to(rows, cols)
        try:
            mc.apply_to_model(sim)
            log.info("Microclimate corrections applied (%d prior runs)", mc._runs)
        except Exception as exc:
            log.warning("Microclimate apply failed: %s", exc)

    # ── Parse ignition wall-clock time ────────────────────────────────────────
    ign_time_str = msg.get("ignition_time_utc", "12:00")
    try:
        ign_h, ign_m = [int(x) for x in ign_time_str.split(":")[:2]]
    except Exception:
        ign_h, ign_m = 12, 0
    ignition_wall_hour = ign_h + ign_m / 60.0

    ignited_count     = 0
    skipped_ignitions = 0

    # ── Ignition: depends on optimizer_mode ───────────────────────────────────
    if optimizer_mode == "ground_truth" and truth_mask_geojson:
        # Perimeter of truth mask burns; interior is already burned
        n_burning, n_burned = _init_sim_from_truth_mask(
            sim, truth_mask_geojson, geo_grid, rows, cols
        )
        if n_burning > 0 or n_burned > 0:
            ignited_count = 1
            log.info("Ground-truth init: %d burning + %d burned cells", n_burning, n_burned)
        else:
            log.warning("Ground-truth init: mask rasterized to 0 cells — using point fallback")
            optimizer_mode = None   # fall through to point ignition

    if optimizer_mode != "ground_truth":
        # Normal point ignition (also used by "best_params_sim" — fast-forward happens below)
        ignition_points = msg.get("ignition_points", [])
        if not ignition_points:
            ignition_points = [{"lat": msg["lat_center"], "lon": msg["lon_center"]}]

        for pt in ignition_points:
            rc = geo_grid.latlon_to_rc(float(pt["lat"]), float(pt["lon"]))
            rc = _snap_to_land(rc, land)
            if rc is None:
                skipped_ignitions += 1
                continue
            r0, c0 = rc
            radius = max(0, int(90.0 / max(cell_m, 1e-6)))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc > radius * radius:
                        continue
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        sim.ignite(rr, cc)
            ignited_count += 1

        # Fast-forward to elapsed_hours for "best_params_sim" mode
        if optimizer_mode == "best_params_sim" and elapsed_hours_opt > 0:
            ff_steps = max(1, int(elapsed_hours_opt * 60.0 / sim.dt))
            log.info("Fast-forwarding %d steps (%.1fh) to reach optimizer match-point…",
                     ff_steps, elapsed_hours_opt)
            for _ in range(ff_steps):
                sim.step()
                if not (sim.state == 1).any():
                    break
            log.info("Fast-forward done: %d burning, %d burned",
                     int((sim.state == 1).sum()), int((sim.state == 2).sum()))

    # ── Wind grid subsampling ─────────────────────────────────────────────────
    _ri   = np.linspace(0, rows - 1, 20, dtype=int)
    _ci   = np.linspace(0, cols - 1, 20, dtype=int)
    wu_20 = sim._wind_u_grid[np.ix_(_ri, _ci)]
    wv_20 = sim._wind_v_grid[np.ix_(_ri, _ci)]

    # ── Optional hourly weather ───────────────────────────────────────────────
    df_hourly = None
    hourly_records: list[dict] = []
    _date_start = msg.get("date_start", "")
    if _date_start:
        try:
            _date_end = msg.get("date_end", _date_start)
            df_hourly = fetch_weather_hourly(lat_c, float(msg["lon_center"]),
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

    log.info("Building vegetation GeoJSON ...")
    geojson_veg = _fuel_map_to_geojson(land, geo_grid, rows, cols)
    log.info("Vegetation GeoJSON: %d features", len(geojson_veg["features"]))

    fire_info = FireInfoPanel()

    # ── Send init_ack ─────────────────────────────────────────────────────────
    mc_summary = None
    if _microclimate is not None:
        try:
            mc = _microclimate.resize_to(rows, cols)
            mc_summary = mc.get_region_stats(0, rows - 1, 0, cols - 1)
        except Exception:
            pass

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
        "microclimate":     mc_summary,
        "optimizer_mode":   optimizer_mode,
        "message":          (
            f"Simulation ready — {ignited_count} ignition point(s)"
            + (f" [mode: {optimizer_mode}]" if optimizer_mode else "")
            + ". Starting..."
            if skipped_ignitions == 0 else
            f"Simulation ready — {ignited_count} ignition point(s), "
            f"{skipped_ignitions} skipped (non-burnable). Starting..."
        ),
    })
    await send(fire_info.init_started(rows, cols, cell_m))

    # ── Shared control state ──────────────────────────────────────────────────
    # Start paused for "best_params_sim" so the user can inspect the fire at
    # the elapsed-hours snapshot before choosing to continue.
    _start_paused = (optimizer_mode == "best_params_sim")

    ctrl = {
        "paused":         _start_paused,
        "stop":           False,
        "steps_per_send": DEFAULT_STEPS_PER_SEND,
    }
    intervention_queue: asyncio.Queue = asyncio.Queue()
    sim_history: list = []

    # ── Simulation task ───────────────────────────────────────────────────────
    async def run_simulation():
        step           = 0
        last_hour      = -1
        last_send_time = 0.0

        # If started paused (best_params_sim mode), send a single snapshot
        # frame so the client can render the fire at the elapsed-hours point,
        # then wait for the user to press Resume.
        if _start_paused:
            _burned  = int((sim.state == 2).sum())
            _active  = int((sim.state == 1).sum())
            _mins    = round(elapsed_hours_opt * 60.0, 1)
            _wall    = round(ignition_wall_hour * 60.0 + _mins, 1)
            _gjb, _gjd = _state_to_geojson(sim.state, geo_grid, rows, cols)
            await send({
                "type":                   "frame",
                "step":                   0,
                "geojson_burning":        _gjb,
                "geojson_burned":         _gjd,
                "burned_ha":              round(_ha(_burned, cell_m), 2),
                "active_ha":              round(_ha(_active, cell_m), 2),
                "minutes_since_ignition": _mins,
                "wall_clock_minutes":     _wall,
                "ros_N": 0.0, "ros_E": 0.0, "ros_S": 0.0, "ros_W": 0.0,
            })
            await send({
                "type":    "status",
                "message": f"Fire at {elapsed_hours_opt:.1f}h mark — press Resume to continue.",
                "paused":  True,
            })

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

                elif action in ("containment_line", "suppression_line", "firebreak"):
                    # Unified containment handler — "firebreak" mapped here for compat
                    n = _apply_containment_line(
                        land, sim, geo_grid,
                        iv.get("points", []),
                        float(iv.get("strength", 0.7)),
                        float(iv.get("effect_radius_m", 100.0)),
                        cell_m,
                        water_application=float(iv.get("water_application", 0.0)),
                        humidity_boost=float(iv.get("humidity_boost", 0.0)),
                        decay_hours=float(iv.get("decay_hours", 0.0)),
                    )
                    await send({"type": "status",
                                "message": f"Containment line deployed: {n} cells protected."})
                    await send(fire_info.intervention("containment_line", n))

                elif action == "water_drop":
                    n = _apply_water_drop(land, sim, geo_grid,
                                          float(iv["lat"]), float(iv["lon"]),
                                          float(iv.get("radius_m", 435.0)), cell_m)
                    await send({"type": "status",
                                "message": f"Water drop: {n} cells affected."})
                    await send(fire_info.intervention("water_drop", n))

                elif action == "water_brush":
                    _apply_water_brush(
                        land, sim, geo_grid,
                        float(iv["lat"]), float(iv["lon"]),
                        float(iv.get("radius_m", 200.0)), cell_m,
                        intensity=float(iv.get("intensity", 1.0)),
                        falloff=float(iv.get("falloff", 0.8)),
                        hardness=float(iv.get("hardness", 0.5)),
                    )

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

            for _ in range(ctrl["steps_per_send"]):
                sim.step()
                step += 1

            burned             = int((sim.state == 2).sum())
            active             = int((sim.state == 1).sum())
            burned_ha          = round(_ha(burned, cell_m), 2)
            active_ha          = round(_ha(active, cell_m), 2)
            simulated_minutes  = step * sim.dt
            wall_clock_minutes = round(ignition_wall_hour * 60 + simulated_minutes, 1)

            if step % 10 == 0 and len(sim_history) < 400:
                gjb, gjd = _state_to_geojson(sim.state, geo_grid, rows, cols)
                sim_history.append((
                    step, gjb, gjd,
                    round(_ha(burned, cell_m), 2),
                    round(_ha(active, cell_m), 2),
                    wall_clock_minutes,
                ))

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

        # Record microclimate observations
        global _microclimate
        try:
            if _microclimate is None:
                _microclimate = MicroclimateLearner(rows, cols)
            else:
                _microclimate = _microclimate.resize_to(rows, cols)
            _microclimate.record_simulation(sim, land)
            mc_stats = _microclimate.get_region_stats(0, rows - 1, 0, cols - 1)
            await send({
                "type":       "microclimate_update",
                "runs":       _microclimate._runs,
                "stats":      mc_stats,
                "message":    f"Microclimate updated from run #{_microclimate._runs}.",
            })
            log.info("Microclimate updated — %d total runs", _microclimate._runs)
        except Exception as exc:
            log.warning("Microclimate record failed: %s", exc)

        # Build history for playback
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

    # ── Listener task ─────────────────────────────────────────────────────────
    async def listen_interventions():
        async for raw_msg in websocket:
            try:
                iv       = json.loads(raw_msg)
                msg_type = iv.get("type", "")

                if msg_type == "scrub":
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

                elif msg_type == "cell_click":
                    try:
                        lat    = float(iv.get("lat", 0))
                        lon    = float(iv.get("lon", 0))
                        rc_ex, cc_ex = geo_grid.latlon_to_rc(lat, lon)
                        expl   = sim.get_cell_explanation(int(rc_ex), int(cc_ex))
                        expl["lat"] = lat
                        expl["lon"] = lon
                        await send({"type": "cell_explanation", **expl})
                    except Exception as exc:
                        await send({"type": "cell_explanation",
                                    "error": str(exc), "row": -1, "col": -1})

                elif msg_type == "rect_analysis":
                    try:
                        lat_min_r = float(iv.get("lat_min", 0))
                        lat_max_r = float(iv.get("lat_max", 0))
                        lon_min_r = float(iv.get("lon_min", 0))
                        lon_max_r = float(iv.get("lon_max", 0))

                        # Convert to grid coords (note: lat_max_r → smaller row index in south-up grid)
                        r_a, c_a = geo_grid.latlon_to_rc(lat_max_r, lon_min_r)
                        r_b, c_b = geo_grid.latlon_to_rc(lat_min_r, lon_max_r)
                        r0_r = max(0, min(r_a, r_b))
                        r1_r = min(rows - 1, max(r_a, r_b))
                        c0_r = max(0, min(c_a, c_b))
                        c1_r = min(cols - 1, max(c_a, c_b))

                        stats = sim.get_region_stats(r0_r, r1_r, c0_r, c1_r)

                        # Dominant fuel in region
                        fuel_sub = land.fuel_map[r0_r:r1_r+1, c0_r:c1_r+1]
                        fuel_counts = {}
                        for idx_f in fuel_sub.ravel().tolist():
                            fuel_counts[idx_f] = fuel_counts.get(idx_f, 0) + 1
                        if fuel_counts:
                            dom_idx = max(fuel_counts, key=fuel_counts.get)
                            dominant_fuel = (land.fuel_names[dom_idx]
                                             if dom_idx < len(land.fuel_names)
                                             else "Unknown")
                        else:
                            dominant_fuel = "Unknown"

                        stats["dominant_fuel"] = dominant_fuel
                        stats["burned_ha"]  = round(_ha(stats["burned_cells"],  cell_m), 2)
                        stats["burning_ha"] = round(_ha(stats["burning_cells"], cell_m), 2)

                        # Microclimate knowledge for this region
                        mc_region = None
                        if _microclimate is not None:
                            try:
                                mc = _microclimate.resize_to(rows, cols)
                                mc_region = mc.get_region_stats(r0_r, r1_r, c0_r, c1_r)
                            except Exception:
                                pass

                        await send({
                            "type":       "rect_analysis_result",
                            "stats":      stats,
                            "microclimate": mc_region,
                            "region":     {"r0": r0_r, "r1": r1_r, "c0": c0_r, "c1": c1_r},
                        })
                    except Exception as exc:
                        await send({"type": "rect_analysis_result",
                                    "error": str(exc)})

                elif msg_type in ("pause", "resume", "set_speed", "intervention"):
                    await intervention_queue.put(iv)
                else:
                    log.debug("Unexpected message type: %s", msg_type)

            except Exception as exc:
                log.warning("Bad message: %s", exc)

            if ctrl["stop"]:
                break

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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    log.info("Project WILSON WebSocket server starting on ws://%s:%d", HOST, PORT)
    async with websockets.serve(_handle, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped.")

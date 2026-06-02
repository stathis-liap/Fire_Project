"""
pipeline/particle_optimizer.py
================================
Particle Swarm Optimizer for fire model parameter calibration.

Searches a 12-dimensional parameter space that includes airflow, terrain,
and fire-physics variables, making air a first-class optimizer variable:

  Environmental / airflow:
    wind_multiplier         — scale ERA5 base wind speed
    moisture_offset         — shift all fuel moisture values
    wind_direction_offset   — rotation of base wind direction (degrees)
    canopy_wind_factor      — midflame wind reduction (WAF proxy)
    slope_factor_mult       — terrain slope amplification of ROS

  Fuel-type ROS multipliers:
    ros_mult_pine, ros_mult_shrub, ros_mult_grass,
    ros_mult_forest, ros_mult_orchard

  Fire physics:
    spotting_rate_mult      — firebrand lofting probability
    ignition_threshold      — heat accumulation needed to ignite (higher = slower)

Runs on a fast 200x200 grid (reuses cached DEM/CORINE files).
Particle evaluations are parallelised with ThreadPoolExecutor.
Returns the top-K burned masks as an ensemble probability heatmap (PNG).
"""

from __future__ import annotations

import base64
import copy
import io
import math
import sys
import types as _types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Ensure project root is importable when this file is run from a subdirectory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─── Parameter space ──────────────────────────────────────────────────────────

PARAM_NAMES = [
    # ── Environmental / airflow ────────────────────────────────────────────────
    "wind_multiplier",       # 0  scale ERA5 base wind speed (0.3–3.0)
    "moisture_offset",       # 1  shift all fuel moisture values (−0.15–+0.05)
    "wind_direction_offset", # 2  rotation of base wind direction in degrees (−45–+45)
    "canopy_wind_factor",    # 3  midflame wind reduction factor / WAF proxy (0.3–2.0)
    "slope_factor_mult",     # 4  terrain slope amplification of p_spread (0.3–3.0)
    # ── Fuel-type ROS multipliers ──────────────────────────────────────────────
    "ros_mult_pine",         # 5  pine / conifer fuels
    "ros_mult_shrub",        # 6  maquis / shrub fuels
    "ros_mult_grass",        # 7  grass / agricultural fuels
    "ros_mult_forest",       # 8  broadleaf forest fuels
    "ros_mult_orchard",      # 9  olive / orchard / riparian fuels
    # ── Fire physics ──────────────────────────────────────────────────────────
    "spotting_rate_mult",    # 10 firebrand lofting probability multiplier (0.1–5.0)
    "ignition_threshold",    # 11 heat accumulation needed for ignition (0.5–2.0)
]

PARAM_BOUNDS = [
    (0.3,   3.0),   # wind_multiplier
    (-0.15, 0.05),  # moisture_offset
    (-45.0, 45.0),  # wind_direction_offset  (degrees)
    (0.3,   2.0),   # canopy_wind_factor
    (0.3,   3.0),   # slope_factor_mult
    (0.3,   3.0),   # ros_mult_pine
    (0.3,   3.0),   # ros_mult_shrub
    (0.3,   3.0),   # ros_mult_grass
    (0.3,   3.0),   # ros_mult_forest
    (0.3,   3.0),   # ros_mult_orchard
    (0.1,   5.0),   # spotting_rate_mult
    (0.5,   2.0),   # ignition_threshold
]

# Fuel → parameter index mapping
_PINE_FUELS    = frozenset({"Aleppo_Pine", "Black_Pine", "Maritime_Pine",
                             "Stone_Pine", "Cypress", "Greek_Fir"})
_SHRUB_FUELS   = frozenset({"Maquis_Dense_Shrub", "Tall_Maquis",
                             "Phrygana_Low_Scrub", "Garrigue"})
_GRASS_FUELS   = frozenset({"Dry_Grass", "Annual_Crops", "Abandoned_Agricultural"})
_FOREST_FUELS  = frozenset({"Oak_Forest", "Chestnut_Forest", "Beech_Forest"})
_ORCHARD_FUELS = frozenset({"Olive_Grove", "Vineyard", "Eucalyptus",
                             "Riparian_Vegetation"})

# PSO hyper-parameters
_W_START   = 0.9   # inertia weight (decreasing linearly)
_W_END     = 0.4
_C1        = 1.5   # cognitive (personal-best) factor
_C2        = 1.5   # social (global-best) factor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ros_mult_for_fuel(name: str, params: np.ndarray) -> float:
    """Return the ROS multiplier for a given fuel name from the parameter vector."""
    if name in _PINE_FUELS:    return float(params[5])
    if name in _SHRUB_FUELS:   return float(params[6])
    if name in _GRASS_FUELS:   return float(params[7])
    if name in _FOREST_FUELS:  return float(params[8])
    if name in _ORCHARD_FUELS: return float(params[9])
    return 1.0  # Non_Combustible, Water, Urban → no change


def _rasterize_geojson(geojson: dict, geo_grid, rows: int, cols: int) -> np.ndarray:
    """
    Rasterise a GeoJSON FeatureCollection (polygons) onto a binary south-up grid.
    Returns bool array shape (rows, cols), True where covered by any polygon.
    """
    try:
        import rasterio.features
        import rasterio.transform

        transform = rasterio.transform.from_bounds(
            geo_grid.lon_min, geo_grid.lat_min,
            geo_grid.lon_max, geo_grid.lat_max,
            cols, rows,
        )
        shapes = [
            (feat["geometry"], 1)
            for feat in geojson.get("features", [])
            if feat.get("geometry")
        ]
        if not shapes:
            return np.zeros((rows, cols), dtype=bool)

        grid = rasterio.features.rasterize(
            shapes,
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        # rasterio rasters top-down; our sim grid is south-up → flip vertically
        return np.flipud(grid).astype(bool)
    except Exception:
        return np.zeros((rows, cols), dtype=bool)


def _snap_to_burnable(rc: tuple, land) -> tuple | None:
    """Walk outward from rc in expanding rings until a burnable cell is found."""
    from core.fuels import NON_BURNING_FUELS
    rows, cols = land.shape
    r0, c0 = rc
    for radius in range(8):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < rows and 0 <= c < cols:
                    name = (land.fuel_names[land.fuel_map[r, c]]
                            if land.fuel_map[r, c] < len(land.fuel_names) else "")
                    if name not in NON_BURNING_FUELS:
                        return r, c
    return None


def heatmap_to_png_b64(heatmap: np.ndarray) -> str:
    """
    Convert a probability grid [0, 1] to a base64-encoded RGBA PNG.
    Colormap: transparent at 0 → blue → yellow → red at 1.
    The grid is flipped vertically (south-up → top-down for image convention).
    """
    try:
        from PIL import Image
    except ImportError:
        return ""

    h = np.flipud(heatmap.astype(np.float32))
    rows, cols = h.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

    # Red: ramps up from p=0.2 to p=1.0
    rgba[:, :, 0] = np.clip((h - 0.2) / 0.8 * 255, 0, 255).astype(np.uint8)
    # Green: peaks at p=0.5 (yellow in the middle)
    rgba[:, :, 1] = np.clip((1.0 - np.abs(h * 2.0 - 1.0)) * 210, 0, 210).astype(np.uint8)
    # Blue: high for low probability, fades to 0 by p=0.5
    rgba[:, :, 2] = np.clip((0.5 - h) * 2.0 * 210, 0, 210).astype(np.uint8)
    # Alpha: invisible for p < 0.05, fully visible at p > 0.3
    rgba[:, :, 3] = np.clip((h - 0.05) / 0.25 * 210, 0, 210).astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─── Single-particle evaluation (thread-safe) ─────────────────────────────────

def _evaluate_particle(
    land_base,
    cfg_base,
    geo_grid,
    cell_m: float,
    ignition_rcs: list[tuple[int, int]],
    truth_mask: np.ndarray,
    params: np.ndarray,
    sim_hours: float,
    microclimate=None,
) -> tuple[float, np.ndarray]:
    """
    Evaluate one particle.
    Deep-copies the landscape so concurrent threads don't share mutable state.
    Returns (iou_score, burned_mask_bool_array).
    sim_hours is the fixed fire duration supplied by the caller (user-specified
    elapsed time since ignition), NOT a search parameter.
    microclimate is an optional MicroclimateLearner (read-only) applied before ignition.
    """
    from core.fire_model import CellularAutomataFire

    land = copy.deepcopy(land_base)
    cfg  = copy.copy(cfg_base)

    wind_mult            = float(params[0])
    moisture_offset      = float(params[1])
    wind_dir_offset      = float(params[2])   # degrees, rotates base wind direction
    canopy_wind_factor   = float(params[3])   # WAF proxy: scales midflame wind field
    slope_factor_mult    = float(params[4])   # scales terrain-slope contribution
    # params[5..9] = per-fuel ROS multipliers (handled by _ros_mult_for_fuel)
    spotting_rate_mult   = float(params[10])
    ignition_threshold   = float(params[11])

    # ── Wind: apply speed multiplier and direction rotation ─────────────────
    # land.wind_speed / wind_dir are the scalar inputs; compute_wind_field()
    # reads these to build the spatially-variable field.
    base_speed = float(getattr(land, "wind_speed", 5.0))
    base_dir   = float(getattr(land, "wind_dir",   0.0))   # push-toward degrees
    land.wind_speed = base_speed * wind_mult
    land.set_wind(land.wind_speed, (base_dir + wind_dir_offset) % 360.0)

    # Shift fuel moisture
    land.moisture = np.clip(land.moisture + moisture_offset, 0.01, 0.99)

    rows, cols = land.shape

    # Derive dt consistent with grid resolution
    lat_span_m = (geo_grid.lat_max - geo_grid.lat_min) * 111_320
    dt_use = max(0.01, 0.1 * (lat_span_m / rows) / (lat_span_m / 400))

    try:
        sim = CellularAutomataFire(land, cfg, dt=dt_use)
    except TypeError:
        sim = CellularAutomataFire(land, cfg)
        sim.dt = dt_use

    # ── Canopy wind factor: scale the computed midflame wind field ───────────
    # This captures canopy sheltering / channelling effects not in the base model.
    if abs(canopy_wind_factor - 1.0) > 1e-4 and hasattr(sim, '_wind_u_grid'):
        sim._wind_u_grid = np.clip(sim._wind_u_grid * canopy_wind_factor,
                                   -100.0, 100.0).astype(np.float32)
        sim._wind_v_grid = np.clip(sim._wind_v_grid * canopy_wind_factor,
                                   -100.0, 100.0).astype(np.float32)
        # Recompute p_spread with the adjusted wind field
        sim._precompute_ros_grid()

    # ── Slope amplification: scale upslope p_spread ──────────────────────────
    if abs(slope_factor_mult - 1.0) > 1e-4 and hasattr(sim, 'p_spread'):
        elev = land.elevation
        cell_m_loc = float(getattr(cfg, 'CELL_SIZE_METERS', 50.0))
        neighbors_local = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for i_nb, (dr_nb, dc_nb) in enumerate(neighbors_local):
            src_elev = np.roll(np.roll(elev, dr_nb, axis=0), dc_nb, axis=1)
            dz = elev - src_elev    # positive = upslope (fire moving uphill)
            dist_m = cell_m_loc * math.sqrt(dr_nb**2 + dc_nb**2)
            tan_phi = np.clip(dz / max(dist_m, 1e-3), 0.0, None)
            # Only amplify upslope directions (phi_s > 0)
            upslope = (tan_phi > 0.01).astype(np.float32)
            boost = 1.0 + (slope_factor_mult - 1.0) * upslope
            sim.p_spread[i_nb] = np.clip(sim.p_spread[i_nb] * boost, 0.0, 1.0)

    # ── Per-fuel ROS multipliers ──────────────────────────────────────────────
    for fuel_idx, name in enumerate(land.fuel_names):
        mult = _ros_mult_for_fuel(name, params)
        if abs(mult - 1.0) > 1e-6:
            mask = (land.fuel_map == fuel_idx)
            if mask.any():
                sim.p_spread[:, mask] = np.clip(
                    sim.p_spread[:, mask] * mult, 0.0, 1.0
                )

    # ── Physics tuning parameters ─────────────────────────────────────────────
    sim._spotting_rate_mult = spotting_rate_mult
    sim._ignition_threshold = ignition_threshold

    # ── Microclimate corrections (read-only; thread-safe) ─────────────────────
    # Boosts p_spread in historically high-ignition corridors and adjusts the
    # wind field based on observed amplification across prior simulation runs.
    if microclimate is not None:
        try:
            microclimate.apply_to_model(sim)
        except Exception:
            pass

    # Ignite with a small initial radius matching the 90 m footprint
    for (r0, c0) in ignition_rcs:
        radius = max(0, int(90.0 / max(cell_m, 1e-6)))
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        sim.ignite(rr, cc)

    # ── Step to elapsed_hours — the known age of the ground-truth mask ──────
    n_steps_truth = max(10, int(sim_hours * 60.0 / sim.dt))
    for _ in range(n_steps_truth):
        sim.step()
        if not (sim.state == 1).any():
            break

    # Score IoU at the ground-truth time point
    pred_at_truth = (sim.state >= 1)
    inter = int((pred_at_truth & truth_mask).sum())
    union = int((pred_at_truth | truth_mask).sum())
    iou   = inter / max(union, 1)

    # ── Continue for 1 more hour to build the forward-projection ensemble ────
    # The ensemble heatmap shows where the calibrated fire is likely to spread
    # *beyond* the known perimeter, not just where it already is.
    n_steps_extra = max(1, int(60.0 / sim.dt))
    for _ in range(n_steps_extra):
        sim.step()
        if not (sim.state == 1).any():
            break

    pred_extended = (sim.state >= 1)
    return iou, pred_extended


# ─── Public API ───────────────────────────────────────────────────────────────

def run_particle_swarm(
    msg_init:           dict,
    truth_mask_geo:     dict,
    n_particles:        int = 20,
    n_iterations:       int = 12,
    top_k:              int = 50,
    n_workers:          int = 4,
    elapsed_hours:      float = 6.0,
    progress_cb:        Optional[Callable] = None,
    microclimate_json:  Optional[str] = None,
) -> dict:
    """
    Run Particle Swarm Optimisation to calibrate fire model parameters
    against a user-drawn ground-truth burned perimeter.

    Parameters
    ----------
    msg_init       : the WebSocket 'init_optimizer' message dict
    truth_mask_geo : GeoJSON FeatureCollection of ground-truth polygon(s)
    n_particles    : swarm size
    n_iterations   : number of PSO generations
    top_k          : number of best runs collected for the ensemble heatmap
    n_workers      : parallel evaluation threads
    elapsed_hours  : fixed fire duration (hours since ignition) — this is
                     the known wall-clock age of the ground-truth mask, so
                     we do NOT search over simulation time; every particle
                     runs for exactly this many hours and is scored against
                     the drawn perimeter
    progress_cb    : callable(iteration, n_done, n_total, best_iou, eta_s)

    Returns
    -------
    dict with best_params, best_iou, heatmap_png_b64, geo_grid, rows, cols, cell_m
    """
    import time as _time

    # ── Build 200×200 optimisation landscape ──────────────────────────────────
    import config as _wcfg

    lat_c  = float(msg_init["lat_center"])
    lon_c  = float(msg_init["lon_center"])
    half   = float(msg_init["bbox_degrees"])
    lat_min, lat_max = lat_c - half, lat_c + half
    lon_min, lon_max = lon_c - half, lon_c + half

    cfg = _types.SimpleNamespace(**{
        k: v for k, v in vars(_wcfg).items()
        if not k.startswith("__") and not isinstance(v, _types.ModuleType)
    })
    cfg.GRID_SIZE         = (200, 200)
    cfg.TEMPERATURE_C     = float(msg_init.get("temperature_c", 35.0))
    cfg.RELATIVE_HUMIDITY = float(msg_init.get("relative_humidity", 30.0))

    from core.landscape import Landscape
    from pipeline.hindcast_optimizer import apply_weather_to_landscape, GeoGrid

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
            lon_min, lat_min, lon_max, lat_max, width=200, height=200
        )
        osm_path = fetch_osm_features(lon_min, lat_min, lon_max, lat_max)
        if dem_path:
            land.load_real_terrain(
                dem_file=dem_path,
                corine_file=corine_path or "",
                target_shape=(200, 200),
                osm_file=osm_path,
            )
            real_loaded = True
    except Exception:
        pass

    if not real_loaded:
        land.generate_random_terrain()
        lat_span_m = (lat_max - lat_min) * 111_320
        lon_span_m = (lon_max - lon_min) * 111_320 * math.cos(math.radians(lat_c))
        cfg.CELL_SIZE_METERS = float(max(lat_span_m, lon_span_m) / 200)
        land.moisture[:] = 0.05

    weather = {
        "wind_speed_ms":     float(msg_init["wind_speed_ms"]),
        "wind_direction":    float(msg_init["wind_dir_deg"]),
        "temperature_c":     float(msg_init.get("temperature_c", 35.0)),
        "relative_humidity": float(msg_init.get("relative_humidity", 30.0)),
        "_dir_is_from":      True,
    }
    apply_weather_to_landscape(land, weather)

    rows, cols = land.shape
    cell_m   = float(cfg.CELL_SIZE_METERS)
    geo_grid = GeoGrid(lat_min, lat_max, lon_min, lon_max, rows, cols)

    # ── Deserialise microclimate and resize to optimizer grid ─────────────────
    # The microclimate may have been built on a different grid size (e.g. 800×800
    # from the interactive sim); we resize with bilinear zoom so it aligns.
    microclimate = None
    mc_runs_used = 0
    if microclimate_json is not None:
        try:
            from core.microclimate import MicroclimateLearner
            _mc_raw      = MicroclimateLearner.from_json(microclimate_json)
            mc_runs_used = _mc_raw._runs
            if mc_runs_used >= 2:
                microclimate = _mc_raw.resize_to(rows, cols)
                print(f"[PSO] Microclimate loaded: {mc_runs_used} prior runs "
                      f"→ applying learned corrections to each particle.")
            else:
                print(f"[PSO] Microclimate has only {mc_runs_used} run(s) — "
                      f"need ≥ 2 for corrections, skipping.")
        except Exception as _mc_exc:
            print(f"[PSO] Microclimate load failed ({_mc_exc}) — running without it.")

    # ── Rasterise truth mask onto the optimisation grid ───────────────────────
    truth_mask = _rasterize_geojson(truth_mask_geo, geo_grid, rows, cols)

    # ── Resolve ignition row/col on the optimisation grid ─────────────────────
    ignition_rcs: list[tuple[int, int]] = []
    pts = msg_init.get("ignition_points", [{"lat": lat_c, "lon": lon_c}])
    for pt in pts:
        rc_raw = geo_grid.latlon_to_rc(float(pt["lat"]), float(pt["lon"]))
        rc     = _snap_to_burnable(rc_raw, land)
        if rc:
            ignition_rcs.append(rc)
    if not ignition_rcs:
        ignition_rcs = [(rows // 2, cols // 2)]

    # ── PSO initialisation ────────────────────────────────────────────────────
    n_dim = len(PARAM_BOUNDS)
    lo    = np.array([b[0] for b in PARAM_BOUNDS], dtype=np.float64)
    hi    = np.array([b[1] for b in PARAM_BOUNDS], dtype=np.float64)

    rng        = np.random.default_rng(seed=42)
    positions  = lo + rng.random((n_particles, n_dim)) * (hi - lo)
    velocities = (rng.random((n_particles, n_dim)) - 0.5) * (hi - lo) * 0.1

    pbest_pos = positions.copy()
    pbest_iou = np.full(n_particles, -np.inf)
    gbest_pos = positions[0].copy()
    gbest_iou = -np.inf

    # Top-K collection (kept sorted ascending by iou so index 0 = worst)
    top_k_heap: list[tuple[float, np.ndarray]] = []

    n_total = n_particles * n_iterations
    n_done  = 0
    t_start = _time.monotonic()

    # ── PSO main loop ─────────────────────────────────────────────────────────
    for iteration in range(n_iterations):
        w = _W_START - (_W_START - _W_END) * iteration / max(n_iterations - 1, 1)

        iter_results: list[tuple[float, np.ndarray] | None] = [None] * n_particles

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(
                    _evaluate_particle,
                    land, cfg, geo_grid, cell_m,
                    ignition_rcs, truth_mask,
                    positions[i].copy(),
                    elapsed_hours,
                    microclimate,      # read-only, thread-safe
                ): i
                for i in range(n_particles)
            }
            for fut in as_completed(future_map):
                i = future_map[fut]
                try:
                    iou, pred = fut.result()
                except Exception:
                    iou, pred = 0.0, np.zeros((rows, cols), dtype=bool)
                iter_results[i] = (iou, pred)
                n_done += 1
                if progress_cb:
                    elapsed = _time.monotonic() - t_start
                    eta     = elapsed / n_done * (n_total - n_done) if n_done > 0 else 0.0
                    progress_cb(iteration + 1, n_done, n_total,
                                max(0.0, gbest_iou), eta)

        # Update personal/global bests and top-K heap
        for i, (iou, pred) in enumerate(iter_results):
            if iou > pbest_iou[i]:
                pbest_iou[i] = iou
                pbest_pos[i] = positions[i].copy()
            if iou > gbest_iou:
                gbest_iou = iou
                gbest_pos = positions[i].copy()

            if len(top_k_heap) < top_k:
                top_k_heap.append((iou, pred.astype(np.float32)))
                top_k_heap.sort(key=lambda x: x[0])
            elif iou > top_k_heap[0][0]:
                top_k_heap[0] = (iou, pred.astype(np.float32))
                top_k_heap.sort(key=lambda x: x[0])

        # Update velocities and positions (standard PSO update rule)
        r1 = rng.random((n_particles, n_dim))
        r2 = rng.random((n_particles, n_dim))
        velocities = (
            w  * velocities
            + _C1 * r1 * (pbest_pos - positions)
            + _C2 * r2 * (gbest_pos - positions)
        )
        positions = np.clip(positions + velocities, lo, hi)

    # ── Build ensemble probability heatmap from top-K simulations ─────────────
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    for _, mask in top_k_heap:
        heatmap += mask
    if top_k_heap:
        heatmap /= len(top_k_heap)

    # ── Build microclimate from top-K optimizer simulations ───────────────────
    # Each burned mask is treated as a fire observation: their average becomes
    # the spatial ignition frequency that the main simulation can use as prior.
    mc_opt_json = None
    try:
        from core.microclimate import MicroclimateLearner
        mc_opt = MicroclimateLearner(rows, cols)
        for _iou_score, _mask in top_k_heap:
            _burned = (_mask > 0.5).astype(np.float32)
            _alpha  = 1.0 / (mc_opt._runs + 2)
            mc_opt.ignition_freq = np.clip(
                (1.0 - _alpha) * mc_opt.ignition_freq + _alpha * _burned,
                0.0, 1.0,
            ).astype(np.float32)
            mc_opt._runs += 1
        if mc_opt._runs > 0:
            mc_opt_json = mc_opt.to_json()
            print(f"[PSO] Optimizer microclimate: {mc_opt._runs} sims → "
                  f"avg_ign_freq={float(mc_opt.ignition_freq.mean()):.3f}")
    except Exception as _mc_e:
        print(f"[PSO] Optimizer microclimate build failed: {_mc_e}")

    best_params = {name: float(gbest_pos[i]) for i, name in enumerate(PARAM_NAMES)}

    return {
        "best_params":                best_params,
        "best_iou":                   float(gbest_iou),
        "heatmap_png_b64":            heatmap_to_png_b64(heatmap),
        "geo_grid":                   geo_grid,
        "rows":                       rows,
        "cols":                       cols,
        "cell_m":                     cell_m,
        "microclimate_used":          microclimate is not None,
        "microclimate_runs":          mc_runs_used,
        "optimizer_microclimate_json":mc_opt_json,
    }

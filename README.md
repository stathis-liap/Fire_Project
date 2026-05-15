# Project WILSON — Wildfire Hindcast & Simulation Engine

> **W**ildfire **I**ntelligent **L**andscape **S**imulation & **O**ptimisation **N**ode

A drone-assisted wildfire digital twin for the Mediterranean. WILSON ingests real satellite fire detections, 30-metre terrain (Copernicus COP30), ERA5 hourly weather, CORINE land cover and FIRED historical perimeters, then runs a physics-based Rothermel Cellular Automata model calibrated against the ground-truth perimeter using Differential Evolution. The result is a data-assimilated hindcast you can replay in 2-D or a full 3-D satellite-textured PyVista terrain mesh.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Quick Start](#3-quick-start)
4. [Data Acquisition Pipeline](#4-data-acquisition-pipeline)
   - 4.1 NASA FIRMS — Active Fire Detections
   - 4.2 Copernicus COP30 DEM — Terrain
   - 4.3 CORINE Land Cover — Vegetation
   - 4.4 ESRI World Imagery — Satellite Texture
   - 4.5 Open-Meteo ERA5 — Hourly Weather
   - 4.6 FIRED — Historical Daily Fire Perimeters
5. [Landscape & Fuel Model](#5-landscape--fuel-model)
   - 5.1 Grid Setup
   - 5.2 Fuel Types (Greek / Mediterranean)
   - 5.3 Equilibrium Moisture Content (EMC)
6. [Wind Field Solver — air/air.py](#6-wind-field-solver--airairpy)
   - 6.1 Layer 1 — Canopy Sheltering (WAF)
   - 6.2 Layer 2 — Thermal Upslope Draft
   - 6.3 Layer 3 — Mass-Consistent Poisson Solver
7. [Fire Spread Model — core/fire_model.py](#7-fire-spread-model--corefire_modelpy)
   - 7.1 Rothermel (1972) Rate of Spread
   - 7.2 Directional Spread (8-neighbour CA)
   - 7.3 Monte Carlo Spotting
8. [Hindcast Optimisation Pipeline](#8-hindcast-optimisation-pipeline)
   - 8.1 Ignition Point Detection
   - 8.2 Two-Scale Resolution Strategy
   - 8.3 Differential Evolution Calibration
   - 8.4 Scoring — IoU + Hausdorff Hybrid
9. [Ground-Truth Sources & Comparison](#9-ground-truth-sources--comparison)
   - 9.1 NASA FIRMS (fallback)
   - 9.2 FIRED Daily Perimeters (preferred)
   - 9.3 Copernicus Shapefile (override)
   - 9.4 Time-Window-Aware Scoring
10. [GUI Tabs Reference](#10-gui-tabs-reference)
11. [3D Earth Viewer](#11-3d-earth-viewer)
12. [Drone Module](#12-drone-module)
13. [Configuration](#13-configuration)
14. [API Keys](#14-api-keys)
15. [Dependencies](#15-dependencies)
16. [Known Presets](#16-known-presets)
17. [References](#17-references)

---

## 1. Architecture Overview

```
ERA5 Weather ──────────┐
COP30 DEM ─────────────┤                          ┌─────────────────┐
CORINE Land Cover ─────┤──► Landscape (200×200) ──►  air/air.py     │
ESRI Satellite JPG ────┘    fuel_map, elevation,   │  WAF + Draft    │
                            moisture, wind          │  + Poisson DCT  │
                                                   └────────┬────────┘
                                                            │ wind_u_grid, wind_v_grid
                                                            ▼
NASA FIRMS ─────────────────────────────────────►  CellularAutomataFire
FIRED Perimeters ────► truth_mask (time-windowed)  Rothermel RoS × 8 dirs
Copernicus SHP ──────┘                             Monte Carlo Spotting
                                                            │
                                                            ▼
                                               Differential Evolution
                                               (moisture_offset, wind_mult)
                                                            │
                                               400×400 final run → IoU + Hausdorff
                                                            │
                                              ┌─────────────────────────┐
                                              │  GUI / 3D PyVista Viewer │
                                              └─────────────────────────┘
```

---

## 2. Project Structure

```
Fire_Project-main/
│
├── gui_launcher.py          ← Main entry point  (python gui_launcher.py)
├── config.py                ← Global constants (grid, dt, Rothermel tuning)
├── api_key.txt              ← OpenTopography API key
│
├── air/                     ← Wind field solver (standalone package)
│   ├── air.py               ← compute_wind_field(): WAF + upslope + Poisson
│   └── __init__.py
│
├── core/                    ← Fire physics engine
│   ├── fire_model.py        ← Rothermel CA + Monte Carlo spotting
│   ├── landscape.py         ← Grid, DEM loader, fuel map, EMC moisture
│   └── fuels.py             ← Greek Rothermel fuel parameters (6 types)
│
├── pipeline/                ← Data acquisition & optimisation
│   ├── hindcast_optimizer.py  ← Full 7-step pipeline: data → DE → IoU
│   ├── auto_fetcher.py        ← COP30, CORINE, satellite texture fetchers
│   ├── fired_loader.py        ← FIRED daily perimeter loader
│   ├── load_dem.py            ← Standalone DEM diagnostic plots
│   └── sanity_plot.py         ← 4-panel run diagnostic figure
│
├── viz/                     ← Visualisation
│   ├── fire_viewer_3d.py    ← PyVista 3-D terrain mesh + fire animation
│   ├── visualizer_3d.py     ← PyVista viewer (Fire Animation tab)
│   └── visualizer.py        ← Lightweight 2-D matplotlib fallback
│
├── drone/                   ← Real-time drone vision integration
│   ├── fire_tracker.py      ← GPS fire footprint extraction from UAV video
│   ├── src/
│   │   ├── main.py          ← Drone pipeline entry point
│   │   ├── vision.py        ← YOLOv8 fire detection on video frames
│   │   └── geometry.py      ← GPS ↔ pixel coordinate transforms
│   └── data/                ← YOLOv8 weights + test footage
│
├── tools/                   ← Dev/testing utilities (not part of main pipeline)
│   ├── simulation.py        ← Synthetic-terrain simulation runner
│   └── optimization.py      ← Synthetic-terrain DE optimizer
│
└── data/
    └── fired_greece_2000_to_2024_daily.gpkg  ← FIRED perimeter dataset
```

---

## 3. Quick Start

### Install dependencies

```bash
conda activate computational_geometry
pip install numpy scipy matplotlib pandas requests pillow pyvista rasterio geopandas
```

### Launch the GUI

```bash
cd Fire_Project-main
python gui_launcher.py
```

Select the **Evoia 2021** preset from the dropdown and click **RUN ALL STEPS**.

### Command-line hindcast

```bash
python pipeline/hindcast_optimizer.py \
  --lat-min 38.5  --lat-max 39.1 \
  --lon-min 22.7  --lon-max 23.7 \
  --date-start 2021-08-01 \
  --date-end   2021-08-13 \
  --hindcast-hours 6 \
  --fired-gpkg data/fired_greece_2000_to_2024_daily.gpkg \
  --maxiter 20 --popsize 12
```

---

## 4. Data Acquisition Pipeline

### 4.1 NASA FIRMS — Active Fire Detections

**Source:** `https://firms.modaps.eosdis.nasa.gov/api/area/csv/`  
**Module:** `pipeline/hindcast_optimizer.py → fetch_firms_data()`

FIRMS (Fire Information for Resource Management System) provides near-real-time and archived satellite fire detections.

| Sensor | Priority | Notes |
|--------|----------|-------|
| VIIRS SNPP (S-NPP) | 1st | 375 m pixel, best resolution |
| VIIRS NOAA-20 | 2nd | Identical spec to SNPP, offset orbit |
| MODIS Terra/Aqua | 3rd | 1 km pixel, older but deeper archive |

**Confidence filtering:**
- VIIRS: keeps `"high"` and `"nominal"` confidence strings
- MODIS: keeps numeric confidence ≥ 30 (out of 100)
- Last-resort fallback: if all rows fail confidence filter, accept all detections

**Day-window chunking:** The NASA FIRMS SP API hard-limits each request to 10 days. For longer windows (e.g. 12-day Evia 2021 fire), `fetch_firms_data()` automatically chunks the request into multiple sub-windows and concatenates the results.

**Truth mask dilation:** Each VIIRS detection is a 375 m × 375 m pixel centroid. On our 200–280 m grid this maps to 1–2 isolated cells. Raw IoU against single-pixel truth would be near-zero even for correct predictions. We dilate each detection by a disc of radius `375 / 2 / cell_m` cells (using `scipy.ndimage.binary_dilation`) to represent the full sensor footprint.

---

### 4.2 Copernicus COP30 DEM — Terrain

**Source:** OpenTopography API → Copernicus GLO-30  
**Module:** `pipeline/auto_fetcher.py → fetch_terrain_from_api()`  
**API:** `https://portal.opentopography.org/API/globaldem?demtype=COP30`

Downloads a GeoTIFF of the 30 m Copernicus Digital Elevation Model for the study bbox. The raw raster (typically 1800×1800 pixels over a 0.5° domain) is bilinearly resampled to the target grid shape (200×200 for DE, 400×400 for the final run) using `rasterio`.

The effective cell size is computed from the DEM bounding box in metres:
```python
lat_span_m = (bbox.top - bbox.bottom) * 111_320          # degrees → metres
lon_span_m = (bbox.right - bbox.left) * 111_320 * cos(lat_centre)
cell_size  = max(lat_span_m / rows, lon_span_m / cols)    # metres per cell
```

The `elevation` array is stored **south-up** (row 0 = southernmost latitude), matching the coordinate system used throughout WILSON.

---

### 4.3 CORINE Land Cover — Vegetation

**Source:** EEA DISCOMAP ArcGIS REST  
**Module:** `pipeline/auto_fetcher.py → fetch_corine_land_cover()`  
**API:** `https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/export`

Downloads the CORINE Land Cover 2018 legend PNG for the terrain bbox. Each pixel's RGB value is matched against the CORINE legend to one of 6 Greek fuel types (see §5.2) using nearest-colour assignment in RGB space (vectorised with NumPy).

CORINE colour → fuel mapping:

| CORINE RGB(s) | Fuel Type |
|---------------|-----------|
| (0,166,0), (77,255,0) | Aleppo_Pine |
| (128,255,0) | Oak_Forest |
| (166,230,77), (166,242,0) | Maquis_Dense_Shrub |
| (230,166,0), (230,230,77) | Olive_Grove |
| (204,242,77), (255,255,168) | Dry_Grass |
| (230,0,77), (166,166,166), (0,0,230), … | Non_Combustible |

---

### 4.4 ESRI World Imagery — Satellite Texture

**Source:** ESRI ArcGIS Online  
**Module:** `pipeline/auto_fetcher.py → fetch_satellite_image_by_bounds()`  
**API:** `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export`

Downloads a high-resolution satellite photograph (JPEG) matching the exact terrain bounding box and aspect ratio. Stored as `texture.jpg`. Used exclusively for the 3D Earth Viewer as per-vertex RGB colours on the terrain mesh — the fire spread model uses CORINE, not this image.

---

### 4.5 Open-Meteo ERA5 — Hourly Weather

**Source:** Open-Meteo ERA5 archive  
**Module:** `pipeline/hindcast_optimizer.py → fetch_weather_hourly()`  
**API:** `https://archive-api.open-meteo.com/v1/archive`

Fetches complete hourly ERA5 reanalysis for the full date range at the ignition point. Variables fetched:

| Variable | ERA5 field |
|----------|-----------|
| Temperature (°C) | `temperature_2m` |
| Relative Humidity (%) | `relative_humidity_2m` |
| Wind speed (m/s) | `wind_speed_10m` |
| Wind direction (°) | `wind_direction_10m` |
| Precipitation (mm) | `precipitation` |
| Surface pressure (hPa) | `surface_pressure` |

The ignition-hour conditions are extracted by matching the ignition timestamp to the hourly table. A full table is printed to the log with a `◄ ignition` marker at the selected row. The scalar wind vector is decomposed into meteorological U/V components:

```
wind_u = -speed × sin(direction_rad)   # East-West (+ = eastward)
wind_v = -speed × cos(direction_rad)   # North-South (+ = northward)
```

These scalars are passed to `air/air.py` which produces the full per-cell wind field.

---

### 4.6 FIRED — Historical Daily Fire Perimeters

**Source:** University of Colorado Boulder Scholar  
**Download:** `https://scholar.colorado.edu/collections/pz50gx05h`  
**File:** `fired_greece_2000_to_2024_daily.gpkg`  
**Module:** `pipeline/fired_loader.py`

FIRED (Fire Events Delineation) provides daily burned-area polygons derived from MODIS MCD64A1 burned-area product. The GeoPackage contains 24,867 daily polygon features for Greece from 2000–2024.

**CRS issue:** The raw file uses the MODIS sinusoidal projection (coordinates in metres, ~2M/5M range). A naive WGS84 bounding-box filter returns zero features. WILSON handles this automatically:

1. Read 1 row to probe the CRS
2. If non-WGS84, read the full file and reproject to EPSG:4326
3. Apply spatial filter using `.cx[lon_min:lon_max, lat_min:lat_max]`
4. Normalise the date column (timezone-naive `datetime64[ms]`) and ID column to `"burn_date"` / `"id"` (integer)

**Confirmed event:** Evia 2021 megafire = event ID **2871**  
- Location: 38.83°N, 23.20°E (northern Evia island)  
- Duration: 2021-08-01 → 2021-08-13 (12 daily polygons)  
- Total area: ~43,047 ha (430 km²)

**Ignition point:** Instead of using the bbox centre (which falls in the Euripus strait — sea), WILSON extracts the centroid of the **first day's polygon** as the ignition point. This guarantees the ignition lands on the actual burn scar on land.

---

## 5. Landscape & Fuel Model

### 5.1 Grid Setup

`core/landscape.py → Landscape`

The landscape is a regular rectangular grid. Two resolutions are used:

| Phase | Grid | Cell size (Evia bbox) |
|-------|------|-----------------------|
| DE optimisation | 200×200 | ~278 m |
| Final IoU run | 400×400 | ~139 m |

Arrays stored per-cell: `elevation` (m), `fuel_map` (int index), `moisture` (fraction), `wind_u` / `wind_v` (m/s, set to full 2D arrays after the air solver runs).

Row index 0 = **south** (smallest latitude). All data — DEM, CORINE, FIRED masks — are stored in this south-up convention. PIL images (top-down) are flipped with `np.flipud` on load.

---

### 5.2 Fuel Types (Greek / Mediterranean)

`core/fuels.py → GREEK_FUELS`

Six fuel types calibrated against Scott & Burgan (2005) standard fuel models and Mediterranean field measurements (Mitsopoulos & Dimitrakopoulos 2007, Fernandes et al. 2000):

| Fuel Type | Closest NFFL model | σ (ft²/ft³) | w₀ (lb/ft²) | Mx | Notes |
|-----------|-------------------|-------------|-------------|-----|-------|
| Aleppo_Pine | TL5 | 1500 | 0.041 | 0.25 | Resinous needle litter, high heat |
| Oak_Forest | TL3 | 1200 | 0.030 | 0.30 | Broadleaf litter, retains moisture |
| Maquis_Dense_Shrub | SH7 | 1800 | 0.092 | 0.25 | Intense heat, high oil content |
| Olive_Grove | TL1/GS1 | 1400 | 0.020 | 0.20 | Sparse managed orchard |
| Phrygana_Low_Scrub | GS2 | 2500 | 0.023 | 0.15 | Fastest spread, dries quickest |
| Dry_Grass | GR2 | 2000 | 0.046 | 0.15 | Scott & Burgan standard |
| Non_Combustible | — | — | 0 | — | Roads, water, urban — blocks spread |

All units are Rothermel (1972) imperial: BTU/lb, ft²/ft³, lb/ft², ft.

---

### 5.3 Equilibrium Moisture Content (EMC)

`core/landscape.py → Landscape.calculate_emc()`

Fine dead-fuel moisture is calculated from ERA5 temperature and relative humidity using the **Nelson EMC model** (standard US fire weather formula):

```
if RH < 10%:   EMC = 0.03229 + 0.281073·RH − 0.000578·RH·T_F
elif RH < 50%: EMC = 2.22749 + 0.160107·RH − 0.01478·T_F
else:          EMC = 21.0606 + 0.005565·RH² − 0.00035·RH·T_F − 0.483199·RH
```

The result (%) is divided by 100 to get a fraction and clamped to [0.01, 0.30].

Spatial variation is added: valley cells (low elevation) hold +2% more moisture than ridge cells, plus ±0.5% random noise for organic spread patterns.

---

## 6. Wind Field Solver — air/air.py

`air/air.py → compute_wind_field(landscape, config)`

Produces a **per-cell (rows×cols) midflame wind field** by applying three sequential physics layers to the ERA5 scalar base wind. Fully vectorised — no Python loops over grid cells. Typical runtime < 20 ms on 200×200.

### 6.1 Layer 1 — Canopy Sheltering (WAF)

**Reference:** Andrews (2012) USDA RMRS-GTR-266  
The **Wind Adjustment Factor** (WAF) reduces the 20-ft open-air wind to the midflame height inside the canopy:

```
U_midflame = WAF × U_20ft
```

WAF values per fuel type (Rothermel/Andrews standard):

| Fuel Type | WAF | Physical meaning |
|-----------|-----|-----------------|
| Aleppo_Pine | 0.10 | Dense closed canopy blocks 90% |
| Oak_Forest | 0.12 | Similar overhead cover |
| Maquis_Dense_Shrub | 0.20 | Tall shrubs, partial sheltering |
| Olive_Grove | 0.28 | Open canopy, moderate |
| Phrygana_Low_Scrub | 0.36 | Low open scrub |
| Dry_Grass | 0.40 | Open terrain, Rothermel default |

Each cell gets its WAF from its fuel-type index, producing a full (rows×cols) WAF matrix applied element-wise:
```python
U = base_u * waf_grid    # (rows, cols)
V = base_v * waf_grid
```

---

### 6.2 Layer 2 — Thermal Upslope Draft

When ambient wind blows uphill, the fire on that slope preheats unburned fuel via a **convective column**, amplifying the effective midflame wind beyond what WAF alone captures.

The draft intensity is proportional to:
1. **Cosine similarity** between the wind vector and the uphill gradient (only positive — downslope wind has no effect)
2. **Slope magnitude** — steeper slopes generate stronger drafts
3. **South-facing aspect bonus** (+15%) for sun-baked Mediterranean terrain in summer

```python
grad_y, grad_x = np.gradient(elevation, dy, dx)
slope_mag = np.hypot(grad_x, grad_y)                       # ≈ tan(slope)
alignment = (base_u * grad_x + base_v * grad_y) / wind_mag  # cosine similarity
draft_intensity = clip(alignment * slope_mag, 0, 2.0)       # upslope only

aspect = arctan2(grad_x, -grad_y)
south_bonus = 1 + 0.15 * clip(cos(aspect - π), 0, 1)

draft_mag = draft_intensity * south_bonus * 0.25            # up to +25% of base wind
U += draft_mag * (base_u / wind_mag)
V += draft_mag * (base_v / wind_mag)
```

---

### 6.3 Layer 3 — Mass-Consistent Poisson Solver

**References:** Sherman (1978), Forthofer (2007), inspired by WindNinja / QUIC-Fire  

The WAF + draft field has artificial divergence (∇·U ≠ 0) from the spatially varying WAF and upslope boosts. A physically realistic wind field must satisfy **mass conservation** (∇·U = 0 for incompressible flow).

**Method:** Solve the 2D Poisson equation for a velocity potential φ, then subtract its gradient to produce a divergence-free field:

```
∇²φ = ∇·(U, V)         [Poisson equation, Neumann BCs]
U_final = U − ∂φ/∂x
V_final = V − ∂φ/∂y
```

The **Helmholtz–Hodge decomposition** guarantees U_final is exactly divergence-free.

**Fast solver — DCT-II diagonalisation:**  
The Neumann Laplacian is diagonalised by the 2D Type-II Discrete Cosine Transform. In transform space the equation becomes trivial pointwise division:

```
φ̂[k,l] = f̂[k,l] / Λ[k,l]

where  Λ[k,l] = (2/dy²)(cos(πk/rows)−1) + (2/dx²)(cos(πl/cols)−1)
```

The DC mode (k=l=0, Λ=0) is in the null space — enforced to zero (zero-mean potential). Complexity: **O(N·M·log(N·M))** — ~1 ms on a 200×200 grid.

**Physical effects produced:**
- Flow **accelerates over ridges** (squeezed cross-section)
- Flow **decelerates in valleys** (expanded cross-section)
- Flow **channels through gaps** (pressure-gradient steering)
- Flow **deflects around blocking terrain**

The resulting `wind_u_grid` / `wind_v_grid` (rows×cols float32) are cached on the `CellularAutomataFire` object and written back to `landscape.wind_u` / `landscape.wind_v` after the final run for use in the 3D viewer.

---

## 7. Fire Spread Model — core/fire_model.py

`core/fire_model.py → CellularAutomataFire`

### 7.1 Rothermel (1972) Rate of Spread

The full Rothermel surface fire spread equation is solved for every cell simultaneously using vectorised NumPy. All fuel properties (σ, w₀, h, ρ_b, Mx) are mapped to (rows×cols) grids at initialisation:

**Step 1 — Packing ratio:**
```
β = ρ_b / ρ_p               (actual / particle density)
β_op = 3.348 × σ^-0.8189    (optimum packing ratio)
```

**Step 2 — Moisture damping:**
```
r_M = min(m_f / M_x, 1)
η_M = 1 − 2.59·r_M + 5.11·r_M² − 3.52·r_M³
```

**Step 3 — Reaction intensity:**
```
Γ_max = σ^1.5 / (495 + 0.0594·σ^1.5)
A = 1 / (4.774·σ^0.1 − 7.27)
Γ = Γ_max · (β/β_op)^A · exp(A·(1 − β/β_op))
I_R = Γ · w_n · h · η_M · η_s
```

**Step 4 — Flux ratio and heat sink:**
```
ξ = exp((0.792 + 0.681·√σ)·(β+0.1)) / (192 + 0.2595·σ)
ε = exp(−138/σ)
Q_ig = 250 + 1116·m_f
```

**Step 5 — Wind and slope modifiers (per direction):**
```
φ_wind = C_w · U_ft_min^B_w · (β/β_op)^{-E_w}
  where C_w, B_w, E_w are fuel-dependent Rothermel coefficients

φ_slope = 5.275 · β^{-0.3} · tan(slope)²
```

`U_ft_min` is the **per-cell midflame wind speed** projected onto the spread direction — a full (rows×cols) matrix drawn from `air/air.py`.

**Step 6 — Final RoS:**
```
R = (I_R · ξ) / (ρ_b · ε · Q_ig) · (1 + φ_wind + φ_slope)   [ft/min]
```

Converted to m/min and stored as a fractional spread probability per time step:
```
p_spread[dir] = min(R × dt / dist, 1.0)   [CFL condition]
```

This `p_spread` (8 × rows × cols) matrix is precomputed **once** at initialisation. No Rothermel arithmetic occurs during the time-stepping loop.

---

### 7.2 Directional Spread (8-neighbour CA)

Each `step()`:

1. **Burn timers** decrement; cells with timer ≤ 0 become `state=2` (burned/ash)
2. **Heat accumulation**: for each of 8 directions, shift the burning mask onto unburned neighbours and add `p_spread[dir]` to their `ignition_fraction`
3. **Deterministic ignition**: cells where `ignition_fraction ≥ 1.0` catch fire — `state=1`, timer reset to `BURN_TIME_STEPS`

Cell states: `0` = unburned, `1` = burning (red), `2` = burned/ash (black)

No random numbers are used in the main spread loop — all stochasticity is isolated in the spotting model.

---

### 7.3 Monte Carlo Spotting

`core/fire_model.py → CellularAutomataFire._apply_spotting()`

Long-range firebrands (Andrews/Albini-style) vectorised over all burning cells simultaneously:

**Lofting:** Each burning cell lofts firebrands with probability
```
P_loft = clip(0.003 × U_midflame, 0, 0.08)   per step
```

**Landing distance:** Log-normal with wind-speed-dependent mean:
```
mean_dist = max(200, 150 × U_midflame)   metres
σ_ln = 0.45
```

**Flight direction:** Local downwind angle ± uniform scatter of ±20°

**Effect:** Firebrands landing on unburned combustible cells deposit 0.5 of an ignition fraction — two overlapping spotfire landings ignite a cell. Enables the model to jump firebreaks, roads and rivers.

---

## 8. Hindcast Optimisation Pipeline

`pipeline/hindcast_optimizer.py → run_hindcast()`

### 8.1 Ignition Point Detection

Priority order:
1. **FIRED mode**: centroid of the first-day polygon for the matched event (most accurate — lands on the actual burn scar)
2. **FIRMS mode**: earliest high/nominal-confidence detection in the date window
3. **Fallback**: bbox centre

For FIRED, the centroid probe uses `load_fired_daily` + `find_fire_event` + `get_event_timeline` — the same functions that handle timezone-naive dates correctly — to avoid the `tz-naive vs tz-aware Timestamp` comparison error.

---

### 8.2 Two-Scale Resolution Strategy

The DE loop runs at **200×200** (~278 m/cell for the Evia 0.5° bbox): each evaluation takes ~0.2 s, so 20 generations × 12 population = 240 evaluations ≈ 50 s.

After the DE converges, the best parameters are applied to a **400×400** grid (~139 m/cell) for the final IoU/Hausdorff score. Both the truth mask and the ignition point are re-rasterised at the fine resolution.

---

### 8.3 Differential Evolution Calibration

`scipy.optimize.differential_evolution` with:

| Parameter | Range | Physical meaning |
|-----------|-------|-----------------|
| `fuel_moisture_offset` | [−0.10, 0.00] | Shift EMC down to account for multi-day drying |
| `wind_multiplier` | [0.50, 2.00] | Scale ERA5 wind (accounts for local channelling not in ERA5) |

At each evaluation:
1. Apply `moisture_offset` to the landscape moisture grid
2. Apply `wind_multiplier` to the scalar base wind before `air/air.py`
3. Run the CA for `hindcast_steps` steps
4. Compute the hybrid error metric (see §8.4)

---

### 8.4 Scoring — IoU + Hausdorff Hybrid

`pipeline/hindcast_optimizer.py → compute_error()`

```
error = 0.30 × (1 − IoU) + 0.70 × H_norm
```

**Term A — IoU loss (30%):**
```
IoU = |Predicted ∩ Truth| / |Predicted ∪ Truth|
```

**Term B — Normalised Symmetric Hausdorff (70%):**  
The symmetric Hausdorff distance captures boundary shape accuracy (IoU misses this — two blobs of equal area but wrong shape can have IoU = 0.5 but Hausdorff >> 0):
```
H_sym = max(directed_hausdorff(P→T), directed_hausdorff(T→P))   [cells]
H_norm = H_sym / max(rows, cols)   ∈ [0, 1]
```

The 70% Hausdorff weight forces the optimiser to match the **shape and location** of the perimeter, not just the total area.

---

## 9. Ground-Truth Sources & Comparison

### 9.1 NASA FIRMS (fallback)

Used when no FIRED gpkg or Copernicus shapefile is provided. Each detection is a point centroid; the truth mask is dilated to a disc of radius ~375/2 m to represent the VIIRS pixel footprint.

### 9.2 FIRED Daily Perimeters (preferred)

`pipeline/fired_loader.py → get_fired_truth_mask()`

FIRED polygons are rasterised directly onto the terrain grid using `rasterio.features.rasterize` with the terrain's affine transform. The result is flipped with `np.flipud` to match the south-up convention.

**Time-window-aware scoring:** The truth mask uses only polygons with `burn_date ≤ ignition_time + hindcast_hours`. This is critical for fair comparison: a 6-hour hindcast should be scored against the 6-hour FIRED footprint, not the 12-day total perimeter. If no polygons fall within the window, the first available day is used as a lower bound.

### 9.3 Copernicus Shapefile (override)

If a `.shp` or `.gpkg` file is provided in the GUI "Copernicus .shp" field, it takes highest priority and is rasterised the same way.

### 9.4 Time-Window-Aware Scoring

The FIRED Timeline tab in the GUI shows:
- Left panel: cumulative burn area (ha) over time as a growth curve
- Right panel: daily perimeter expansion coloured yellow → red

The Comparison tab overlays the FIRED daily perimeter for the matching time window as a **lime green contour** on each of the three snapshot panels.

---

## 10. GUI Tabs Reference

| Tab | Description |
|-----|-------------|
| **Log** | Live pipeline output, step timings, weather table with ◄ ignition marker |
| **Terrain** | DEM elevation + fuel map side by side; truth mask overlay; weather summary |
| **Fire Animation** | 2D flat fire spread animation — play/pause/seek slider, speed control |
| **Hillshade Fire** | Terrain-draped fire on hillshaded DEM; ignition star; "Launch 3D Viewer" button |
| **Comparison** | 3-panel snapshot comparison (early / mid / final); FIRMS/FIRED/Copernicus truth overlaid |
| **FIRED Timeline** | Cumulative burn area chart + daily perimeter growth map |
| **3D Earth View** | Launch button for standalone PyVista 3D viewer |
| **Analysis** | DE convergence curve; TP/FP/FN breakdown; burn-area-over-time chart |

---

## 11. 3D Earth Viewer

`viz/fire_viewer_3d.py` — launched as a **subprocess** so the main GUI never lags.

**Terrain mesh:** `pyvista.StructuredGrid(xx, yy, zz)` at full 200×200 (or 400×400) resolution with 1.5× vertical exaggeration. The satellite JPEG texture is mapped as **per-vertex RGB** (same technique as `Fire_Project-sim-dem/visualizer_3d.py`) — each mesh vertex gets the RGB colour of the corresponding pixel in `texture.jpg`.

**Fire overlay:** A second `StructuredGrid` floats 2 m above the terrain. Per-vertex RGBA colours are updated every frame:
- `state=1` (burning): `[255, 50, 0, 230]` — red-orange flame
- `state=2` (burned): `[30, 20, 10, 200]` — near-black char

**Wind quivers:** `pyvista.PolyData` glyphs using the actual air-corrected `wind_u_grid` / `wind_v_grid` from `air/air.py` (not the raw ERA5 scalar).

**Controls:**

| Key | Action |
|-----|--------|
| `Space` / `P` | Play / pause animation |
| `←` `→` | Step one frame back / forward |
| `Q` / `Esc` | Quit |
| Mouse drag | Rotate terrain |
| Scroll | Zoom |

**Data flow:**  
`gui_launcher._launch_earth3d_pyvista()` → saves `.npz` with `elevation`, `snapshots`, `ignition_rc`, `cell_size_m`, `wind_u`, `wind_v`, `texture_path` → `subprocess.Popen([python, fire_viewer_3d.py, npz_path])`

---

## 12. Drone Module

`drone/` — **not integrated into the main WILSON pipeline yet** (standalone module for future real-time assimilation).

| File | Role |
|------|------|
| `drone/src/vision.py` | YOLOv8n fire detection on video frames |
| `drone/src/geometry.py` | GPS ↔ pixel coordinate transforms (camera intrinsics, drone altitude, NED coordinates) |
| `drone/src/main.py` | Full drone pipeline: video + telemetry → fire GPS footprint |
| `drone/fire_tracker.py` | Extracts fire GPS centroid from drone video + telemetry for injection into the CA |
| `drone/data/` | YOLOv8 weights (`best.pt`, `fire_yolov8n.pt`) and test footage |

The geometry module solves the ray-casting problem: given a pixel location in the camera frame and the drone's GPS + attitude (roll, pitch, yaw, altitude), project the ray to the ground plane and return the GPS coordinates of the fire pixel.

---

## 13. Configuration

`config.py`:

```python
GRID_SIZE         = (200, 200)   # Overwritten at runtime by load_real_terrain()
CELL_SIZE_METERS  = 5.0          # Overwritten at runtime from DEM bbox
dt                = 0.1          # Time step (minutes per CA step)
BURN_TIME_STEPS   = 30           # Steps a cell stays state=1 before becoming ash
TEMPERATURE_C     = 35.0         # Overwritten from ERA5 ignition-hour value
RELATIVE_HUMIDITY = 0.25         # Overwritten from ERA5 (fraction 0–1)
WIND_SPEED        = 5.0          # Overwritten from ERA5 (m/s)
WIND_DIRECTION    = 0.0          # Overwritten from ERA5 (degrees, meteorological)
```

At runtime `apply_weather_to_landscape()` overwrites the config scalars and calls `landscape.set_wind()` to decompose wind speed + direction into U/V components.

---

## 14. API Keys

| Service | Key location | Notes |
|---------|-------------|-------|
| OpenTopography (COP30) | `api_key.txt` | Free academic key — register at opentopography.org |
| NASA FIRMS | GUI "API Key" field or `--map-key` CLI arg | Free — register at firms.modaps.eosdis.nasa.gov/api |

---

## 15. Dependencies

```bash
# Core
numpy scipy matplotlib pandas pillow requests

# Terrain & GIS
rasterio geopandas fiona

# 3D viewer
pyvista vtk

# Optional (drone)
ultralytics opencv-python  # YOLOv8
```

```bash
conda install -c conda-forge geopandas rasterio pyvista
pip install ultralytics  # for drone vision
```

---

## 16. Known Presets

| Preset | Location | Dates | Notes |
|--------|----------|-------|-------|
| **Evoia 2021** | Northern Evia, Greece (38.5–39.1°N, 22.7–23.7°E) | 2021-08-01 → 2021-08-13 | Largest Greek fire in recorded history; ~50,000 ha. FIRED event ID 2871 |
| **Rhodes 2023** | Rhodes island (35.8–36.5°N, 27.0–28.5°E) | 2023-07-19 → 2023-07-22 | Major tourist-area evacuation fire |
| **Alexandroupolis 2023** | Evros, NE Greece (40.6–41.2°N, 25.8–26.3°E) | 2023-08-21 → 2023-08-24 | Deadliest fire in EU history |

---

## 17. References

| Citation | Used for |
|----------|----------|
| Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire Spread in Wildland Fuels. USDA For. Serv. Res. Paper INT-115. | Core RoS equation |
| Andrews, P.L. (2012). Modeling Wind Adjustment Factor and Midflame Wind Speed. USDA RMRS-GTR-266. | WAF values |
| Scott, J.H. & Burgan, R.E. (2005). Standard Fire Behavior Fuel Models. USDA RMRS-GTR-153. | Fuel parameter validation |
| Sherman, C.A. (1978). A Mass-Consistent Model for Wind Fields over Complex Terrain. J. Appl. Meteor. 17, 312–319. | Poisson wind solver |
| Forthofer, J.M. (2007). Modeling Wind in Complex Terrain for Use in Fire Spread Models. MS Thesis, Colorado State University. | DCT-based Poisson solver |
| Mitsopoulos, I. & Dimitrakopoulos, A. (2007). Canopy fuel characteristics of Mediterranean pine forests. IJWF 16, 351–361. | Greek fuel calibration |
| Fernandes, P. et al. (2000). Shrubland fire behaviour in Portugal. For. Ecol. Manag. | Shrub fuel calibration |
| Balch, J.K. et al. (2022). FIRED: A Daily Fire Event Perimeter Dataset (2000–2021). Remote Sensing 14(14), 3498. | FIRED dataset |

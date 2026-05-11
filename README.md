# Hellenic Wildfire Digital Twin × Project WILSON

End-to-end pipeline that takes drone telemetry coordinates and produces a 3-D wildfire simulation on real Mediterranean terrain, using the Rothermel (1972) physics model on a vectorized cellular automaton grid.

## Overview

The system follows a 5-phase pipeline:

```
drone_telemetry.txt  (Lat/Lon)
         │
         ▼
Phase 1: DATA ACQUISITION
  • Download DEM from OpenTopography API
  • Download satellite imagery from ESRI
  • Download land cover data from EEA CORINE
         │
         ▼
Phase 2: LANDSCAPE CONSTRUCTION
  • Load elevation grid from DEM
  • Map CORINE land cover to Greek fuel models
  • Calculate fuel moisture based on weather
         │
         ▼
Phase 3: ROTHERMEL SIMULATION
  • Run vectorized cellular automaton
  • Calculate 8-directional rate of spread
  • Accumulate heat and track burn progression
         │
         ▼
Phase 4: SAVE OUTPUTS
  • Generate diagnostic PNG (4 panels)
  • Save final burn map
  • Export full frame history (.npz)
  • Write summary statistics (.txt)
         │
         ▼
Phase 5: 3-D VISUALIZATION
  • View results in interactive PyVista viewer
```

---

## 📂 Directory Structure

```
DEM_AND_SIM_liap_kall/
│
├── main_dem_sim.py              ← ENTRY POINT: Run this to start simulation
│
├── CONFIG FILES (Edit before running)
├── config.py                    Weather, wind, grid size, simulation parameters
├── api_key.txt                  OpenTopography API key
├── drone_telemetry.txt          Latitude and Longitude of fire ignition point
│
├── CORE SIMULATION MODULES
├── auto_fetcher.py              Downloads DEM, satellite imagery, CORINE data
├── landscape.py                 Loads elevation grid and fuel mapping
├── corine_mapper.py             Maps CORINE codes to Mediterranean fuel types
├── fuels.py                     Greek fuel model properties and fire behavior
├── fire_model.py                Rothermel physics & cellular automaton engine
├── visualizer_3d.py             PyVista 3D interactive viewer
│
├── UTILITIES
├── load_dem.py                  Standalone 2D/3D DEM terrain viewer
├── sanity_plot.py               Generates 4-panel diagnostic figure
│
├── README.md                    Documentation (this file)
│
├── data/                        (Auto-created) Downloaded input data
│   ├── auto_downloaded_terrain.tif
│   ├── corine_cover.png
│   └── texture.jpg
│
├── output/                      (Auto-created) Simulation results
│   ├── sanity_check.png         4-panel diagnostic figure
│   ├── final_burn_map.png       Final burned area extent
│   ├── fire_history.npz         Full simulation history (frame by frame)
│   └── run_summary.txt          Summary statistics
│
└── AUTO-CREATED (first run only)
    ├── fire_history.npz         Frame data from last run
    └── run_summary.txt          Statistics from last run
```

---

## File Directory & Descriptions

### **ENTRY POINT**
- **`main_dem_sim.py`** ← **RUN THIS FILE**
  - Main orchestrator that executes the entire pipeline (5 phases)
  - Reads configuration from `config.py` and text files
  - Calls all sub-modules in sequence
  - Displays 2D animation and offers 3D visualization at end

### **CONFIGURATION FILES (Edit These Before Running)**
- **`config.py`**
  - Grid size (cells × cells) and cell resolution in meters
  - Weather: temperature, humidity, wind speed, wind direction
  - Fire physics: burn time steps, maximum simulation duration
  - **Edit this to customize your simulation**

- **`api_key.txt`** (one line)
  - Your OpenTopography API key for downloading DEM data
  - Get free key at: https://cloud.sdsc.edu/v1/AUTH_opentopography

- **`drone_telemetry.txt`** (two lines)
  - Latitude and Longitude of fire ignition point
  - Format: `Lat: 38.894939` and `Lon: 23.401405`

### **CORE SIMULATION MODULES**
- **`auto_fetcher.py`**
  - Downloads terrain DEM from OpenTopography API
  - Fetches satellite imagery from ESRI World Imagery
  - Retrieves CORINE land cover classification from EEA
  - Parses drone telemetry coordinates

- **`landscape.py`**
  - Loads elevation data from downloaded DEM (.tif)
  - Converts CORINE pixel values to fuel types using Greek fuel models
  - Initializes fuel moisture based on weather conditions
  - Creates the computational grid for simulation

- **`corine_mapper.py`**
  - Maps CORINE land cover codes to Mediterranean fuel types
  - Handles vegetation classification (shrubs, forests, agriculture, etc.)

- **`fuels.py`**
  - Defines Greek fuel model properties (particle density, heat content, moisture, etc.)
  - Contains fire behavior characteristics for Mediterranean vegetation
  - Includes "Non_Combustible" category for roads and water bodies

- **`fire_model.py`**
  - Implements the Rothermel (1972) wildfire physics model
  - Calculates rate of spread (ROS) for each cell
  - Handles 8-directional fire propagation
  - Manages burn states: unburned → burning → ash

- **`visualizer_3d.py`**
  - Creates interactive 3D visualization using PyVista
  - Displays terrain elevation with burn progression overlay
  - Allows rotation, zoom, and exploration of fire spread pattern

### **UTILITY MODULES**
- **`load_dem.py`**
  - Standalone viewer for DEM terrain in 2D and 3D
  - Useful for inspecting downloaded elevation data
  - Can be run separately: `python load_dem.py`

- **`sanity_plot.py`**
  - Generates a 4-panel diagnostic figure:
    1. Elevation map
    2. Fuel types
    3. Moisture content
    4. Final burn extent
  - Automatically called by `main_dem_sim.py`

### **AUTO-CREATED DIRECTORIES**
- **`data/`** (created automatically)
  - `auto_downloaded_terrain.tif` — DEM elevation raster
  - `corine_cover.png` — Land cover classification
  - `texture.jpg` — Satellite imagery (optional)

- **`output/`** (created automatically)
  - `sanity_check.png` — 4-panel diagnostic figure
  - `final_burn_map.png` — Final extent of burned area
  - `fire_history.npz` — Complete simulation frame-by-frame data
  - `run_summary.txt` — Statistics (duration, area burned, etc.)

---

## How to Run

### **Step 1: Install Dependencies**
```bash
pip install numpy scipy matplotlib pyvista vtk rasterio pillow requests
```

### **Step 2: Set Up Configuration Files**

**Create `api_key.txt`** (request free key from OpenTopography):
```
your_opentopography_api_key_here
```

**Create `drone_telemetry.txt`** (location of fire ignition):
```
Lat: 38.894939
Lon: 23.401405
```

### **Step 3: Customize Simulation (Optional)**

Edit `config.py` to adjust:
- `GRID_SIZE` — simulation resolution
- `TEMPERATURE_C`, `RELATIVE_HUMIDITY` — weather conditions
- `WIND_SPEED`, `WIND_DIRECTION` — wind parameters
- `MAX_SIMULATION_STEPS` — how long to run simulation
- `BURN_TIME_STEPS` — how long cells stay burning

### **Step 4: Run the Simulation**
```bash
python main_dem_sim.py
```

The program will:
1. Download terrain data (may take 1-2 minutes on first run)
2. Build the landscape from DEM + CORINE land cover
3. Run the fire simulation
4. Display a 2D animation of fire spread
5. Save outputs to `output/` directory
6. Prompt to view 3D visualization in PyVista

### **Step 5: View Results**

After execution, check the `output/` folder:
- **`sanity_check.png`** — Quick diagnostic overview
- **`final_burn_map.png`** — Final burned area
- **`run_summary.txt`** — Statistics (fire duration, area burned, etc.)
- **`fire_history.npz`** — Full frame history for replay/analysis

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| API key error | Verify `api_key.txt` contains a valid OpenTopography key |
| Telemetry error | Check `drone_telemetry.txt` format: `Lat: X` and `Lon: Y` on separate lines |
| Download timeout | Check internet connection; some data files are large |
| Memory error | Reduce `GRID_SIZE` in `config.py` to lower resolution |
| No outputs | Check that `output/` folder was created and verify file permissions |

---

## Example Locations

Try these coordinates for testing:

- **Mati, Greece** (2018 wildfire): `Lat: 38.017, Lon: 23.866`
- **Attiki Region**: `Lat: 38.894939, Lon: 23.401405` (default)
- **Rhodes, Greece**: `Lat: 36.409, Lon: 28.226`

Edit `config.py` for the fire weather scenario you want to simulate.

## Running

```bash
# Full run: download (if needed) → simulate → save outputs → 3D viewer
python main_dem_sim.py

# Headless (skip 3D viewer; everything else still saved)
python main_dem_sim.py --no-viz

# Longer simulation
python main_dem_sim.py --steps 1500

# Force re-download of DEM/CORINE/texture
python main_dem_sim.py --refetch

# Custom ignition point (row, col — note row 0 = South after flipud)
python main_dem_sim.py --ignite 20 50

# Also show the matplotlib DEM analysis (Delaunay mesh etc.)
python main_dem_sim.py --show-dem

# All flags
python main_dem_sim.py --help
```

## Outputs

After a successful run you get four files in `output/`:

| File                   | What it is                                      |
|------------------------|-------------------------------------------------|
| `sanity_check.png`     | 4-panel diagnostic: elevation, fuel map, fire spread at mid-time and end-time |
| `final_burn_map.png`   | Single clean map of the final burn extent on top of elevation |
| `fire_history.npz`     | NumPy archive with `history` (T, rows, cols), `elevation`, `fuel_map`, `fuel_names`, `ignite_rc` |
| `run_summary.txt`      | Human-readable statistics: weather, grid, ignition, burned area per fuel type |

To replay the saved history later in a script:

```python
import numpy as np
d = np.load("output/fire_history.npz")
history = d["history"]      # (T, rows, cols)  state codes 0/1/2
elev    = d["elevation"]
print(f"Loaded {len(history)} frames, grid {history.shape[1:]}.")
```

## Tweaking the scenario — edit `config.py`

```python
TEMPERATURE_C       = 38.0    # heatwave conditions
RELATIVE_HUMIDITY   = 10.0    # very dry
WIND_SPEED          = 1.0     # m/s
WIND_DIRECTION      = 45      # 0 = North, 90 = East, 180 = South, 270 = West
BURN_TIME_STEPS     = 60      # how long a cell stays burning
MAX_SIMULATION_STEPS = 5000   # ceiling (overridden by --steps)
```

`config.GRID_SIZE` and `config.CELL_SIZE_METERS` are **derived from the DEM**
at runtime — don't edit them, `landscape.load_real_data()` overwrites them.

## Troubleshooting

* **`Missing api_key.txt`** — get a free key from
  <https://portal.opentopography.org/myopentopo>.
* **`status code 401`** from OpenTopography — your key is invalid or
  rate-limited.
* **3D viewer won't open** — usually a missing display/X-server. Use
  `--no-viz`. The other outputs are still written to `output/`.
* **CORINE mostly maps to one fuel type** — the EEA WMS antialiases at
  small extents; this is expected for buffer values ≤ 0.005°. Either
  enlarge the bbox or post-process with an OSM building/road overlay.
* **Fire dies immediately** — your weather is too damp or wind too low.
  Try `TEMPERATURE_C ≥ 30`, `RELATIVE_HUMIDITY ≤ 25`, `WIND_SPEED ≥ 2`.

## What's next

* Apply `texture.jpg` as the surface texture of the 3-D mesh (currently
  rendered with discrete fuel colors).
* Re-run the Differential Evolution data-assimilation in
  `optimization.py` against drone post-fire footage on real DEM.
* Replace `corine_mapper.py`'s nearest-RGB matching with an OSM
  building/road mask to pick up urban firebreaks the EEA WMS smudges.
```

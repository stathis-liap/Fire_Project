# WILSON Rapid — rapid-response fire spread estimator

A small, fast C++ application for fire crews. You drop a pin where the fire is, or type
coordinates. Within a few seconds you get:

* **Where the fire will be**, for exactly as far ahead as you need (any length from 30 min to
  48 h): colour-coded zones, fire-front lines every 30 min (short forecasts) or every hour, and arrows.
* **When it reaches each village, building, road or river**, with a confidence level.
* **What to do about it, with what you actually have.** Enter how many aircraft, fire
  trucks and bulldozers you can get. The planner picks the best plan for exactly those
  resources, each action with coordinates and a deadline. It also tells you what one more
  aircraft, truck or bulldozer would buy.
* **How dangerous every spot is.** A danger map (low → extreme) for the whole area. It combines
  how fast fire would spread from each spot with how quickly it would reach a village or other
  important place. For each village it also shows the *1-hour danger zone*, the ground from which
  a fire would reach the village within an hour.
* **What could happen.** A "chance of fire" heat map from 64–200 scenarios, with a time
  slider. Switch on the wide range when you are unsure of wind or fuel.
* **Updates from the field.** Draw the burned area, report the wind, or let a drone drop
  its detections into a folder. The model adjusts itself to what actually happened.

It is a separate tool from the full WILSON research engine (`server.py`, `sandbox.html`)
and uses the same Greek fuel models (`core/fuels.py`).

---

## Quick start

### 1. Install what is needed (once)

You need a C++17 compiler, CMake ≥ 3.16, OpenSSL development headers and git. Everything else
(HTTP server, JSON, PNG, MapLibre map library) is already included in the repository.

| System | Command |
|---|---|
| Ubuntu / Debian / Pop!_OS | `sudo apt update && sudo apt install -y build-essential cmake libssl-dev git` |
| Fedora | `sudo dnf install -y gcc-c++ make cmake openssl-devel git` |
| Arch | `sudo pacman -S --needed base-devel cmake openssl git` |
| macOS (Homebrew) | `xcode-select --install && brew install cmake openssl@3` (if CMake can't find OpenSSL, add `-DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3)` to the configure command) |

To view the UI you need a recent browser with WebGL: Chrome, Firefox, Edge or Safari, on a
desktop, tablet or phone. You also need internet access the first time you open an area, to
download terrain, vegetation, weather and the satellite base map. After that the area is cached.

### 2. Get the code and build

```bash
git clone https://github.com/stathis-liap/Fire_Project.git
cd Fire_Project
git checkout rapid-response-cpp
cd rapid
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/rapid_tests            # optional: offline tests, should end with "all tests passed"
```

### 3. Run

```bash
./build/wilson_rapid           # from inside rapid/
```

Open **http://localhost:8090** in the browser, tap the map where the fire is (or type
coordinates) and press **Start forecast**. Stop the server with `Ctrl+C`.

```bash
./build/wilson_rapid --host 0.0.0.0             # let tablets/phones on the same Wi-Fi use it:
                                                # open http://<this-computer's-IP>:8090 on them
./build/wilson_rapid --port 9000                # if port 8090 is already in use
./build/wilson_rapid --offline                  # no network: cached areas only
./build/wilson_rapid --forecast 38.56,21.94 --ago 60 --hours 6   # text briefing, no browser
./build/wilson_rapid --forecast 38.56,21.94 --hours 1.5           # any forecast length, 0.5–48 h
```

After pulling new code, rebuild with `cmake --build build -j`. If the build behaves strangely,
start clean with `rm -rf build` and repeat step 2.

### Troubleshooting

* **`Could NOT find OpenSSL`**: install `libssl-dev` (Debian/Ubuntu) or `openssl-devel` (Fedora).
* **`Could not listen on 127.0.0.1:8090`**: another copy is already running, or the port is
  taken. Stop it, or use `--port`.
* **Map stays dark but the panel works**: the browser has WebGL disabled, or there is no
  internet for the base map. The forecast itself still works.
* **"No weather forecast available" / "Land-cover map unavailable"**: no internet on first
  use of that area. The tool falls back to defaults and lowers its confidence. Report the
  local wind to improve it.
* **First forecast for a new area takes 10–40 s**: it is downloading data. Later runs in the
  same area take about a second.

Useful options:

| Option | Meaning |
|---|---|
| `--host 0.0.0.0` | serve tablets and phones on the same network (the default is this computer only) |
| `--port 8090` | HTTP port |
| `--cache DIR` | where map tiles, land cover and weather are cached (default `./cache`) |
| `--dropin DIR` | folder watched for drone/crew files (default `./dropin`) |
| `--offline` | never touch the network: use only cached data. Pre-load an area while you still have a connection. |
| `--members N` | number of ensemble scenarios used for confidence (default 64) |

Headless / scripted use prints a plain-text briefing:

```bash
./build/wilson_rapid --forecast 38.56,21.94 --ago 60 --hours 6
```

## Using it (no training needed)

1. **Tap the map where the fire is**, or type coordinates (`38.56, 21.94`, `38°33'36"N 21°56'24"E`)
   or a place name.
2. Say **when it started** (just now / 30 min / 1 h / …) and **how far ahead** to forecast (1, 2, 3,
   6, 12, 24 h, or type any length from 0.5 to 48 h), then press **Start forecast**. The length can
   be changed later at the top of **Map shows** (*Forecast for the next … h*).
3. Read the panel from top to bottom:
   * **Fire is moving**: the direction, head speed, flame height and what kind of attack is
     safe (hauling-chart limits).
   * **Places at risk**: time to impact for each place, how likely it is, and the confidence.
   * **Your resources**: use − / + to set how many aircraft, fire trucks and bulldozers you
     have. The plan is rebuilt about a second later. *Ask for: …* lines show what one more
     unit would achieve, and *keep in reserve* lines show units the plan doesn't need.
   * **Map shows**: choose **Most likely** (hourly zones and fire-front lines), **Chance of
     fire** (heat map) or **Danger** (how dangerous each spot is, with the 1-hour danger zones
     around villages), and **No action** vs. **With the plan**. Tapping the map gives each spot's
     danger level, spread speed, flame height and how long fire from there would take to
     reach the nearest important place. In heat-map mode, the slider
     picks the time. You can also run 64, 100 or 200 scenarios, and tick *not sure about wind /
     fuel* to widen the range early on, before anything has been calibrated.
     *Show wind on the ground* draws the terrain-adjusted wind (now, or at the slider time in
     heat-map mode).
   * **Recommended plan**: numbered actions, each with *act before HH:MM*, coordinates, its
     effect within the plan and how often it works across scenarios. Switch an action off to
     drop it, or open **Other options** and switch one on to force it in. The rest of the plan
     is re-optimised around your choice.
4. **Tap anywhere** on the map to see when the fire could get there. Use **Protect this place**
   to add it to the list.
5. When you know more, use the bottom toolbar:
   * **Burned area**: tap around what has burned. The model re-fits to it (calibration) and
     forecasts from there.
   * **Local wind**: the wind measured on site overrides the forecast.
   * **Firebreak / Water drop / Truck**: test your own plan.
   * **Update**: re-run for the current time with fresh weather.

Map colours: dark = burned · red = within 1 h · orange = 1–2 h · amber = 2–3 h ·
yellow = later · faint yellow = *possible* (only the pessimistic scenarios reach it).

## How it works

```
 pin / coordinates
        │
        ▼
 ┌──────────────────────── data (parallel, cached on disk) ───────────────────────┐
 │ Open-Meteo forecast   AWS Terrain Tiles (DEM)   CORINE 2018 land cover   OSM    │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         ▼
  landscape: slope, aspect, terrain wind field (mass-consistent + lee + slope winds)
                                         ▼
  spread engine: Rothermel surface + crown fire, elliptical spread, minimum travel time
                                         ▼
   64-scenario Latin-hypercube ensemble (wind / moisture / spread-rate)
                                         ▼
     ┌──────────────┬──────────────────────┬──────────────────────┬──────────────┐
  arrival-time   time to impact +       tactics: breaks, drops,    what-if run of
  zones, lines,  confidence per place   trucks, evacuation         enabled actions
  arrows
```

**Spread engine** (`src/spread.cpp`). Instead of stepping a cellular automaton, the
engine computes the *arrival time* of the fire at every cell with a Dijkstra-style
minimum-travel-time search over a 16-neighbour stencil, the same idea as FlamMap's MTT.
The local spread rate comes from:

* Rothermel (1972) surface spread, with Albini's corrections, using the Greek fuel
  parameters from `core/fuels.py`. Pine and oak stands use their typical shrub understory
  as the surface fuel.
* Wind and slope combined as vectors, then an elliptical fire shape with the
  length-to-breadth ratio from Anderson (1983).
* Crown fire: initiation from Van Wagner (1977), active and passive crown spread from
  Cruz et al. (2005), and crown flame length from Thomas (1963).
* Weather by the hour. Each cell uses the weather of the hour when the front reaches it.
* Wind on the ground, cell by cell (`src/windfield.cpp`), instead of one wind for the whole map:
  * *Mass-consistent flow*: the forecast wind is adjusted so that air in the layer between the
    ground and a fixed top is conserved (∇·(h u) = 0, the 2-D form of the conservation-of-mass
    solvers such as WindNinja's). The flow is steered around high ground, sped up over ridges
    (up to about 2×) and channelled along valleys. The problem is linear in the forecast wind, so
    two conjugate-gradient solves per map (on ≤ 160×160 cells, about 0.1–0.2 s) give the field
    for every hour and every scenario.
  * *Lee shelter*: the upwind horizon angle (Winstral et al. 2002), precomputed for 24 directions,
    slows the wind behind ridges where the flow separates.
  * *Slope winds*: upslope by day on sunlit slopes (up to ~2 m/s, strongest on slopes facing the
    sun), downslope at night (~1.2 m/s), fading as the forecast wind gets stronger.
* South-facing slopes are drier.

A 6-hour forecast on a 450×450 grid takes tens of milliseconds. The whole 64-member
ensemble plus the what-if run typically finishes in well under a second.

**Confidence** (`src/incident.cpp`). Every forecast is run as an ensemble. The overall
score combines:

* data quality: forecast vs. measured wind, land cover present, terrain present, and
  whether the model has been calibrated, plus how well it matched and how well the previous
  forecast did against the newest observation (the out-of-sample check);
* how much the scenarios agree on the burned area.

Each place gets a confidence from its probability of being reached and how spread out the
arrival times are. Each action gets its own paired mini-ensemble: "effective in X % of
scenarios". Grades: **High** ≥ 70, **Medium** 45–70, **Low** < 45.

**Calibration** (`src/calibrate.cpp`), tuned for the next 1–2 hours. When the burned area
(or drone hotspots) is known at time *t*, four parameters are fitted so the simulated fire
matches it: spread-rate multiplier, wind-speed bias, wind-direction bias and fuel-moisture bias.

* *What is scored*: overlap of the **new growth** since the previous fire state (80 %), so
  ground that had already burned cannot inflate the match, plus (20 %) how close to *t* the
  simulated front reaches the observed edge. The timing term is smooth, so the optimiser can
  tell "a little too slow" from "far too slow". Hotspots count as burning only if the front
  reached them shortly before the observation, so a model that runs too fast is penalised.
* *Search*: 96 quasi-random (Halton) starting points over the whole 4-D space, run in
  parallel. The 4 best, from different basins, each start a Nelder–Mead search. That is
  about 400–700 runs and 0.5–0.9 s.
* *Earlier observations* take part with a small weight that halves every 2 h. Benchmarks showed
  that weighting them heavily makes the 1–2 h forecast *worse*: the fire has since moved into
  other fuel.
* *Local correction near the front*: if, in some direction, the real fire grew further or less
  far than the fitted model (wrong vegetation map, local wind), that ratio is carried forward
  as a per-cell spread correction (×0.3–×3, by bearing in 16 sectors). It fades with distance
  from the burned area (e-folding ≈ 1.5 × the observed growth, at least 600 m). It is used only
  for drawn perimeters, and only when it really improves the match.
* *Honest check*: before re-fitting, the forecast that was in use is compared with the new
  observation. "Last check: the forecast made 45 min before … put the fire edge within ~80 m"
  is an out-of-sample error, and it feeds the confidence score.
* *Spread that follows the evidence*: how far each parameter can move before the fit gets
  clearly worse becomes the ensemble spread (see below).

The forecast then continues from the *observed* fire.

Benchmark (40 synthetic fires: hilly terrain, 7 fuel types, drifting wind, hidden biases, and a
"true" vegetation map with 20 % of patches mis-mapped. Observed at +60 and +120 min, forecast
from +120 min):

| | growth overlap +1 h / +2 h | fire-edge error +1 h / +2 h | Brier +1 h / +2 h (lower is better) |
|---|---|---|---|
| not calibrated | 0.53 / 0.53 | 121 / 211 m | 0.146 / 0.141 |
| previous calibration (25-point scan, whole-area IoU, 24 random scenarios) | 0.764 / 0.756 | 61 / 115 m | 0.083 / 0.092 |
| current | 0.771 / 0.770 | 59 / 118 m | 0.053 / 0.062 |

In these tests the single best-guess forecast is now limited by the vegetation-map errors
themselves. Even the *true* parameters only reach 0.72 / 0.75. The large gain is in the
probability map, which is now honest about that uncertainty: the Brier score is about a third
lower, and 98 % of the ground that really burned lies inside the ≥ 10 % zone.

**Danger map** (`SpreadSolver::hazard`, `SpreadSolver::time_to_targets`, `src/incident.cpp`).
It covers the whole area, not just where the fire is now:

* *Fire potential*: head-fire spread rate at every cell for the worst hour of the forecast
  (Rothermel and crown fire, wind, slope).
* *Exposure*: a **reverse** minimum-travel-time search from every village, hamlet, hospital,
  school and marked place, running backwards through the spread ellipses. For each cell it gives
  how long a fire starting there would take to reach the nearest important place, and which one.
* *Danger* = 0.6 × potential + 0.4 × exposure (exposure only counts where fire can spread fast).
  Potential runs from 0 at ≤ 0.05 km/h to 1 at ≥ 5 km/h on a log scale. Exposure is 1 within
  30 min of a place and 0 beyond 3 h. Classes: low / moderate / high / very high / extreme.
* *1-hour danger zone* of a place: all ground from which fire reaches it within 60 min,
  reported as distance, direction and area.

The danger map is cached and only recomputed when the weather, model parameters, places, map
area or forecast hour change. It takes about 0.1–0.5 s.

**Probability heat map** (`src/render.cpp`). For every frame time (every 30 min up to 3 h,
hourly up to 24 h, then every 2 h), each cell shows the share of ensemble scenarios in which
the fire has reached it. The 64 scenarios (default) vary wind speed and direction, fuel moisture
and spread rate on a Latin hypercube, so even a few dozen cover the tails evenly. Each scenario
also gets its own patchy spread-rate field (σ = 0.3, ~600 m patches) for vegetation-map errors
that vary from place to place. Before calibration the variation is set by the data quality.
After calibration it is 1.5 × the parameter range the observations still allow, but never less
than 0.15 (spread rate), 0.25 (wind speed), 20° (wind direction) and 1.5 % (moisture). These
floors were set on the benchmark above, where narrower ensembles scored clearly worse. The
spread fades back to the uncalibrated level over about 6 h. The *wide range* switch multiplies
everything by 1.7. 100 scenarios take about half a second on a 450×450 grid.

**Tactics** (`src/tactics.cpp`) generate a pool of candidate actions. The geometry comes from a
*reasonable worst case*: the scenario at the 70th percentile of burned area, so that places
which may burn get protective options too. The candidates follow that scenario's fire paths:

* *Firebreaks / dead zones* go across the path to a threatened place, at the first point
  the fire reaches late enough for a dozer and crew (≈1.2 km/h) to finish with a 30-minute
  margin. Each break extends across the threatening front and stops at non-burnable ground
  (anchor points). Its width is about 1.5 × flame length. The chance that fire jumps it
  rises with flame length and wind.
* *Aircraft*: a 400 m drop line where the front will be when the aircraft arrives (+30 min).
  The aircraft keeps re-dropping on that line every ~45 min for 3 h. Candidates are placed on
  the paths to threatened places and on the most intense parts of the front.
* *Fire trucks* go between the fire and each threatened place, snapped to the nearest OSM
  road, with a warning when flames exceed the 3.4 m engine limit. When no place is
  threatened, trucks are placed on the flanks.
* *Warn / evacuate* appears when fire may reach a place within 2 h.

Every recommendation is simulated. Firebreaks only work once finished, drops wear off, and
engines are overrun by high flames, so the what-if map shows realistic effects.

**Resource planner** (`src/planner.cpp`). It picks from the candidate pool under the
commander's budget using *lazy greedy* selection. Each round adds the action with the
largest gain. Earlier gains act as upper bounds, so only the most promising candidates are
re-simulated. Every candidate is scored on 6 perturbed scenarios, paired with the same
no-action scenarios:

> value = Σ places weight × protection + 0.02 × hectares saved

* Protection is 1 if the fire is kept out for the whole forecast, otherwise delay ÷ 2 h.
* Weights: city 10, town 6, hospital/school/care home/your marked places 5, village 4,
  hamlet 2, named road 0.5.
* Actions worth less than ~1.5 ha are not given a resource.
* Your own drawn actions and forced options use up budget first.

After planning, it measures what one more unit of each exhausted resource would add.
Planning typically takes 0.2–0.6 s (50–150 simulations).

## Local data: API and drop-in folder

Everything the UI does goes through a JSON API, so other systems (drones, weather stations,
dispatch) can feed it.

| Method & path | Body / query | Purpose |
|---|---|---|
| `POST /api/fire` | `{lat, lon, started_min_ago?, horizon_h?, polygon?}` | start an incident (`horizon_h` 0.5–48, default 6) |
| `GET /api/result` | | latest forecast (zones, lines, places, actions, confidence) |
| `GET /api/status` | | `{status, version}`; poll it to detect updates |
| `GET /api/point?lat=&lon=&plan=0/1` | | time to impact and probability at any point |
| `POST /api/local/weather` | `{wind_kmh \| wind_speed_ms, wind_dir_deg, temp_c?, rh?, time?, source?}` | measured weather |
| `POST /api/local/perimeter` | `{polygon: [[lon,lat],…] \| GeoJSON, time?, source?}` | observed burned area → calibration |
| `POST /api/local/hotspots` | `{points: [[lon,lat],…], time?, source?}` | active fire points (drone / thermal) |
| `POST /api/local/fuel` | `{polygon, fuel: "Dry_Grass" \| "burned" \| …}` | correct the vegetation map |
| `POST /api/destination` | `{lat, lon, name}` or `{remove: id}` | add or remove a place to protect |
| `POST /api/resources` | `{aircraft, trucks, dozers}` | resources the commander can give → re-plan |
| `POST /api/settings` | `{members: 8–200, wide: bool}` | scenarios for confidence and the heat map (default 64) |
| `POST /api/plan` | `{toggle:{key,enabled}}`, `{add:{type, points}}`, `{remove:id}`, `{reset_choices:true}` | force options in or out, add your own actions |
| `POST /api/refresh` | `{horizon_h?, refetch_weather?}` | re-run for the current time, optionally with a new forecast length |
| `POST /api/reset` | | clear the incident |

`time` is a Unix timestamp (seconds or milliseconds). It defaults to now.

**Drop-in folder.** Copy a file into `dropin/`. It is applied within 2 seconds, then moved to
`dropin/processed/`. Unreadable files are prefixed `FAILED_`. Files wait in the folder until
a fire has been started. Accepted formats (see `examples/`):

* JSON with a `type` of `weather`, `perimeter`, `hotspots`, `fuel` or `destination`, using the
  same fields as the API;
* a GeoJSON `Feature` or `FeatureCollection`. `properties.type` selects the kind, and polygons
  default to `perimeter`, points to `hotspots`;
* the **drone module's detection log** (`drone/src/main.py`, `FireEventLogger` CSV). Detections
  with confidence ≥ 0.3 from the last 20 minutes of the log become hotspots.

```bash
curl -X POST localhost:8090/api/local/weather -d @examples/weather.json
cp examples/perimeter.json dropin/
```

## Data sources

| Data | Source | Notes |
|---|---|---|
| Terrain | AWS Terrain Tiles (Terrarium), zoom 11–13 | global, no key |
| Vegetation | EEA CORINE Land Cover 2018 (MapServer export, legend colours verified) | Europe; generic scrub elsewhere |
| Weather | Open-Meteo forecast, hourly, past 24 h + next 48 h | global, no key; cached 30 min |
| Places, roads, rivers | OpenStreetMap via Overpass (three mirrors) | loads in the background; cached 30 days |
| Base map | Esri World Imagery + reference labels | browser only |

The forecast area sizes itself from the weather: about how far a grass fire could run
during the whole period, from 4 to 30 km radius, with 25–130 m cells. It grows automatically
if the fire would leave the map.

## Limitations, read before relying on it

* It is a *rapid estimate*, not a replacement for a fire behaviour analyst. Always check
  the confidence and the reasons listed under **Forecast quality**.
* Spotting (embers starting new fires ahead of the front) is not simulated explicitly. It
  only enters through the firebreak breach probability and the ensemble spread.
* Fuel comes from CORINE (100 m, 2018). Recent burns, clearings and plantations will be
  wrong until you correct them with `/api/local/fuel` or calibrate on the burned area.
* The terrain wind is a fast diagnostic model (mass conservation, lee shelter, slope winds), not
  a CFD solution. It does not simulate fire-induced winds, sea breezes or thunderstorm outflows.
  Report the measured wind when it differs from the forecast.
* Tactical lead times (aircraft 30 min, engines 25 min, dozer 1.2 km/h, aircraft on station
  3 h) are defaults in `TacticsTiming` (`src/tactics.hpp`). The planner's place weights are
  in `place_weight` (`src/planner.cpp`). Adjust both to your service's resources and priorities.
* The planner is greedy: it finds a good plan quickly, not a proven optimum.

## Code map

```
rapid/
  src/geo.hpp          grid, projections, rasterisation of lines/polygons/discs
  src/fuels.*          fuel table (from core/fuels.py), Rothermel, crown fire, CORINE mapping
  src/weather.hpp      hourly series + on-site overrides
  src/fetch.*          HTTP + disk cache: terrain, CORINE, Open-Meteo, Overpass
  src/landscape.*      grid building, slope/aspect
  src/windfield.*      terrain wind: mass-consistent flow, lee shelter, slope winds
  src/spread.*         minimum-travel-time spread engine, interventions
  src/ensemble.*       parallel perturbed runs, arrival quantiles
  src/calibrate.*      fit to observed burned area / hotspots, local correction, parameter spread
  src/tactics.*        candidate firebreaks, drops, trucks, evacuation warnings
  src/planner.*        resource-constrained plan selection (lazy greedy on paired scenarios)
  src/render.*         zone PNG, isochrone contours, GeoJSON
  src/incident.*       incident state, pipeline, JSON for the UI
  src/main.cpp         HTTP server, drop-in watcher, CLI
  web/                 map UI (index.html, app.js, style.css, vendored MapLibre)
  tests/tests.cpp      offline tests (physics, wind field, calibration, tactics, ensemble)
  examples/            sample drop-in files
```

# Project WILSON — 3D Additions

## Quick start

```powershell
cd c:\Fire\Fire_Project
pip install -r requirements.txt
python main.py
```

Then open: **http://localhost:8000/sandbox.html**

`main.py` starts the local app stack:

| Port | Process | Purpose |
|------|---------|---------|
| 8000 | `http.server` | Serves `sandbox.html` |
| 8765 | `server.py` | WebSocket fire simulation |
| 5000 | `mesh_api.py` | 3D terrain mesh API |

---

## Latest updates

- Added `fire_info_panel.py` to centralize fire event generation in the backend.
- Updated `server.py` to emit structured `fire_event` messages for simulation start, weather, interventions, fire progress, slowdown, and completion.
- Updated `sandbox.html` to render `fire_event` messages in the footer panel, limit the visible list to 3 entries, auto-scroll, and support collapse/expand behavior.
- Made the 3D popup and mesh modal smaller for a cleaner viewing experience.

---

## How the 3D popup works

**Ctrl+click** anywhere on the map to:
- Drop a yellow **3D** pin at the geographic coordinate.
- Open the Plotly-based 3D popup with three slide views.
- Keep the pin anchored while panning, zooming, and rotating the map.

Click the yellow pin again to reopen the popup instantly using cached results.

The **`✕ Pin` button** removes the pin without closing the map UI.

---

## 3D popup slides

### Slide 1 — DEM Heatmap
A 2D elevation heatmap of the DEM area around the clicked point.

### Slide 2 — Point Cloud
A 3D scatter of DEM points colored by elevation, with optional wind cones.

### Slide 3 — Terrain Surface
A triangulated mesh draped with a real satellite texture for the same DEM region.

---

## How the fire info panel works

The fire info panel now receives explicit server-generated `fire_event` messages instead of inferring events from every frame.

### Backend event generation
`fire_info_panel.py` contains the `FireInfoPanel` class, which tracks:
- whether ignition has begun
- burned-area milestones
- intervention type and cooldown
- rapid-spread alerts
- slowdown conditions and humidity
- natural extinguishment

It exposes methods for each event type:
- `init_started(rows, cols, cell_m)`
- `weather_update(wind_speed_ms, wind_direction, temperature_c, relative_humidity)`
- `intervention(action, affected_cells)`
- `process_frame(burned_ha, active_ha, minutes_since_ignition, ros)`
- `simulation_completed(count)`

### Server-side wiring
`server.py` now sends `fire_event` payloads in these places:
- after `init_ack` using `init_started`
- after hourly weather updates using `weather_update`
- after firebreak or water drop actions using `intervention`
- after selected frame updates using `process_frame`
- after history is ready using `simulation_completed`

### Client-side rendering
`sandbox.html` handles `msg.type === "fire_event"` and displays each event in the footer panel.
- The panel keeps only the latest 3 events.
- It automatically scrolls to show new messages.
- It can be minimized with the header button and reopened by clicking the header.

---

## Live 3D simulation view

The project also supports a live 3D simulation mode from the **🧊 3D View** button during a running fire simulation.

### What it does
- Computes an AABB bounding box around active ignition points.
- Adds a kilometer margin so the view includes context beyond the fire footprint.
- Extracts a subset of the simulation state and elevation grid.
- Launches `simulation_3d_view.py` as a PyVista viewer subprocess.
- Streams state updates via `_sim3d_live_state.npy`.

### Data flow
1. Browser sends `{ type: "request_3d_view", offset_km: 5.0 }`.
2. `server.py` snapshots current fire state and elevation subset.
3. It writes a payload file and launches `simulation_3d_view.py`.
4. The viewer polls the live state file and refreshes the 3D display.

---

## Files added / modified

| File | Change |
|------|--------|
| `fire_info_panel.py` | New — centralized fire event generation for the simulation server |
| `server.py` | Updated event emission and 3D live state wiring |
| `sandbox.html` | Updated 3D popup sizes, fire event panel, collapse behavior |
| `mesh_api.py` | 3D terrain / satellite texture backend API |
| `main.py` | Launcher for the local HTTP + WebSocket + mesh API stack |
| `requirements.txt` | Install dependencies for Flask, Plotly, rasterio, and mapping tools |

---

## Notes

- The README now starts with the most important commands: change to the project directory, install dependencies, and run the app.
- The 3D popup and fire info panel are separated: popup shows terrain/DEM/mesh, panel shows live simulation events.

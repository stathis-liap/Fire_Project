# Project WILSON — 3D Additions

## Overview

This documents the interactive 3D terrain feature added to the WILSON Sandbox UI. The sandbox already ran the Rothermel CA fire model on a 2D map; these additions layer a real-DEM 3D popup on top of it — triggered by a single Ctrl+click anywhere on the map.

---

## How to run

```powershell
cd c:\Fire\Fire_Project
python main.py
```

`main.py` starts three servers simultaneously and opens the browser:

| Port | Process | Purpose |
|------|---------|---------|
| 8000 | `http.server` | Serves `sandbox.html` |
| 8765 | `server.py` | WebSocket fire simulation |
| 5000 | `mesh_api.py` | 3D terrain mesh API (Flask) |

Then open: **http://localhost:8000/sandbox.html**

---

## The 3D popup

**Ctrl+click** anywhere on the map:
- A yellow **3D** pin is dropped at the clicked coordinate and stays anchored to that geographic point as the map pans/zooms.
- A popup opens with three interactive Plotly slides.

**Clicking the yellow pin** reopens the popup for that location (result is cached — instant reload).

**`✕ Pin` button** in the popup header removes the pin from the map and closes the popup.

---

## Three popup slides

### Slide 1 — DEM Heatmap
Flat 2D heatmap of the real elevation grid. Shows the raw terrain data fetched from the API for the 2 km × 2 km area around the clicked point.

### Slide 2 — Point Cloud
3D scatter of all DEM points coloured by elevation. Vertical exaggeration is applied automatically based on terrain relief. Wind cones (toggle with the **Wind** button) float above the terrain at their correct elevation.

### Slide 3 — Terrain Surface
Full-resolution triangulated mesh (130 × 130 vertices, ~16 900 points) draped with a real satellite image. Wind cones are shown with VEX-adjusted height.

---

## Real terrain pipeline (`mesh_api.py`)

### DEM fetching
`mesh_api.py` calls `pipeline/auto_fetcher.py → fetch_terrain_from_api()` which:
1. Tries Terrarium z14 tiles (Mapzen / Nextzen) — ~9.5 m/px.
2. Falls back to COP30 via OpenTopography if tiles are unavailable.
3. Saves to a `.tif` cache file named `terrain_<lat>_<lon>_<buf>_z14.tif`.

The bounds fetched are **physically square** (2 km × 2 km) using a cos(latitude) correction:

```python
lon_buf = half_m / (111_320 * cos(lat_rad))   # accounts for meridian convergence
lat_buf = half_m / 111_320
```

Without this correction the DEM would be ~4.4 km wide × 3.5 km tall at Greek latitudes.

After fetching, rasterio crops the GeoTIFF exactly to the computed bounds and resamples to a 130 × 130 grid.

### Satellite texture
Each Mesh slide vertex is coloured with a real satellite pixel:

1. ESRI World Imagery tiles are fetched at **zoom 17** (~1.2 m/px) in parallel via `ThreadPoolExecutor`.
2. Tiles are stitched into a single high-res image, then cropped to the exact geographic bbox.
3. Each of the 16 900 mesh vertices is mapped to its geographic coordinate (lat, lon), converted to image (row, col), and the RGB value is read via **bilinear interpolation**.

This produces sharp, geographically-aligned texture at full DEM resolution. Satellite tiles are cached as `sat_<lat>_<lon>_z17.jpg`.

### Triangulation
The mesh uses **explicit grid connectivity** (pre-computed `i, j, k` index arrays for a regular grid) instead of Plotly's `delaunayaxis='z'`. Delaunay re-triangulates in 2D which creates spurious long-distance triangle edges at terrain discontinuities; explicit grid triangles follow the known row/column structure exactly.

---

## Wind overlay

Toggled with the **Wind** button in the popup header. Reads **Wind Speed** and **Wind Direction** from the left panel.

- **Slide 1 (heatmap)**: 2D quiver arrows drawn as line segments + dot tips.
- **Slides 2 & 3 (3D)**: Plotly cone traces placed at the terrain surface elevation for each arrow grid point. The cones on the mesh slide use VEX-adjusted z values so they sit correctly on the exaggerated terrain.

---

## Yellow 3D pin

Implemented as a `position:fixed` `<div>` appended to `document.body` — completely outside the MapLibre DOM so MapLibre cannot touch its position. Geographic-to-screen conversion:

```javascript
var mapRect = map.getContainer().getBoundingClientRect();
var pt      = map.project([lon, lat]);
el.style.left = (mapRect.left + pt.x - 15) + 'px';
el.style.top  = (mapRect.top  + pt.y - 15) + 'px';
```

The pin subscribes to `move`, `zoom`, `rotate`, `pitch`, and `resize` map events to stay locked to its geographic coordinate. `transform` is used only for the hover scale animation — it does not interfere with `left`/`top` positioning.

---

## Files added / modified

| File | Change |
|------|--------|
| `main.py` | New — single-command launcher for all three servers |
| `mesh_api.py` | New — Flask API on port 5000; real DEM fetch, satellite texture, Plotly figure builder |
| `sandbox.html` | Modified — Ctrl+click 3D pin, popup carousel, wind toggle, delete-pin button |
| `air/air.py` | Modified — wind grid data included in mesh API response |
| `core/landscape.py` | Modified — used by mesh_api for fuel map scaffolding |
| `pipeline/hindcast_optimizer.py` | Modified — cos(lat) aware bounds used by DEM fetcher |
| `requirements.txt` | New — Flask, flask-cors, rasterio, numpy, requests |

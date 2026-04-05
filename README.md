# Wildfire Digital Twin

A high-fidelity, 3D wildfire simulation engine built in Python. This project serves as a "Digital Twin" for Mediterranean environments, predicting surface fire spread by combining a Cellular Automata (CA) grid with the industry-standard Rothermel (1972) physics model. 

It was specifically designed to handle the complex topography, diverse fuel types, and dynamic meteorological conditions characteristic of Greece.

## 🌲 Key Features

* **Rothermel Physics Kernel:** Implements the complete Rothermel Surface Fire Spread Model (including Albini's 1976 modifications) to calculate the deterministic Rate of Spread (m/min) based on fuel geometry, wind vectors, and 3D slope.
* **Mediterranean Fuel Libraries:** Pre-configured with empirical constants (surface-area-to-volume ratio, bulk density, heat content) for native Greek vegetation, including Aleppo Pine, Phrygana (Low Scrub), Maquis, and Dry Grass.
* **Dynamic Weather & Moisture:** Calculates Equilibrium Moisture Content (EMC) dynamically based on live ambient temperature and relative humidity inputs, mimicking the explosive fire conditions of summer heatwaves.
* **3D Topographic Visualization:** Built on the powerful VTK/PyVista backend, the visualizer renders the terrain in full 3D, mapping fuel patches to colors and dynamically updating the fire front and burnt ash areas.
* **Decoupled Architecture:** Separates the heavy physics calculation loop from the rendering loop, allowing for "Batch Processing" of simulations and smooth, high-FPS playback of the results.
* **Custom HUD & Wind Vectoring:** Features a custom 2D VTK overlay compass that accurately translates meteorological wind inputs (direction and speed) into Cartesian mathematical vectors applied to the 3D landscape.

## ⚙️ Technical Architecture

The project is structured with strict software engineering principles, centralizing controls and separating concerns:

* `config.py`: The master control panel. Dictates grid size, resolution, simulation time-steps (dt), and live weather variables (Temperature, Humidity, Wind).
* `landscape.py`: Generates the 3D topographical mesh and uses a KDTree Voronoi algorithm to distribute realistic "patches" of heterogeneous vegetation.
* `fire_model.py`: The physics engine. Computes joint probabilities for the Cellular Automata grid, correcting for grid-bias (diagonal vs. orthogonal spread) and enforcing Courant–Friedrichs–Lewy (CFL) condition stability.
* `visualizer_3d.py`: The PyVista rendering engine. Converts the physical state matrices into C-contiguous VTK grid scalars for rapid GPU updates.

## 🧮 The Mathematics

This is not a simple probabilistic game. The spread probability of any given cell is mathematically derived from the continuous Rate of Spread:

1.  **Reaction Intensity:** Calculated using the net fuel load, fuel moisture, and optimal packing ratios.
2.  **Propagating Flux & Damping:** Accounts for moisture damping and mineral damping.
3.  **Vector Modifiers:** Applies non-linear multipliers for the effective wind speed aligned with the spread vector, and the topographical slope.

To ensure organic, elliptical fire spread (and prevent CA grid-bias), the engine soft-caps probabilities and applies distance penalties to diagonal neighbors.

## 🚀 Getting Started

**Prerequisites:**
Ensure you have the required Python libraries installed:
```bash
pip install numpy pyvista scipy matplotlib vtk

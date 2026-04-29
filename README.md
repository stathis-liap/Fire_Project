# Hellenic Wildfire Digital Twin

A high-fidelity, high-performance 3D wildfire simulation engine built in Python. This project serves as a "Digital Twin" for Mediterranean environments, predicting surface fire spread by combining a fully vectorized Cellular Automata (CA) grid with the industry-standard Rothermel (1972) physics model. 

It was specifically designed to handle the complex topography, diverse fuel types, dynamic meteorological conditions characteristic of Greece, and includes a Data Assimilation pipeline to calibrate predictions against real-world drone footage.

## 🌲 Key Features

* **Fully Vectorized Deterministic Engine:** Replaces traditional stochastic "dice-roll" probabilities with a deterministic fractional heat accumulation model. By utilizing pure NumPy matrix operations, the engine calculates the Rothermel physics for the entire grid simultaneously, offering massive HPC (High-Performance Computing) speedups.
* **Data Assimilation (Swarm Optimization):** Features an advanced model-calibration pipeline. It uses a Differential Evolution optimization algorithm to evaluate dozens of parallel "universes," tweaking unmeasured wind microclimates and moisture offsets to perfectly match empirical ground-truth data.
* **Ensemble Forecasting:** Stacks the top-performing optimized particle simulations to generate an industry-standard Confidence Probability Map, preventing the "equifinality" trap inherent in single-run predictions.
* **Mediterranean Fuel Libraries:** Pre-configured with empirical constants (surface-area-to-volume ratio, bulk density, heat content) for native Greek vegetation, including Aleppo Pine, Phrygana (Low Scrub), Maquis, and Dry Grass.
* **Dynamic Weather & Moisture:** Calculates Equilibrium Moisture Content (EMC) dynamically based on live ambient temperature and relative humidity inputs, mimicking the explosive fire conditions of summer heatwaves.
* **3D Topographic Visualization:** Built on the powerful VTK/PyVista backend, the visualizer renders the terrain in full 3D, mapping fuel patches to colors and dynamically updating the fire front and burnt ash areas.

## ⚙️ Technical Architecture

The project is structured with strict software engineering principles, centralizing controls and separating concerns:

* `config.py`: The master control panel. Dictates grid size, resolution, simulation time-steps (dt), and live weather variables (Temperature, Humidity, Wind).
* `landscape.py`: Generates the 3D topographical mesh and uses a KDTree Voronoi algorithm to distribute realistic "patches" of heterogeneous vegetation.
* `fire_model.py`: The HPC physics engine. Computes deterministic fractional heat accumulation across the grid using vectorized matrix shifts, eliminating Python `for`-loop bottlenecks and stochastic grid-bias.
* `optimization.py`: The Data Assimilation module. Wraps the simulation in a SciPy Differential Evolution optimizer to compare outputs against drone ground-truth matrices, returning optimized environmental parameters and ensemble maps.
* `visualizer_3d.py`: The PyVista rendering engine. Converts the physical state matrices into C-contiguous VTK grid scalars for rapid GPU updates.

## 🧮 The Mathematics

This engine operates as a fully deterministic, continuous-time simulation mapped to a discrete grid. The spread of the fire is mathematically derived from the continuous Rate of Spread (R):

1.  **Reaction Intensity:** Calculated using the net fuel load, fuel moisture, and optimal packing ratios.
2.  **Propagating Flux & Damping:** Accounts for moisture damping and mineral damping.
3.  **Vector Modifiers:** Applies non-linear multipliers for the effective wind speed aligned with the spread vector, and the topographical 3D slope.
4.  **Fractional Heat Accumulation:** Instead of rolling probabilities, the algorithm calculates the precise spatial fraction of a target cell crossed per time step. When a cell's accumulated thermal threat reaches 1.0, it deterministically ignites, yielding perfectly smooth, elliptical spread patterns even on a square grid.

## 🚀 Getting Started

**Prerequisites:**
Ensure you have the required Python libraries installed:
```bash
pip install numpy pyvista scipy matplotlib vtk

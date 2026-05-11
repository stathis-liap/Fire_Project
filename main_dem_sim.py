import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from PIL import Image

import config
from auto_fetcher import read_drone_telemetry, fetch_terrain_from_api, fetch_satellite_image_by_bounds, fetch_corine_land_cover
from load_dem import visualize_terrain_technical
from corine_mapper import visualize_corine_vegetation

from landscape import Landscape
from fire_model import CellularAutomataFire
from visualizer_3d import Visualizer3D

def play_2d_history(history):
    """ FIGURE 3: 2D Simulation Player """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Figure 3: 2D Rothermel Spread')
    
    # 0: Unburned (Green), 1: Burning (Red), 2: Ash (Blue/Black mapping)
    cmap = ListedColormap(['#2ca02c', '#d62728', '#1f77b4']) 
    img = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=2)
    ax.set_title("Figure 3: 2D Cellular Automata Simulation", fontsize=14)

    def update(frame):
        img.set_data(history[frame])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=33, blit=True)
    plt.show()


def run_digital_twin():
    print("=====================================================")
    print("|  PROJECT WILSON: DIGITAL TWIN - FULL SIMULATION   |")
    print("=====================================================")

    # --- 1. FILE & SETTINGS CHECK ---
    if not os.path.exists("api_key.txt") or not os.path.exists("drone_telemetry.txt"):
        print("[Error] Missing API Key or Telemetry file!")
        return

    with open("api_key.txt", "r") as f:
        MY_API_KEY = f.read().strip()

    # --- 2. DATA ACQUISITION ---
    print("\n>>> PHASE 1: DATA ACQUISITION <<<")
    lat, lon = read_drone_telemetry("drone_telemetry.txt")
    dem_tif = fetch_terrain_from_api(lat, lon, MY_API_KEY)
    
    with rasterio.open(dem_tif) as src:
        bounds = src.bounds
        w, h = src.width, src.height
    
    texture_jpg = fetch_satellite_image_by_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, w, h)
    
    # Attempt to fetch CORINE with a Fallback (Bypassing Errno 11001 if server is down)
    try:
        corine_png = fetch_corine_land_cover(bounds.left, bounds.bottom, bounds.right, bounds.top, w, h)
    except Exception as e:
        corine_png = None
        
    if not corine_png:
        print("[WARNING] CORINE Server is down. Generating Fallback Vegetation Map to bypass error...")
        # Create a solid green image (Pine Forest) to prevent the simulation from crashing
        img = Image.new('RGB', (w, h), color=(0, 166, 0)) 
        corine_png = "corine_cover.png"
        img.save(corine_png)

    # --- 3. LANDSCAPE CONSTRUCTION ---
    print("\n>>> PHASE 2: BUILDING PHYSICS LANDSCAPE <<<")
    land = Landscape(config)
    land.load_real_terrain(dem_file=dem_tif, corine_file=corine_png)
    land.set_wind(speed=config.WIND_SPEED, direction_degrees=config.WIND_DIRECTION)

    # ==================================
    # PHASE 3: EXTRACTING THE 4 FIGURES
    # ==================================

    print("\n[Presentation] FIGURE 1: 3D Technical Mesh (Close window to proceed)")
    visualize_terrain_technical(dem_tif)

    print("\n[Presentation] FIGURE 2: Vegetation Map & Overlay (Close window to proceed)")
    try:
        # --> FIX: We now pass both the CORINE map and the Satellite Texture
        visualize_corine_vegetation(corine_file=corine_png, texture_file=texture_jpg)
    except Exception as e:
        print(f"[Warning] Failed to show Figure 2: {e}")

    # --- SIMULATION CALCULATION ---
    print("\n>>> PHASE 4: RUNNING ROTHERMEL FIRE EQUATIONS <<<")
    fire_sim = CellularAutomataFire(land, config)
    
    rows, cols = land.shape
    cr, cc = rows // 2, cols // 2
    for i in range(cr - 1, cr + 2):
        for j in range(cc - 1, cc + 2):
            if 0 <= i < rows and 0 <= j < cols: fire_sim.ignite(i, j)

    history = []
    print(f"[Simulation] Computing Matrix... Please wait.")
    for step in range(config.MAX_SIMULATION_STEPS):
        fire_sim.step()
        history.append(fire_sim.state.copy())
        if np.sum(fire_sim.state == 1) == 0 and step > 10: break

    print("\n[Presentation] FIGURE 3: 2D Simulation (Close window to proceed)")
    play_2d_history(history)

    print("\n[Presentation] FIGURE 4: 3D Texture & Wind Quiver (Press 'P' to play)")
    viz = Visualizer3D(fire_sim)
    viz.play_history(history, fps=30)
    
    print("\n[WILSON] Digital Twin Process Completed Successfully.")

if __name__ == "__main__":
    run_digital_twin()
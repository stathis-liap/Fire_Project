# simulation.py
import config
from landscape import Landscape
from fire_model import CellularAutomataFire
from visualizer_3d import Visualizer3D
import numpy as np

if __name__ == "__main__":
    # 1. Setup
    # Pass 'config' instead of config.GRID_SIZE
    land = Landscape(config) 
    land.generate_random_terrain()
    
    # LINK TO CONFIG: Pull wind data directly
    land.set_wind(speed=config.WIND_SPEED, direction_degrees=config.WIND_DIRECTION)
    
    fire_sim = CellularAutomataFire(land, config)
    
    # Dynamically find the center of the grid
    rows, cols = config.GRID_SIZE
    center_r, center_c = rows // 2, cols // 2
    
    # Ignite a 3x3 block exactly in the middle
    for i in range(center_r - 1, center_r + 2):
        for j in range(center_c - 1, center_c + 2):
            fire_sim.ignite(i, j)

    # 2. PRE-CALCULATION PHASE
    history = []
    
    # LINK TO CONFIG: Pull loop limit
    max_steps = config.MAX_SIMULATION_STEPS 
    print(f"Calculating fire spread for {max_steps} steps...")
    
    for step in range(max_steps):
        fire_sim.step()
        
        history.append(fire_sim.state.copy())
        
        active = np.sum(fire_sim.state == 1)
        if step % 50 == 0:
            print(f"Calculating: Step {step} | Active: {active}")
        
        if active == 0 and step > 10:
            print(f"Fire extinguished at step {step}. Calculation complete.")
            break

    # 3. VISUALIZATION PHASE
    viz = Visualizer3D(fire_sim)
    viz.play_history(history, fps=30)
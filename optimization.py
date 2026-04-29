import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import differential_evolution

import config
from landscape import Landscape
from fire_model import CellularAutomataFire

# -------------------------------------------------------------------
# 1. Utility Functions
# -------------------------------------------------------------------
def calculate_iou(simulated_grid, ground_truth_grid):
    """Calculates Intersection over Union (Jaccard Index)."""
    sim_fire = (simulated_grid > 0)
    truth_fire = (ground_truth_grid > 0)
    
    intersection = np.logical_and(sim_fire, truth_fire).sum()
    union = np.logical_or(sim_fire, truth_fire).sum()
    
    return intersection / union if union > 0 else 0.0

def ignite_center(fire_sim, grid_size):
    """Helper to ignite a 3x3 block in the center."""
    rows, cols = grid_size
    cr, cc = rows // 2, cols // 2
    for i in range(cr - 1, cr + 2):
        for j in range(cc - 1, cc + 2):
            fire_sim.ignite(i, j)

# -------------------------------------------------------------------
# 2. The Optimization Wrapper
# -------------------------------------------------------------------
class FireOptimizer:
    def __init__(self, base_landscape, ground_truth, steps):
        self.base_landscape = base_landscape
        self.base_moisture = base_landscape.moisture.copy()
        self.ground_truth = ground_truth
        self.steps = steps
        
        # History to store all swarm evaluations: (IoU_Score, [Parameters])
        self.evaluation_history = []

    def fitness_function(self, params):
        """
        The objective function for the swarm. 
        params: [wind_multiplier, wind_dir_offset, moisture_offset]
        """
        wind_mult, wind_dir_offset, moisture_offset = params
        
        # 1. Apply the particle's guessed parameters to the landscape
        self.base_landscape.set_wind(
            config.WIND_SPEED * wind_mult, 
            (config.WIND_DIRECTION + wind_dir_offset) % 360
        )
        # Apply moisture offset, ensuring it stays physically valid
        self.base_landscape.moisture = np.clip(self.base_moisture + moisture_offset, 0.01, 0.30)
        
        # 2. Run the Vectorized Simulation
        fire_sim = CellularAutomataFire(self.base_landscape, config)
        ignite_center(fire_sim, config.GRID_SIZE)
        
        for _ in range(self.steps):
            fire_sim.step()
            
        # 3. Score the result
        iou = calculate_iou(fire_sim.state, self.ground_truth)
        
        # Save to history for the Ensemble map later
        self.evaluation_history.append((iou, params.copy()))
        
        # Optimizers minimize, so we return Error (1.0 - Accuracy)
        return 1.0 - iou


# -------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Generating Base Environment ---")
    land = Landscape(config)
    land.generate_random_terrain()
    
    # ---------------------------------------------------------------
    # A. MOCK GROUND TRUTH GENERATION
    # ---------------------------------------------------------------
    print("--- 2. Simulating 'Real World' Ground Truth ---")
    # Manually clone the landscape to avoid pickling the 'config' module
    truth_land = Landscape(config)
    truth_land.elevation = land.elevation.copy()
    truth_land.moisture = land.moisture.copy()
    truth_land.fuel_map = land.fuel_map.copy()
    truth_land.fuel_names = list(land.fuel_names)
    
    # Let's say the REAL world had double the wind speed, a 20-degree 
    # shift in direction, and was 3% drier than the weather station reported.
    truth_land.set_wind(config.WIND_SPEED * 2.0, config.WIND_DIRECTION + 20)
    truth_land.moisture = np.clip(truth_land.moisture - 0.03, 0.01, 0.30)
    
    truth_sim = CellularAutomataFire(truth_land, config)
    ignite_center(truth_sim, config.GRID_SIZE)
    
    EVAL_STEPS = 80 # Compare at T=80 steps
    for _ in range(EVAL_STEPS):
        truth_sim.step()
        
    actual_fire_mask = (truth_sim.state > 0)
    
    # ---------------------------------------------------------------
    # B. OPTIMIZATION SWARM
    # ---------------------------------------------------------------
    print("--- 3. Launching Swarm Optimization ---")
    optimizer = FireOptimizer(land, actual_fire_mask, EVAL_STEPS)
    
    # Bounds for the Swarm particles:
    # [Wind Mult (0.5x to 3.0x), Dir Offset (-45 to +45 deg), Moisture Offset (-0.1 to +0.1)]
    bounds = [(0.5, 3.0), (-45.0, 45.0), (-0.1, 0.1)]
    
    # Run the Evolutionary Swarm
    result = differential_evolution(
        optimizer.fitness_function, 
        bounds, 
        maxiter=15,    # Generations
        popsize=10,    # Particles per generation
        mutation=0.7, 
        recombination=0.5,
        disp=True      # Show progress in console
    )
    
    best_params = result.x
    best_iou = 1.0 - result.fun
    print("\n--- 4. Optimization Complete ---")
    print(f"Optimal Wind Multiplier: {best_params[0]:.2f}x")
    print(f"Optimal Wind Dir Offset: {best_params[1]:.1f} degrees")
    print(f"Optimal Moisture Offset: {best_params[2]:.3f}")
    print(f"Best Accuracy (IoU):     {best_iou * 100:.1f}%")

    # ---------------------------------------------------------------
    # C. ENSEMBLE FORECAST (TOP 50 SIMS)
    # ---------------------------------------------------------------
    print("\n--- 5. Generating Ensemble Confidence Map ---")
    # Sort history by IoU (descending) and take top 50
    optimizer.evaluation_history.sort(key=lambda x: x[0], reverse=True)
    top_50 = optimizer.evaluation_history[:50]
    
    ensemble_grid = np.zeros(config.GRID_SIZE, dtype=float)
    
    for iou, params in top_50:
        w_mult, d_off, m_off = params
        
        # Rerun with these parameters
        land.set_wind(config.WIND_SPEED * w_mult, (config.WIND_DIRECTION + d_off) % 360)
        land.moisture = np.clip(optimizer.base_moisture + m_off, 0.01, 0.30)
        
        sim = CellularAutomataFire(land, config)
        ignite_center(sim, config.GRID_SIZE)
        
        for _ in range(EVAL_STEPS):
            sim.step()
            
        # Add a 1.0 to the grid wherever this specific simulation burned
        ensemble_grid += (sim.state > 0).astype(float)
        
    # Divide by 50 to get a probability from 0.0 to 1.0
    confidence_map = ensemble_grid / len(top_50)

    # ---------------------------------------------------------------
    # D. VISUALIZATION
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Un-optimized Guess (What we thought would happen based on config)
    land.set_wind(config.WIND_SPEED, config.WIND_DIRECTION)
    land.moisture = optimizer.base_moisture
    blind_sim = CellularAutomataFire(land, config)
    ignite_center(blind_sim, config.GRID_SIZE)
    for _ in range(EVAL_STEPS): blind_sim.step()
    
    axes[0].imshow(blind_sim.state > 0, cmap='Reds', origin='lower')
    axes[0].set_title("Blind Prediction (Config Defaults)")
    
    # Plot 2: Ground Truth (Drone Data)
    axes[1].imshow(actual_fire_mask, cmap='Oranges', origin='lower')
    axes[1].set_title("Actual Drone Ground Truth")
    
    # Plot 3: Ensemble Confidence Map
    # Display the terrain as a faint background
    axes[2].imshow(land.elevation, cmap='terrain', alpha=0.3, origin='lower')
    # Overlay the probability map (only showing areas > 0% probability)
    im = axes[2].imshow(np.ma.masked_where(confidence_map == 0, confidence_map), 
                        cmap='jet', alpha=0.8, origin='lower', vmin=0.0, vmax=1.0)
    axes[2].set_title("Ensemble Prediction (Top 50 Swarm)")
    
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.7)
    cbar.set_label("Burn Probability")
    
    plt.tight_layout()
    plt.show()
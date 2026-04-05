import numpy as np
from fuels import GREEK_FUELS
from scipy.spatial import KDTree

class Landscape:
    def __init__(self, config):
        self.config = config
        self.shape = self.config.GRID_SIZE
        self.elevation = np.zeros(self.shape)
        self.moisture = np.ones(self.shape) * 0.2  
        
        self.fuel_map = np.zeros(self.shape, dtype=int)
        self.fuel_names = list(GREEK_FUELS.keys())
        
        self.wind_speed = 0
        self.wind_dir = 0
        self.wind_u = 0
        self.wind_v = 0

    def calculate_emc(self, temp_c, rh):
        """
        Estimates the Equilibrium Moisture Content (EMC) of fine dead fuels.
        A simplified version of the Nelson model used in fire behavior.
        Returns moisture as a fraction (e.g., 0.05).
        """
        # Convert Celsius to Fahrenheit for standard forestry formulas
        temp_f = (temp_c * 9/5) + 32
        
        if rh < 10:
            emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp_f
        elif rh < 50:
            emc = 2.22749 + 0.160107 * rh - 0.01478 * temp_f
        else:
            emc = 21.0606 + 0.005565 * rh**2 - 0.00035 * rh * temp_f - 0.483199 * rh

        # EMC is a percentage (e.g., 5.0%). Convert to fraction (0.05) for Rothermel.
        # Clamp it between 1% and 30% for safety.
        return np.clip(emc / 100.0, 0.01, 0.30)

    def generate_random_terrain(self, num_patches=8):
        """
        Generates 3D hills, patchy vegetation, and physics-based moisture mapping.
        """
        rows, cols = self.shape
        x = np.linspace(0, 10, cols)
        y = np.linspace(0, 10, rows)
        X, Y = np.meshgrid(x, y)
        
        # 1. Elevation Logic (Rolling Hills)
        self.elevation = (np.sin(X) * np.cos(Y) * 30 + 
                          np.sin(X/2) * 20 + 50) 

        # 2. Fuel Patch Logic (Voronoi/Nearest Neighbor Seeds)
        num_fuels = len(self.fuel_names)
        seeds_coords = np.random.rand(num_patches, 2) * [rows, cols]
        seeds_fuels = np.random.randint(0, num_fuels, size=num_patches)
        
        tree = KDTree(seeds_coords)
        all_coords = np.argwhere(np.ones(self.shape))
        _, indices = tree.query(all_coords)
        self.fuel_map = seeds_fuels[indices].reshape(self.shape)

        # 3. Dynamic Moisture Logic (Driven by config.py Weather)
        base_moisture = self.calculate_emc(self.config.TEMPERATURE_C, self.config.RELATIVE_HUMIDITY)
        
        # Normalize elevation to 0.0 - 1.0 range
        elev_norm = (self.elevation - np.min(self.elevation)) / (np.max(self.elevation) - np.min(self.elevation))
        
        # Modifier: Valleys (1.0 - elev_norm) hold slightly more moisture (+2%) than peaks
        # Noise: Add a tiny bit of noise (±0.5%) for organic spread patterns
        noise = np.random.uniform(-0.005, 0.005, size=self.shape)
        self.moisture = base_moisture + ((1.0 - elev_norm) * 0.02) + noise
        
        # Final safety clamp to prevent mathematical errors in the Rothermel equation
        self.moisture = np.clip(self.moisture, 0.01, 0.30)

    def set_wind(self, speed, direction_degrees):
        """
        Sets global wind vector.
        direction_degrees: The compass bearing the fire is pushed TOWARDS.
        0 = North (Up), 90 = East (Right), 180 = South (Down), 270 = West (Left).
        """
        self.wind_speed = speed
        self.wind_dir = direction_degrees
        
        # Convert map azimuth to Cartesian mathematical angle
        math_angle_deg = (90 - direction_degrees) % 360
        rad = np.radians(math_angle_deg)
        
        # U is the X-axis (Columns / East-West)
        # V is the Y-axis (Rows / North-South)
        self.wind_u = speed * np.cos(rad)
        self.wind_v = speed * np.sin(rad)

    def get_fuel_at(self, r, c):
        """Returns the dictionary of fuel properties for a specific cell."""
        fuel_idx = self.fuel_map[r, c]
        fuel_name = self.fuel_names[fuel_idx]
        return GREEK_FUELS[fuel_name]
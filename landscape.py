import numpy as np
import rasterio
from PIL import Image
from fuels import GREEK_FUELS

class Landscape:
    def __init__(self, config):
        self.config = config
        
        # Αυτά θα πάρουν τις διαστάσεις τους δυναμικά από το πραγματικό DEM
        self.shape = None
        self.elevation = None
        self.moisture = None
        self.fuel_map = None
        
        # Παίρνουμε τα ονόματα των καυσίμων από τον κώδικα του συμφοιτητή
        self.fuel_names = list(GREEK_FUELS.keys())
        
        # Προσθέτουμε τα "Άκαυστα" υλικά (Δρόμοι, Νερό)
        if "Non_Combustible" not in self.fuel_names:
            self.fuel_names.append("Non_Combustible")
            
        self.wind_speed = 0
        self.wind_dir = 0
        self.wind_u = 0
        self.wind_v = 0

    def calculate_emc(self, temp_c, rh):
        """
        Estimates the Equilibrium Moisture Content (EMC) of fine dead fuels.
        """
        temp_f = (temp_c * 9/5) + 32
        
        if rh < 10:
            emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp_f
        elif rh < 50:
            emc = 2.22749 + 0.160107 * rh - 0.01478 * temp_f
        else:
            emc = 21.0606 + 0.005565 * rh**2 - 0.00035 * rh * temp_f - 0.483199 * rh

        return np.clip(emc / 100.0, 0.01, 0.30)

    def load_real_terrain(self, dem_file="auto_downloaded_terrain.tif", corine_file="corine_cover.png"):
        """ Loads real-world topography and vegetation data. """
        print(f"\n[Landscape] Loading Real World Topography from {dem_file}...")
        
        # 1. Load real elevation data
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            dem_data = np.where(dem_data < 0, 0, dem_data)
            self.elevation = np.flipud(dem_data).astype(float)
        
        # Dynamically set grid size based on loaded map
        self.shape = self.elevation.shape
        self.config.GRID_SIZE = self.shape 

        # 2. Load and Decode CORINE vegetation (Native Vectorized Calculation)
        print(f"[Landscape] Decoding CORINE vegetation from {corine_file}...")
        img = Image.open(corine_file).convert('RGB')
        corine_data = np.array(img)
        corine_data = np.flipud(corine_data)
        
        color_profiles = {
            'Aleppo_Pine': [(0, 166, 0), (77, 255, 0)],
            'Oak_Forest': [(128, 255, 0)],
            'Maquis_Dense_Shrub': [(166, 230, 77), (166, 242, 0)],
            'Olive_Grove': [(230, 166, 0), (230, 230, 77), (255, 230, 166)],
            'Dry_Grass': [(204, 242, 77), (255, 255, 168)],
            'Non_Combustible': [(230, 0, 77), (255, 0, 0), (166, 166, 166), (0, 204, 242), (0, 0, 230), (255, 255, 255)]
        }
        
        pixels = corine_data.reshape(-1, 3).astype(int)
        best_fuel_indices = np.zeros(pixels.shape[0], dtype=int)
        min_dists = np.full(pixels.shape[0], np.inf)
        
        for fuel_name, colors in color_profiles.items():
            if fuel_name in self.fuel_names:
                fuel_idx = self.fuel_names.index(fuel_name)
                for pr, pg, pb in colors:
                    dist = (pixels[:, 0] - pr)**2 + (pixels[:, 1] - pg)**2 + (pixels[:, 2] - pb)**2
                    mask = dist < min_dists
                    min_dists[mask] = dist[mask]
                    best_fuel_indices[mask] = fuel_idx
                
        self.fuel_map = best_fuel_indices.reshape(self.shape)

        # 3. Dynamic moisture based on real elevation
        print("[Landscape] Calculating physics-based moisture on real terrain...")
        base_moisture = self.calculate_emc(self.config.TEMPERATURE_C, self.config.RELATIVE_HUMIDITY)
        
        elev_norm = (self.elevation - np.min(self.elevation)) / (np.max(self.elevation) - np.min(self.elevation))
        noise = np.random.uniform(-0.005, 0.005, size=self.shape)
        self.moisture = base_moisture + ((1.0 - elev_norm) * 0.02) + noise
        self.moisture = np.clip(self.moisture, 0.01, 0.30)
        
        print(f"[Landscape] Matrix initialized at {self.shape[0]}x{self.shape[1]} cells.")

    def set_wind(self, speed, direction_degrees):
        """ Sets global wind vector. """
        self.wind_speed = speed
        self.wind_dir = direction_degrees
        
        math_angle_deg = (90 - direction_degrees) % 360
        rad = np.radians(math_angle_deg)
        
        self.wind_u = speed * np.cos(rad)
        self.wind_v = speed * np.sin(rad)

    def get_fuel_at(self, r, c):
        """ Returns fuel properties for a specific cell. """
        fuel_idx = self.fuel_map[r, c]
        fuel_name = self.fuel_names[fuel_idx]
        
        if fuel_name == "Non_Combustible":
            return {} 
            
        return GREEK_FUELS[fuel_name]
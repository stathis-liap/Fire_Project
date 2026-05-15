import numpy as np
from core.fuels import GREEK_FUELS
from scipy.spatial import KDTree

try:
    import rasterio
    from PIL import Image
    _REAL_TERRAIN_AVAILABLE = True
except ImportError:
    _REAL_TERRAIN_AVAILABLE = False

class Landscape:
    def __init__(self, config):
        self.config = config
        self.shape = self.config.GRID_SIZE
        self.elevation = np.zeros(self.shape)
        self.moisture = np.ones(self.shape) * 0.2  
        
        self.fuel_map = np.zeros(self.shape, dtype=int)
        self.fuel_names = list(GREEK_FUELS.keys())
        # Non-combustible class (roads, water, urban) used by real CORINE data
        if "Non_Combustible" not in self.fuel_names:
            self.fuel_names.append("Non_Combustible")
        
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

    # ------------------------------------------------------------------
    # Real-World Terrain Loader  (from sim-dem / Project WILSON)
    # ------------------------------------------------------------------

    def load_real_terrain(self, dem_file="auto_downloaded_terrain.tif",
                          corine_file="corine_cover.png",
                          target_shape=None):
        """
        Load real-world topography and vegetation data into the landscape.

        Parameters
        ----------
        dem_file     : path to the downloaded COP30 GeoTIFF.
        corine_file  : path to the downloaded CORINE land-cover PNG.
        target_shape : (rows, cols) to resample the DEM/CORINE to, or None to
                       use config.GRID_SIZE. Raw COP30 can be 5000+ pixels wide
                       over a 1.5° bbox — resampling is essential for the CA
                       optimizer to run in minutes rather than hours.
        """
        if not _REAL_TERRAIN_AVAILABLE:
            raise ImportError(
                "rasterio and Pillow are required for load_real_terrain(). "
                "Install with:  pip install rasterio Pillow"
            )

        if target_shape is None:
            target_shape = tuple(self.config.GRID_SIZE)

        tgt_rows, tgt_cols = target_shape

        # 1. Elevation from COP30 DEM — resample to target_shape
        print(f"\n[Landscape] Loading real DEM from '{dem_file}' ...")
        from rasterio.enums import Resampling as _Resampling
        with rasterio.open(dem_file) as src:
            raw_rows, raw_cols = src.height, src.width
            print(f"[Landscape] Raw DEM: {raw_rows}x{raw_cols} -> resampling to {tgt_rows}x{tgt_cols}")
            dem_data = src.read(
                1,
                out_shape=(tgt_rows, tgt_cols),
                resampling=_Resampling.bilinear,
            ).astype(float)
            dem_data = np.where(dem_data < 0, 0.0, dem_data)
            self.elevation = np.flipud(dem_data)
            self._dem_bounds = src.bounds

        self.shape = self.elevation.shape
        self.config.GRID_SIZE = self.shape

        # Update CELL_SIZE_METERS to reflect the resampled resolution
        # COP30 is ~30m native; at target_shape the effective cell size scales.
        lat_span_m = (self._dem_bounds.top - self._dem_bounds.bottom) * 111_320
        lon_span_m = (self._dem_bounds.right - self._dem_bounds.left) * 111_320 * \
                     np.cos(np.radians((self._dem_bounds.top + self._dem_bounds.bottom) / 2))
        self.config.CELL_SIZE_METERS = float(max(lat_span_m / tgt_rows,
                                                  lon_span_m / tgt_cols))
        print(f"[Landscape] Effective cell size: {self.config.CELL_SIZE_METERS:.1f} m")

        # 2. CORINE fuel map — resize PNG directly to target_shape
        print(f"[Landscape] Decoding CORINE vegetation from '{corine_file}' ...")
        img = Image.open(corine_file).convert("RGB")
        img_resized = img.resize((tgt_cols, tgt_rows), Image.NEAREST)
        corine_data = np.flipud(np.array(img_resized))

        # Colour profiles: CORINE legend RGB → Greek fuel name
        color_profiles = {
            "Aleppo_Pine":        [(0, 166, 0), (77, 255, 0)],
            "Oak_Forest":         [(128, 255, 0)],
            "Maquis_Dense_Shrub": [(166, 230, 77), (166, 242, 0)],
            "Olive_Grove":        [(230, 166, 0), (230, 230, 77), (255, 230, 166)],
            "Dry_Grass":          [(204, 242, 77), (255, 255, 168)],
            "Non_Combustible":    [(230, 0, 77), (255, 0, 0), (166, 166, 166),
                                   (0, 204, 242), (0, 0, 230), (255, 255, 255)],
        }

        pixels     = corine_data.reshape(-1, 3).astype(int)
        best_idx   = np.zeros(pixels.shape[0], dtype=int)
        min_dists  = np.full(pixels.shape[0], np.inf)

        for fuel_name, colors in color_profiles.items():
            if fuel_name not in self.fuel_names:
                continue
            fuel_idx = self.fuel_names.index(fuel_name)
            for pr, pg, pb in colors:
                dist = ((pixels[:, 0] - pr) ** 2 +
                        (pixels[:, 1] - pg) ** 2 +
                        (pixels[:, 2] - pb) ** 2)
                mask = dist < min_dists
                min_dists[mask] = dist[mask]
                best_idx[mask]  = fuel_idx

        self.fuel_map = best_idx.reshape(self.shape)

        # 3. Physics-based moisture from real elevation + config T/RH
        print("[Landscape] Calculating EMC-based moisture on real terrain ...")
        base_emc  = self.calculate_emc(self.config.TEMPERATURE_C,
                                       self.config.RELATIVE_HUMIDITY)
        elev_range = np.max(self.elevation) - np.min(self.elevation)
        if elev_range > 0:
            elev_norm = (self.elevation - np.min(self.elevation)) / elev_range
        else:
            elev_norm = np.zeros_like(self.elevation)
        noise         = np.random.uniform(-0.005, 0.005, size=self.shape)
        self.moisture = np.clip(
            base_emc + (1.0 - elev_norm) * 0.02 + noise, 0.01, 0.30
        )

        print(f"[Landscape] Real terrain loaded: {self.shape[0]}×{self.shape[1]} cells  "
              f"({self.elevation.min():.0f}–{self.elevation.max():.0f} m)")

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
        """Returns the dictionary of fuel properties for a specific cell.
        Returns empty dict for Non_Combustible cells (roads, water, urban)."""
        fuel_idx  = self.fuel_map[r, c]
        fuel_name = self.fuel_names[fuel_idx]
        if fuel_name == "Non_Combustible":
            return {}
        return GREEK_FUELS[fuel_name]
import numpy as np
from core.fuels import GREEK_FUELS
from scipy.spatial import KDTree

try:
    import rasterio
    from PIL import Image
    _REAL_TERRAIN_AVAILABLE = True
except ImportError:
    _REAL_TERRAIN_AVAILABLE = False

# ── CLC 2018 integer class code → Greek fuel name mapping ─────────────────────
# Supports both 3-digit CLC codes (111–523) as used in the EEA standard legend
# and sequential codes (1–44) that some ESRI services return.
# Reference: EEA (2018) CORINE Land Cover — Nomenclature guidelines.
#            https://land.copernicus.eu/pan-european/corine-land-cover
CLC_TO_FUEL = {
    # ── Artificial surfaces ──────────────────────────────────────────────────
    111: "Urban_Fabric",          # Continuous urban fabric
    112: "Urban_Fabric",          # Discontinuous urban fabric
    121: "Non_Combustible",       # Industrial or commercial units
    122: "Urban_Road",            # Road and rail networks
    123: "Non_Combustible",       # Port areas
    124: "Non_Combustible",       # Airports
    131: "Non_Combustible",       # Mineral extraction sites
    132: "Non_Combustible",       # Dump sites
    133: "Abandoned_Agricultural",# Construction sites (bare ground in transition)
    141: "Dry_Grass",             # Green urban areas (parks, golf courses)
    142: "Dry_Grass",             # Sport and leisure facilities
    # ── Agricultural areas ───────────────────────────────────────────────────
    211: "Annual_Crops",          # Non-irrigated arable land
    212: "Annual_Crops",          # Permanently irrigated land
    213: "Riparian_Vegetation",   # Rice fields (wet/flooded)
    221: "Vineyard",
    222: "Olive_Grove",           # Fruit trees and berry plantations
    223: "Olive_Grove",           # Olive groves
    231: "Dry_Grass",             # Pastures
    241: "Annual_Crops",          # Annual crops assoc. with permanent crops
    242: "Abandoned_Agricultural",# Complex cultivation patterns
    243: "Abandoned_Agricultural",# Land principally used for agriculture
    244: "Oak_Forest",            # Agro-forestry areas (open woodland)
    # ── Forests ──────────────────────────────────────────────────────────────
    311: "Oak_Forest",            # Broad-leaved forest
    312: "Aleppo_Pine",           # Coniferous forest
    313: "Aleppo_Pine",           # Mixed forest
    # ── Scrub and/or herbaceous vegetation ───────────────────────────────────
    321: "Dry_Grass",             # Natural grasslands
    322: "Phrygana_Low_Scrub",    # Moors and heathland
    323: "Maquis_Dense_Shrub",    # Sclerophyllous vegetation
    324: "Garrigue",              # Transitional woodland-shrub
    331: "Non_Combustible",       # Beaches, dunes, sands
    332: "Non_Combustible",       # Bare rocks
    333: "Phrygana_Low_Scrub",    # Sparsely vegetated areas
    334: "Non_Combustible",       # Burnt areas (already scorched)
    335: "Non_Combustible",       # Glaciers and perpetual snow
    # ── Wetlands ─────────────────────────────────────────────────────────────
    411: "Riparian_Vegetation",   # Inland marshes
    412: "Riparian_Vegetation",   # Peat bogs
    421: "Water",                 # Salt marshes
    422: "Water",                 # Salines
    423: "Water",                 # Intertidal flats
    # ── Water bodies ─────────────────────────────────────────────────────────
    511: "Water",                 # Water courses
    512: "Water",                 # Water bodies
    521: "Water",                 # Coastal lagoons
    522: "Water",                 # Estuaries
    523: "Water",                 # Sea and ocean
}

# Sequential codes 1–44 (some ESRI services return these instead of 3-digit codes)
_CLC_SEQ_TO_3DIGIT = {
    1: 111, 2: 112, 3: 121, 4: 122, 5: 123, 6: 124,
    7: 131, 8: 132, 9: 133, 10: 141, 11: 142,
    12: 211, 13: 212, 14: 213, 15: 221, 16: 222, 17: 223, 18: 231,
    19: 241, 20: 242, 21: 243, 22: 244,
    23: 311, 24: 312, 25: 313,
    26: 321, 27: 322, 28: 323, 29: 324,
    30: 331, 31: 332, 32: 333, 33: 334, 34: 335,
    35: 411, 36: 412,
    37: 421, 38: 422, 39: 423,
    40: 511, 41: 512,
    42: 521, 43: 522, 44: 523,
}


class Landscape:
    def __init__(self, config):
        self.config = config
        self.shape = self.config.GRID_SIZE
        self.elevation = np.zeros(self.shape)
        self.moisture = np.ones(self.shape) * 0.2  
        
        self.fuel_map = np.zeros(self.shape, dtype=int)
        self.fuel_names = list(GREEK_FUELS.keys())
        # Non-combustible class (roads, urban, bare rock) used by CORINE data.
        # "Water" is already in GREEK_FUELS with fuel_load=0; "Non_Combustible"
        # is appended here for CORINE pixels that map to roads/urban/rock.
        if "Non_Combustible" not in self.fuel_names:
            self.fuel_names.append("Non_Combustible")
        
        self.wind_speed = 0
        self.wind_dir = 0
        self.wind_u = 0
        self.wind_v = 0
        self.osm_overlay_stats = None

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
        # Only use combustible fuel types so the synthetic scenario always burns.
        _non_burn = {"Non_Combustible", "Water"}
        comb_indices = [i for i, n in enumerate(self.fuel_names) if n not in _non_burn]
        if not comb_indices:
            comb_indices = list(range(len(self.fuel_names)))
        seeds_coords = np.random.rand(num_patches, 2) * [rows, cols]
        seeds_fuels = np.array([comb_indices[i % len(comb_indices)]
                                for i in np.random.randint(0, len(comb_indices), size=num_patches)])
        
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
                          target_shape=None,
                          osm_file=None):
        """
        Load real-world topography, land-cover and (optionally) OSM features.

        Parameters
        ----------
        dem_file     : path to COP30 GeoTIFF (from fetch_terrain_from_api).
        corine_file  : path to CLC 2018 data file.  Two formats accepted:
                         • .tif / .tiff — single-band GeoTIFF with integer CLC
                           class codes (111–523 or sequential 1–44).  Direct
                           lookup via CLC_TO_FUEL; no colour matching needed.
                         • .png         — rendered colour export from EEA WMS.
                           Falls back to nearest-RGB colour matching.
        target_shape : (rows, cols) to resample the DEM/CORINE to, or None to
                       use config.GRID_SIZE.
        osm_file     : path to GeoJSON from fetch_osm_features(), or None.
                       When provided, roads and urban areas are rasterised on top
                       of the CLC fuel map as Urban_Road / Urban_Fabric cells.
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

        # 2. Fuel map — CLC integer GeoTIFF (primary) or rendered PNG (fallback)
        print(f"[Landscape] Decoding land cover from '{corine_file}' ...")
        from rasterio.enums import Resampling as _Resampling

        corine_ext = str(corine_file).lower()
        non_comb_idx = self.fuel_names.index("Non_Combustible")

        if corine_ext.endswith((".tif", ".tiff")):
            # ── GeoTIFF path: direct integer CLC code lookup ──────────────────
            # rasterio reads GeoTIFFs with row-0 = north; flipud → row-0 = south
            # to match our DEM orientation.
            print("[Landscape]   Format: GeoTIFF (integer CLC codes) — direct lookup")
            with rasterio.open(corine_file) as clc_src:
                clc_raw = clc_src.read(
                    1,
                    out_shape=(tgt_rows, tgt_cols),
                    resampling=_Resampling.nearest,
                ).astype(np.int32)

            # Auto-detect sequential (1–44) vs 3-digit (111–523) codes
            unique_vals = np.unique(clc_raw[clc_raw > 0])
            if len(unique_vals) > 0 and int(unique_vals.max()) <= 44:
                print(f"[Landscape]   Sequential CLC codes detected (max={unique_vals.max()}) "
                      "— converting to 3-digit")
                vec_convert = np.vectorize(
                    lambda v: _CLC_SEQ_TO_3DIGIT.get(int(v), 0), otypes=[np.int32]
                )
                clc_codes = vec_convert(clc_raw)
            else:
                clc_codes = clc_raw

            fuel_idx_grid = np.full(clc_codes.shape, non_comb_idx, dtype=int)
            for code, fuel_name in CLC_TO_FUEL.items():
                if fuel_name in self.fuel_names:
                    fuel_idx_grid[clc_codes == code] = self.fuel_names.index(fuel_name)

            self.fuel_map = np.flipud(fuel_idx_grid)

            # Log class coverage
            unique_fuels, counts = np.unique(self.fuel_map, return_counts=True)
            total = self.fuel_map.size
            coverage = {self.fuel_names[i]: int(c) for i, c in zip(unique_fuels, counts)}
            top = sorted(coverage.items(), key=lambda x: -x[1])[:6]
            print("[Landscape]   Top fuel classes: " +
                  ", ".join(f"{n} {c*100/total:.1f}%" for n, c in top))

        else:
            # ── PNG path: nearest-RGB colour matching (fallback) ──────────────
            print("[Landscape]   Format: PNG (colour-matched) — nearest-RGB lookup")
            img = Image.open(corine_file).convert("RGB")
            img_resized = img.resize((tgt_cols, tgt_rows), Image.NEAREST)
            corine_data = np.flipud(np.array(img_resized))

            # CLC 2018 legend colours (8-bit RGB).  Each fuel mapped to its
            # closest official EEA palette entry.
            color_profiles = {
                "Aleppo_Pine":              [(0, 166, 0), (77, 255, 0), (204, 242, 77)],
                "Black_Pine":               [(0, 128, 0)],
                "Greek_Fir":                [(0, 100, 0)],
                "Maritime_Pine":            [(0, 153, 51)],
                "Oak_Forest":               [(128, 255, 0)],
                "Chestnut_Forest":          [(96, 192, 0)],
                "Beech_Forest":             [(64, 160, 0)],
                "Maquis_Dense_Shrub":       [(166, 230, 77)],
                "Tall_Maquis":              [(140, 210, 40)],
                "Phrygana_Low_Scrub":       [(166, 242, 0)],
                "Garrigue":                 [(204, 230, 77)],
                "Dry_Grass":                [(230, 230, 0), (255, 255, 168)],
                "Annual_Crops":             [(255, 255, 0), (255, 230, 77)],
                "Abandoned_Agricultural":   [(166, 242, 128)],
                "Olive_Grove":              [(230, 166, 0), (255, 230, 166)],
                "Vineyard":                 [(230, 230, 77)],
                "Eucalyptus":               [(0, 180, 60)],
                "Riparian_Vegetation":      [(0, 200, 100)],
                "Cypress":                  [(0, 120, 20)],
                "Stone_Pine":               [(60, 190, 0)],
                "Urban_Fabric":             [(230, 0, 77), (255, 0, 0), (255, 77, 77)],
                "Urban_Road":               [(204, 204, 204), (153, 153, 153)],
                "Water":                    [(0, 0, 230), (0, 204, 242), (166, 166, 230)],
                "Non_Combustible":          [(166, 166, 166), (255, 255, 255)],
            }

            pixels    = corine_data.reshape(-1, 3).astype(int)
            best_idx  = np.zeros(pixels.shape[0], dtype=int)
            min_dists = np.full(pixels.shape[0], np.inf)

            for fuel_name, colors in color_profiles.items():
                if fuel_name not in self.fuel_names:
                    continue
                fidx = self.fuel_names.index(fuel_name)
                for pr, pg, pb in colors:
                    dist = ((pixels[:, 0] - pr) ** 2 +
                            (pixels[:, 1] - pg) ** 2 +
                            (pixels[:, 2] - pb) ** 2)
                    mask = dist < min_dists
                    min_dists[mask] = dist[mask]
                    best_idx[mask]  = fidx

            self.fuel_map = best_idx.reshape(self.shape)

        # 3. OSM overlay: rasterise roads and urban areas on top of the CLC map
        if osm_file:
            bounds = (self._dem_bounds.bottom, self._dem_bounds.top,
                      self._dem_bounds.left,  self._dem_bounds.right)
            self.apply_osm_overlay(osm_file, bounds)

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

    def apply_osm_overlay(self, osm_geojson_path, terrain_bounds):
        """
        Rasterise OpenStreetMap roads and urban areas onto the existing fuel_map.

        Roads  → Urban_Road  (buffered by ½ cell width so narrow ways are visible)
        Urban  → Urban_Fabric

        OSM overlays REPLACE the underlying CLC class for matched pixels.  This
        is intentional: OSM has higher resolution than CLC 100 m and its road
        network is the authoritative source for firebreak geometry.

        Parameters
        ----------
        osm_geojson_path : path to GeoJSON FeatureCollection from fetch_osm_features().
        terrain_bounds   : (lat_min, lat_max, lon_min, lon_max) of the loaded terrain.
        """
        try:
            import json
            from shapely.geometry import shape as _shape
            from rasterio.features import rasterize as _rasterize
            from rasterio.transform import from_bounds as _from_bounds
        except ImportError as e:
            print(f"[Landscape] apply_osm_overlay skipped — missing dependency: {e}")
            return

        lat_min, lat_max, lon_min, lon_max = terrain_bounds
        rows, cols = self.shape

        # Affine transform: (west, south, east, north) → (cols × rows) raster
        # rasterio from_bounds produces row-0 = north (image convention).
        # After rasterize we flipud to match our row-0 = south convention.
        transform = _from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)

        road_idx  = (self.fuel_names.index("Urban_Road")
                     if "Urban_Road"  in self.fuel_names else -1)
        urban_idx = (self.fuel_names.index("Urban_Fabric")
                     if "Urban_Fabric" in self.fuel_names else -1)

        with open(osm_geojson_path) as fh:
            geojson = json.load(fh)

        urban_geoms = []
        road_geoms  = []

        for feat in geojson.get("features", []):
            ft   = feat["properties"].get("feature_type", "")
            geom = _shape(feat["geometry"])
            if ft == "urban" and urban_idx >= 0:
                urban_geoms.append(geom)
            elif ft == "road" and road_idx >= 0:
                # Keep roads as true centerlines so rasterization is 1-cell wide.
                # Width expansion (buffer) is intentionally disabled.
                road_geoms.append(geom)

        overlay = np.zeros((rows, cols), dtype=np.int32)

        if urban_geoms:
            burn = _rasterize(
                [(g, urban_idx) for g in urban_geoms],
                out_shape=(rows, cols), transform=transform,
                fill=0, dtype=np.int32,
            )
            overlay = np.where(burn > 0, burn, overlay)

        if road_geoms:
            burn = _rasterize(
                [(g, road_idx) for g in road_geoms],
                out_shape=(rows, cols), transform=transform,
                fill=0, dtype=np.int32, all_touched=False,
            )
            overlay = np.where(burn > 0, burn, overlay)

        # flipud: rasterio image orientation → our south-first orientation
        overlay_flipped = np.flipud(overlay)
        mask = overlay_flipped > 0
        self.fuel_map = np.where(mask, overlay_flipped, self.fuel_map)

        n_road  = int((self.fuel_map == road_idx).sum())  if road_idx  >= 0 else 0
        n_urban = int((self.fuel_map == urban_idx).sum()) if urban_idx >= 0 else 0
        self.osm_overlay_stats = {"road_cells": n_road, "urban_cells": n_urban}
        print(f"[Landscape] OSM overlay applied: {n_urban:,} urban cells, "
              f"{n_road:,} road cells")

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
        Returns empty dict for cells that cannot sustain surface fire propagation:
        Non_Combustible (roads surface, bare rock, urban hard surfaces) and
        Water (fuel_load=0 also suppresses spread in fire_model via valid_fuel mask)."""
        fuel_idx  = self.fuel_map[r, c]
        fuel_name = self.fuel_names[fuel_idx]
        if fuel_name in ("Non_Combustible", "Water", "Urban_Fabric", "Urban_Road"):
            return {}
        return GREEK_FUELS[fuel_name]

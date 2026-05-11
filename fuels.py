# fuels.py
# Greek/Mediterranean fuel models for the Rothermel (1972) equations.
# Now includes Non_Combustible to handle firebreaks from CORINE land cover
# (urban, water, bare rock).

GREEK_FUELS = {
    "Aleppo_Pine": {
        "heat_content": 8000,    # h (BTU/lb)
        "surface_ratio": 1500,   # sigma (ft^2/ft^3) - Pine needles
        "fuel_load": 0.5,        # w0 (lb/ft^2) - Dense canopy/litter
        "fuel_depth": 2.5,       # delta (ft) - Height of the fuel bed
        "moisture_ext": 0.25,    # Mx (fraction) - 25% moisture stops spread
        "bulk_density": 0.2      # rho_b (lb/ft^3)
    },

    "Phrygana_Low_Scrub": {
        "heat_content": 8000,
        "surface_ratio": 2500,   # High ratio (small leaves/twigs)
        "fuel_load": 0.15,
        "fuel_depth": 1.5,
        "moisture_ext": 0.15,    # Dries out very easily
        "bulk_density": 0.1
    },

    "Maquis_Dense_Shrub": {
        "heat_content": 8500,    # Higher oils in some Mediterranean shrubs
        "surface_ratio": 1800,
        "fuel_load": 0.45,       # Very heavy, dense thickets
        "fuel_depth": 4.0,
        "moisture_ext": 0.25,
        "bulk_density": 0.11
    },

    "Dry_Grass": {
        "heat_content": 8000,
        "surface_ratio": 3000,   # Extremely high (ignites instantly)
        "fuel_load": 0.08,
        "fuel_depth": 1.0,
        "moisture_ext": 0.12,
        "bulk_density": 0.08
    },

    "Oak_Forest": {
        "heat_content": 8000,
        "surface_ratio": 1200,
        "fuel_load": 0.35,
        "fuel_depth": 0.5,
        "moisture_ext": 0.30,
        "bulk_density": 0.7
    },

    "Olive_Grove": {
        "heat_content": 8000,
        "surface_ratio": 1400,
        "fuel_load": 0.25,
        "fuel_depth": 1.0,
        "moisture_ext": 0.20,
        "bulk_density": 0.25
    },

    # NEW: firebreak / non-combustible class. fuel_load=0 makes the
    # `valid_fuel = w0 > 0.0` mask in fire_model.py automatically
    # zero the rate of spread for these cells. Used for urban areas,
    # bare rock, water bodies, and roads detected from CORINE.
    "Non_Combustible": {
        "heat_content": 0.0,
        "surface_ratio": 1e-9,   # avoid div-by-zero in Rothermel constants
        "fuel_load": 0.0,        # THE critical zero - kills ROS
        "fuel_depth": 0.0,
        "moisture_ext": 0.30,
        "bulk_density": 1e-9
    }
}

# config.py

# --- 1. Grid & Scale ---
GRID_SIZE = (200, 200)       # Size of the map in cells
CELL_SIZE_METERS = 5.0       # Physical size of one cell

# --- 2. Environment & Weather ---
TEMPERATURE_C = 38.0         # Air temperature in Celsius
RELATIVE_HUMIDITY = 10.0     # Relative humidity in percentage (0-100)
WIND_SPEED = 10.0            
WIND_DIRECTION = 45

# --- 3. Fire Physics ---
BURN_TIME_STEPS = 60         # Increased to 60 so fire survives low-wind conditions
P_BASE_SPREAD = 0.5          # Balanced spread (if used in future probabilistic models)
MAX_SIMULATION_STEPS = 5000  # How many steps to calculate before stopping
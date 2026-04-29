import numpy as np

# ---------------------------------------------------------------------------
# Physical / empirical constants (Rothermel 1972)
# ---------------------------------------------------------------------------
_PARTICLE_DENSITY   = 32.0    
_MINERAL_CONTENT_Se = 0.01    
_ETA_S = 0.174 * (_MINERAL_CONTENT_Se ** -0.19)  

# Unit-conversion helpers
_M_TO_FT      = 3.28084       
_MS_TO_FTMIN  = 196.8504      
_FTMIN_TO_MPM = 0.3048        


def shift_array(arr, dr, dc):
    """
    Fast NumPy array shifting to align neighbor matrices.
    Shifts the array 'arr' by 'dr' rows and 'dc' columns.
    """
    shifted = np.zeros_like(arr)
    r_src_start = max(0, -dr)
    r_src_end = arr.shape[0] - max(0, dr)
    c_src_start = max(0, -dc)
    c_src_end = arr.shape[1] - max(0, dc)

    r_tgt_start = max(0, dr)
    r_tgt_end = arr.shape[0] + min(0, dr)
    c_tgt_start = max(0, dc)
    c_tgt_end = arr.shape[1] + min(0, dc)

    shifted[r_tgt_start:r_tgt_end, c_tgt_start:c_tgt_end] = arr[r_src_start:r_src_end, c_src_start:c_src_end]
    return shifted


class CellularAutomataFire:
    """
    Fully Vectorized, Deterministic Rothermel CA Fire Spread Model.
    """

    def __init__(self, landscape, config):
        self.landscape = landscape
        self.config    = config
        
        rows, cols = landscape.shape
        self.state             = np.zeros((rows, cols), dtype=np.int8)
        self.burn_timer        = np.zeros((rows, cols), dtype=np.float32)
        self.ignition_fraction = np.zeros((rows, cols), dtype=np.float32) # Replaces randomness

        self.dt = config.dt if hasattr(config, 'dt') else 0.1   

        self._neighbors = [
            (-1, 0), (1,  0), (0, -1), (0,  1),
            (-1,-1), (-1, 1), (1, -1), (1,  1)
        ]

        # Pre-calculate the entire grid's physics matrices
        self._precompute_ros_grid()


    def ignite(self, r: int, c: int) -> None:
        """Set a single cell alight."""
        if self.state[r, c] == 0:
            self.state[r, c] = 1
            self.burn_timer[r, c] = self.config.BURN_TIME_STEPS


    def _precompute_ros_grid(self) -> None:
        """
        Calculates the Rate of Spread for the entire matrix simultaneously.
        Creates self.p_spread (8, rows, cols) which holds the fractional heat
        transferred per dt for all 8 incoming directions.
        """
        rows, cols = self.landscape.shape
        self.p_spread = np.zeros((8, rows, cols), dtype=np.float32)

        # 1. Map fuel dictionary properties to pure NumPy grids
        w0 = np.zeros((rows, cols), dtype=np.float32)
        sigma = np.zeros((rows, cols), dtype=np.float32)
        mx = np.zeros((rows, cols), dtype=np.float32)
        h = np.zeros((rows, cols), dtype=np.float32)
        rho_b = np.zeros((rows, cols), dtype=np.float32)

        # Extract properties by sampling one cell of each type to decouple from fuels.py
        for fuel_idx in range(len(self.landscape.fuel_names)):
            coords = np.argwhere(self.landscape.fuel_map == fuel_idx)
            if len(coords) > 0:
                r, c = coords[0]
                fuel = self.landscape.get_fuel_at(r, c)
                mask = (self.landscape.fuel_map == fuel_idx)
                
                w0[mask]    = fuel.get('fuel_load', 0.0)
                sigma[mask] = fuel.get('surface_ratio', 1e-9)
                mx[mask]    = fuel.get('moisture_ext', 0.3)
                h[mask]     = fuel.get('heat_content', 8000)
                rho_b[mask] = fuel.get('bulk_density', 0.1)

        # Safe matrices to prevent division by zero
        valid_fuel = w0 > 0.0
        sigma = np.where(sigma > 0, sigma, 1e-9)
        rho_b = np.where(rho_b > 0, rho_b, 1e-9)
        mf = self.landscape.moisture

        # 2. Packing ratios & Moisture damping
        beta = rho_b / _PARTICLE_DENSITY
        beta_op = 3.348 * (sigma ** -0.8189)
        rel_beta = beta / beta_op

        rm = np.clip(mf / mx, 0.0, 1.0)
        eta_m = 1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3

        # 3. Reaction intensity
        gamma_max = (sigma ** 1.5) / (495.0 + 0.0594 * sigma ** 1.5)
        A = 1.0 / np.maximum(4.774 * (sigma ** 0.1) - 7.27, 1e-6)
        gamma = gamma_max * (rel_beta ** A) * np.exp(A * (1.0 - rel_beta))
        wn = w0 * (1.0 - 0.0555)
        ir = gamma * wn * h * eta_m * _ETA_S

        # 4. Flux ratio & heat sink
        xi = np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
        epsilon = np.exp(-138.0 / sigma)
        q_ig = 250.0 + 1116.0 * mf
        denom = rho_b * epsilon * q_ig

        base_ros_numerator = ir * xi
        base_ros_denominator = np.where(denom > 0, denom, 1e-9)

        valid_cells = (denom > 0) & (ir > 0) & valid_fuel
        wind_vec = np.array([float(self.landscape.wind_u), float(self.landscape.wind_v)])

        # 5. Calculate directional modifiers (Wind and Slope) for all 8 directions
        for i, (dr, dc) in enumerate(self._neighbors):
            
            # Shift elevation to find the slope from the source cell to the target cell
            source_elev = shift_array(self.landscape.elevation, dr, dc)
            dz = self.landscape.elevation - source_elev

            dist_m = self.config.CELL_SIZE_METERS * np.sqrt(dr**2 + dc**2)
            dist_ft = dist_m * _M_TO_FT

            tan_phi = np.clip((dz * _M_TO_FT) / dist_ft, 0.0, None)
            phi_s = 5.275 * (np.maximum(beta, 1e-9) ** -0.3) * (tan_phi ** 2)

            # Wind factor (Scalar dot product applied grid-wide)
            dr_vec = np.array([dc, dr], dtype=float)
            dr_vec /= np.linalg.norm(dr_vec)
            u_ft_min = max(0.0, float(np.dot(wind_vec, dr_vec))) * _MS_TO_FTMIN

            c_w = 7.47 * np.exp(-0.133 * sigma ** 0.55)
            b_w = 0.0252 * sigma ** 0.54
            e_w = 0.715 * np.exp(-3.59e-4 * sigma)
            phi_w = c_w * (u_ft_min ** b_w) * (np.maximum(rel_beta, 1e-9) ** -e_w)

            # 6. Final Rate of Spread Matrix for this direction
            r_ft_min = (base_ros_numerator * (1.0 + phi_w + phi_s)) / base_ros_denominator
            r_m_min = r_ft_min * _FTMIN_TO_MPM
            r_m_min = np.where(valid_cells, r_m_min, 0.0)

            # Store the fraction of a cell crossed per time step (dt)
            self.p_spread[i] = (r_m_min * self.dt) / dist_m


    def step(self) -> None:
        """Vectorized CA time step. No random rolls, no loops over cells."""
        
        # 1. Update Burn Timers using Boolean masks
        burning_mask = (self.state == 1)
        self.burn_timer[burning_mask] -= 1

        # Turn cells to ash (2) when timer expires
        burned_out = burning_mask & (self.burn_timer <= 0)
        self.state[burned_out] = 2
        burning_mask[burned_out] = False # Update the mask to exclude new ash

        # 2. Accumulate heat (ignition fraction) onto unburned cells
        unburned_mask = (self.state == 0)

        for i, (dr, dc) in enumerate(self._neighbors):
            # Shift the burning mask to overlay onto the unburned target cells
            burning_neighbors = shift_array(burning_mask, dr, dc)
            threatened = burning_neighbors & unburned_mask
            
            # Add the precise fractional heat from that specific neighbor
            self.ignition_fraction[threatened] += self.p_spread[i][threatened]

        # 3. Deterministic Ignition: Any cell that accumulated >= 1.0 heat catches fire
        ignited = unburned_mask & (self.ignition_fraction >= 1.0)
        
        self.state[ignited] = 1
        self.burn_timer[ignited] = self.config.BURN_TIME_STEPS
        self.ignition_fraction[ignited] = 0.0 # Reset heat
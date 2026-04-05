import numpy as np

# ---------------------------------------------------------------------------
# Physical / empirical constants (Rothermel 1972)
# ---------------------------------------------------------------------------
_PARTICLE_DENSITY   = 32.0    # rho_p  (lb/ft^3) — standard oven-dry wood assumption
_MINERAL_CONTENT_Se = 0.01    # S_e    (fraction) — effective mineral content (1 %)
_ETA_S = 0.174 * (_MINERAL_CONTENT_Se ** -0.19)  # ≈ 0.417  (dimensionless, constant)

# Unit-conversion helpers
_M_TO_FT      = 3.28084       # 1 m  → ft
_MS_TO_FTMIN  = 196.8504      # 1 m/s → ft/min  (= 3.28084 × 60)
_FTMIN_TO_MPM = 0.3048        # 1 ft/min → m/min


class CellularAutomataFire:
    """
    Rothermel (1972) fire-spread model mapped onto a 2-D cellular automaton.

    Cell states
    -----------
    0  unburned
    1  burning
    2  burned out
    """

    def __init__(self, landscape, config):
        self.landscape = landscape
        self.config    = config
        self.state     = np.zeros(landscape.shape, dtype=np.int8)
        self.burn_timer = np.zeros(landscape.shape, dtype=np.float32)

        # Time step — 1 simulated minute per CA step
        self.dt = 0.001   

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ignite(self, x: int, y: int) -> None:
        """Set a single cell alight."""
        if self.state[x, y] == 0:          # only unburned cells can ignite
            self.state[x, y]      = 1
            self.burn_timer[x, y] = self.config.BURN_TIME_STEPS

    # ------------------------------------------------------------------
    # Rate-of-Spread  (Rothermel 1972)
    # ------------------------------------------------------------------

    def _calculate_ros(self, r: int, c: int, nr: int, nc: int) -> float:
        """
        Compute Rate of Spread from burning cell (r, c) into neighbour (nr, nc).

        All intermediate quantities follow Rothermel (1972) Tables 5–8.
        Working units are imperial (lb, ft, BTU) internally; the return
        value is converted to metres per minute.

        Returns 0.0 whenever the target cell carries no fuel or when the
        physics produce a non-positive / non-finite result.
        """

        # ----------------------------------------------------------------
        # 1.  Fuel properties at the *target* cell
        # ----------------------------------------------------------------
        fuel = self.landscape.get_fuel_at(nr, nc)
        mf   = float(self.landscape.moisture[nr, nc])   # fraction (e.g. 0.08)

        # Guard: rocky / non-fuel cells
        w0_raw = float(fuel.get('fuel_load', 0.0))
        if w0_raw <= 0.0:
            return 0.0

        sigma   = float(fuel['surface_ratio'])   # ft²/ft³
        w0      = w0_raw                         # lb/ft²  (already imperial)
        mx      = float(fuel['moisture_ext'])    # fraction
        h       = float(fuel['heat_content'])    # BTU/lb
        rho_b   = float(fuel['bulk_density'])    # lb/ft³

        # ----------------------------------------------------------------
        # 2.  Packing-ratio terms  (Table 7)
        # ----------------------------------------------------------------
        beta     = rho_b / _PARTICLE_DENSITY            # packing ratio β
        beta_op  = 3.348 * (sigma ** -0.8189)           # optimal β
        rel_beta = beta / beta_op                       # β / β_op

        # ----------------------------------------------------------------
        # 3.  Moisture-damping coefficient η_m  (Table 7)
        # ----------------------------------------------------------------
        rm   = min(mf / mx, 1.0)                        # clamp at extinction
        eta_m = 1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3

        # ----------------------------------------------------------------
        # 4.  Reaction intensity I_R  (Table 8)
        # ----------------------------------------------------------------
        #   Optimum reaction velocity  Γ'_max
        gamma_max = (sigma ** 1.5) / (495.0 + 0.0594 * sigma ** 1.5)

        #   Full Rothermel Γ' using the A-exponent  (eq. 36–38)
        #     A = 1 / (4.774 σ^0.1 − 7.27)
        A     = 1.0 / max(4.774 * (sigma ** 0.1) - 7.27, 1e-6)
        gamma = gamma_max * (rel_beta ** A) * np.exp(A * (1.0 - rel_beta))

        wn = w0 * (1 - 0.0555) #Standard total mineral content is usually assumed to be 5.55% (minerals dont burn)
        ir = gamma * wn * h * eta_m * _ETA_S

        # ----------------------------------------------------------------
        # 5.  Slope factor φ_s  (Table 8)
        # ----------------------------------------------------------------
        dz     = float(self.landscape.elevation[nr, nc]) \
                 - float(self.landscape.elevation[r,  c ])

        dist_m  = self.config.CELL_SIZE_METERS * np.sqrt((nr - r)**2 + (nc - c)**2)
        dist_ft = dist_m * _M_TO_FT

        # Guard against zero distance (should never happen in a proper grid,
        # but defensive programming costs nothing)
        if dist_ft < 1e-6:
            return 0.0

        # Only uphill spread is accelerated; downhill → φ_s = 0
        tan_phi = max(0.0, (dz * _M_TO_FT) / dist_ft)
        phi_s   = 5.275 * (max(beta, 1e-9) ** -0.3) * (tan_phi ** 2)

        # ----------------------------------------------------------------
        # 6.  Wind factor φ_w  (Table 8)
        # ----------------------------------------------------------------
        # X-axis is columns (nc - c)
        # Y-axis is rows (nr - r)
        dr_vec = np.array([nc - c, nr - r], dtype=float) 
        dr_vec /= np.linalg.norm(dr_vec)                 # unit spread direction

        wind_vec = np.array([
            float(self.landscape.wind_u), # X-component
            float(self.landscape.wind_v)  # Y-component
        ])

        # Effective wind component along the spread direction (ft/min)
        u_ft_min = max(0.0, float(np.dot(wind_vec, dr_vec))) * _MS_TO_FTMIN

        c_w = 7.47   * np.exp(-0.133 * sigma ** 0.55)
        b_w = 0.0252 * sigma ** 0.54
        e_w = 0.715  * np.exp(-3.59e-4 * sigma)

        phi_w = c_w * (u_ft_min ** b_w) * (max(rel_beta, 1e-9) ** -e_w)

        # ----------------------------------------------------------------
        # 7.  Propagating flux ratio ξ  (Table 8)
        # ----------------------------------------------------------------
        xi = (
            np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1))
            / (192.0 + 0.2595 * sigma)
        )

        # ----------------------------------------------------------------
        # 8.  Heat sink terms  (Table 8)
        # ----------------------------------------------------------------
        epsilon = np.exp(-138.0 / sigma)

        # Heat of pre-ignition Q_ig  (BTU/lb).
        # mf is a fraction (e.g. 0.08), so 1116 × 0.08 = 89.3 BTU/lb — correct.
        q_ig = 250.0 + 1116.0 * mf

        # ----------------------------------------------------------------
        # 9.  Rate of Spread R  (ft/min → m/min)
        # ----------------------------------------------------------------
        denom = rho_b * epsilon * q_ig
        if denom <= 0.0 or ir <= 0.0:
            return 0.0

        r_ft_min = (ir * xi * (1.0 + phi_w + phi_s)) / denom

        if not np.isfinite(r_ft_min) or r_ft_min < 0.0:
            return 0.0

        return r_ft_min * _FTMIN_TO_MPM

    # ------------------------------------------------------------------
    # CA time-step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one time step (self.dt minutes)."""
        new_state = self.state.copy()
        burning_cells = np.argwhere(self.state == 1)

        _neighbors = [
            (-1, 0), (1,  0), (0, -1), (0,  1),
            (-1,-1), (-1, 1), (1, -1), (1,  1),
        ]
        rows, cols = self.landscape.shape

        # ----------------------------------------------------------------
        # PHASE 1: Calculate cumulative probability for all unburned cells
        # ----------------------------------------------------------------
        # Matrix to store the calculated chance of ignition for each cell
        p_ignition_grid = np.zeros((rows, cols), dtype=float)

        for (r, c) in burning_cells:
            
            # Update the burn timer for the current fire cell
            self.burn_timer[r, c] -= 1
            if self.burn_timer[r, c] <= 0:
                new_state[r, c] = 2          # burned out
                continue

            # Look at neighbors and calculate how much threat we pose to them
            for dr, dc in _neighbors:
                nr, nc = r + dr, c + dc

                # Guard: Out of bounds
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                # Guard: Only calculate for unburned vegetation
                if self.state[nr, nc] != 0:  
                    continue

                ros = self._calculate_ros(r, c, nr, nc)
                if ros <= 0.0:
                    continue
                
                # --- NEW: Add ±15% stochastic noise to break grid symmetry ---
                # noise_factor = np.random.uniform(0.85, 1.15)
                # ros *= noise_factor

                # Base probability from this specific neighbor
                dist = self.config.CELL_SIZE_METERS * np.sqrt(dr**2 + dc**2)
                p_incoming = np.clip((ros * self.dt) / dist, 0.0, 1.0)

                # Combine probabilities safely without exceeding 1.0
                # P(A U B) = 1 - (1 - P(A)) * (1 - P(B))
                p_current = p_ignition_grid[nr, nc]
                p_ignition_grid[nr, nc] = 1.0 - (1.0 - p_current) * (1.0 - p_incoming)


        # ----------------------------------------------------------------
        # PHASE 2: Apply the probabilities (Roll the dice exactly once)
        # ----------------------------------------------------------------
        # Find all cells that have a > 0% chance of igniting this step
        threatened_cells = np.argwhere(p_ignition_grid > 0.0)

        for (r, c) in threatened_cells:
            # Roll a single random number for the cell
            if np.random.rand() < p_ignition_grid[r, c]:
                new_state[r, c] = 1
                self.burn_timer[r, c] = self.config.BURN_TIME_STEPS

        self.state = new_state

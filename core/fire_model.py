import numpy as np
from air.air import compute_wind_field  # air/ is at project root

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

_DIR_NAMES = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]


def shift_array(arr, dr, dc):
    """Fast NumPy array shifting to align neighbor matrices."""
    shifted = np.zeros_like(arr)
    r_src_start = max(0, -dr);  r_src_end = arr.shape[0] - max(0, dr)
    c_src_start = max(0, -dc);  c_src_end = arr.shape[1] - max(0, dc)
    r_tgt_start = max(0, dr);   r_tgt_end = arr.shape[0] + min(0, dr)
    c_tgt_start = max(0, dc);   c_tgt_end = arr.shape[1] + min(0, dc)
    shifted[r_tgt_start:r_tgt_end, c_tgt_start:c_tgt_end] = \
        arr[r_src_start:r_src_end, c_src_start:c_src_end]
    return shifted


def _compass_bearing(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((deg + 22.5) / 45) % 8]


def _line_hits_mask(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    """Return True if the raster line between two cells crosses a masked cell."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        if 0 <= r0 < mask.shape[0] and 0 <= c0 < mask.shape[1] and mask[r0, c0]:
            return True
        if r0 == r1 and c0 == c1:
            break
        err2 = err * 2
        if err2 > -dc:
            err -= dc
            r0 += step_r
        if err2 < dr:
            err += dr
            c0 += step_c
    return False


class CellularAutomataFire:
    """
    Fully Vectorised, Deterministic Rothermel CA Fire Spread Model.

    State arrays
    ------------
    water_amount          : accumulated water from brush (0..∞); drives probabilistic extinction.
    containment_strength  : firefighter-line damping [0..1]; reduces heat at target cells.
    containment_humidity_boost : local moisture increase from containment water application.
    _containment_decay    : per-step decay rate for containment_strength.
    _ignition_sources     : dict (r, c) -> list of contributing neighbor dicts at ignition time.
    """

    def __init__(self, landscape, config, dt: float = None):
        self.landscape = landscape
        self.config    = config

        rows, cols = landscape.shape
        self.state             = np.zeros((rows, cols), dtype=np.int8)
        self.burn_timer        = np.zeros((rows, cols), dtype=np.float32)
        self.ignition_fraction = np.zeros((rows, cols), dtype=np.float32)

        self.blocked_mask             = np.zeros((rows, cols), dtype=np.bool_)
        self.wetness                  = np.zeros((rows, cols), dtype=np.float32)
        self.water_amount             = np.zeros((rows, cols), dtype=np.float32)
        self.containment_strength     = np.zeros((rows, cols), dtype=np.float32)
        self.containment_humidity_boost = np.zeros((rows, cols), dtype=np.float32)
        self._containment_decay       = np.zeros((rows, cols), dtype=np.float32)

        # Backward-compat alias
        self.suppression_strength = self.containment_strength

        # Explanation tracking
        self._ignition_step    = np.full((rows, cols), -1, dtype=np.int32)
        self._ignition_sources: dict = {}   # (r, c) -> list of source dicts
        self._sim_step         = 0
        # Transient list of blocked encounter events detected during step()
        self._blocked_encounters = []  # list of (r, c, dir_index)

        # Optimizer-tunable physics parameters
        self._ignition_threshold  = 1.0   # heat accumulation needed to ignite
        self._spotting_rate_mult  = 1.0   # multiplier on firebrand lofting probability
        self._canopy_wind_mult    = 1.0   # midflame wind scaling (WAF proxy), survives rebuilds

        config_dt = config.dt if hasattr(config, 'dt') else 0.1
        self.dt   = dt if dt is not None else config_dt

        base_bts = config.BURN_TIME_STEPS if hasattr(config, 'BURN_TIME_STEPS') else 60
        self._burn_time_steps = max(1, int(round(base_bts * config_dt / self.dt)))

        self._neighbors = [
            (-1, 0), (1,  0), (0, -1), (0,  1),
            (-1,-1), (-1, 1), (1, -1), (1,  1)
        ]

        self._precompute_ros_grid()

    # ──────────────────────────────────────────────────────────────────────────
    # Ignition API
    # ──────────────────────────────────────────────────────────────────────────
    def ignite(self, r: int, c: int) -> None:
        """Set a single cell alight."""
        if not self.landscape.get_fuel_at(r, c):
            return
        if self.state[r, c] == 0 and not self.blocked_mask[r, c]:
            self.state[r, c] = 1
            self.burn_timer[r, c] = self._burn_time_steps
            self._ignition_step[r, c] = self._sim_step

    def ignite_region(self, mask: np.ndarray) -> int:
        """Ignite all cells in a boolean mask. Returns count of newly ignited cells."""
        rows, cols = self.state.shape
        if mask.shape != (rows, cols):
            raise ValueError("ignite_region mask shape mismatch")
        valid = mask & (self.state == 0) & (~self.blocked_mask) & self._combustible_mask
        count = int(valid.sum())
        self.state[valid] = 1
        self.burn_timer[valid] = self._burn_time_steps
        self._ignition_step[valid] = self._sim_step
        return count

    # ──────────────────────────────────────────────────────────────────────────
    # Firebreak / water API
    # ──────────────────────────────────────────────────────────────────────────
    def apply_firebreak_mask(self, mask: np.ndarray) -> None:
        if mask.shape != self.state.shape:
            raise ValueError("firebreak mask shape mismatch")
        self.blocked_mask[mask] = True
        self.wetness[mask] = 1.0
        self.state[mask] = 0
        self.burn_timer[mask] = 0
        self.ignition_fraction[mask] = 0.0

    def apply_water_mask(self, mask: np.ndarray, wetness: float = 0.92) -> None:
        """Apply temporary wetting (backward-compatible; also increments water_amount)."""
        if mask.shape != self.state.shape:
            raise ValueError("water mask shape mismatch")
        wet = float(np.clip(wetness, 0.0, 1.0))
        self.wetness[mask] = np.maximum(self.wetness[mask], wet)
        self.water_amount[mask] = np.minimum(self.water_amount[mask] + wet * 1.5, 10.0)
        active = mask & (self.state == 1)
        # Doused cells were on fire — they stay scorched (state 2, the dark
        # "burned" overlay); only the orange "burning" indication goes away.
        self.state[active] = 2
        self.burn_timer[active] = 0
        clear_heat = mask & (self.state != 2)
        self.ignition_fraction[clear_heat] = 0.0

    def apply_water_brush(self, r: int, c: int, radius_cells: float,
                          intensity: float = 1.0, falloff: float = 0.8,
                          hardness: float = 0.5) -> int:
        """Photoshop-style water brush. Returns number of cells affected."""
        rows, cols = self.state.shape
        rc_int = max(1, int(np.ceil(radius_cells)))

        r0 = max(0, r - rc_int);  r1 = min(rows, r + rc_int + 1)
        c0 = max(0, c - rc_int);  c1 = min(cols, c + rc_int + 1)

        rr = (np.arange(r0, r1, dtype=np.float32) - r) / max(radius_cells, 1e-3)
        cc = (np.arange(c0, c1, dtype=np.float32) - c) / max(radius_cells, 1e-3)
        dist = np.hypot(rr[:, None], cc[None, :])

        within    = dist <= 1.0
        hard_zone = dist < max(hardness, 0.0)
        soft_zone = within & ~hard_zone

        weight = np.where(hard_zone, 1.0, 0.0).astype(np.float32)
        if soft_zone.any():
            denom  = max(1.0 - hardness, 1e-3)
            t_outer = np.clip((dist - hardness) / denom, 0.0, 1.0)
            power_w = np.maximum(1.0 - t_outer, 0.0) ** max(falloff, 0.5)
            weight  = np.where(soft_zone, power_w.astype(np.float32), weight)

        add = (weight * float(intensity)).astype(np.float32)
        self.water_amount[r0:r1, c0:c1] = np.minimum(
            self.water_amount[r0:r1, c0:c1] + add, 10.0)
        self.wetness[r0:r1, c0:c1] = np.minimum(
            np.maximum(self.wetness[r0:r1, c0:c1], add * 0.6), 1.0)

        local_burning = (self.state[r0:r1, c0:c1] == 1) & within
        if local_burning.any():
            burning_full = np.zeros(self.state.shape, dtype=bool)
            burning_full[r0:r1, c0:c1] = local_burning
            self._apply_water_extinction(burning_full)

        return int(within.sum())

    # ──────────────────────────────────────────────────────────────────────────
    # Containment line (unified firebreak + suppression)
    # ──────────────────────────────────────────────────────────────────────────
    def apply_containment_line(self, line_mask: np.ndarray,
                               strength: float = 0.5,
                               effect_radius_cells: int = 3,
                               water_application: float = 0.0,
                               humidity_boost: float = 0.0,
                               decay_rate: float = 0.0) -> int:
        """
        Paint a firefighter containment corridor.

        Parameters
        ----------
        line_mask            : boolean (rows, cols) rasterised polyline
        strength             : peak damping [0..1] — 1.0 ≈ near-total blockage
        effect_radius_cells  : corridor half-width in cells
        water_application    : additional water applied along line [0..3]
        humidity_boost       : local moisture increase along corridor [0..0.3]
        decay_rate           : containment_strength decay per simulation step

        Returns
        -------
        Number of cells with non-zero containment applied.
        """
        if line_mask.shape != self.state.shape:
            raise ValueError("containment mask shape mismatch")

        try:
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~line_mask)
        except ImportError:
            rows, cols = self.state.shape
            dist = np.full((rows, cols), float(effect_radius_cells + 1))
            lr, lc = np.where(line_mask)
            if len(lr):
                r_idx = np.arange(rows)[:, None]
                c_idx = np.arange(cols)[None, :]
                for i in range(min(len(lr), 5000)):
                    d = np.hypot(r_idx - lr[i], c_idx - lc[i])
                    dist = np.minimum(dist, d)

        max_d = max(effect_radius_cells, 1)
        weight = np.clip(1.0 - dist / max_d, 0.0, 1.0).astype(np.float32)
        supp   = weight * float(strength)

        self.containment_strength = np.maximum(
            self.containment_strength, supp)
        # Keep alias in sync
        self.suppression_strength = self.containment_strength

        if humidity_boost > 0.0:
            boost = weight * float(humidity_boost)
            self.containment_humidity_boost = np.maximum(
                self.containment_humidity_boost, boost)

        if water_application > 0.0:
            add_w = weight * float(water_application)
            self.water_amount = np.minimum(
                self.water_amount + add_w, 10.0)

        if decay_rate > 0.0:
            active = supp > 0
            self._containment_decay[active] = np.maximum(
                self._containment_decay[active], float(decay_rate))

        return int((supp > 0).sum())

    # Keep old name as alias for backward compatibility
    def apply_suppression_line(self, line_mask, strength=0.5,
                               effect_radius_cells=3, **kw):
        return self.apply_containment_line(
            line_mask, strength=strength,
            effect_radius_cells=effect_radius_cells, **kw)

    # ──────────────────────────────────────────────────────────────────────────
    # ROS pre-computation
    # ──────────────────────────────────────────────────────────────────────────
    def _precompute_ros_grid(self) -> None:
        rows, cols = self.landscape.shape
        self.p_spread = np.zeros((8, rows, cols), dtype=np.float32)

        w0    = np.zeros((rows, cols), dtype=np.float32)
        sigma = np.zeros((rows, cols), dtype=np.float32)
        mx    = np.zeros((rows, cols), dtype=np.float32)
        h     = np.zeros((rows, cols), dtype=np.float32)
        rho_b = np.zeros((rows, cols), dtype=np.float32)

        for fuel_idx in range(len(self.landscape.fuel_names)):
            coords = np.argwhere(self.landscape.fuel_map == fuel_idx)
            if len(coords) > 0:
                r, c   = coords[0]
                fuel   = self.landscape.get_fuel_at(r, c)
                mask_f = (self.landscape.fuel_map == fuel_idx)
                w0[mask_f]    = fuel.get('fuel_load', 0.0)
                sigma[mask_f] = fuel.get('surface_ratio', 1e-9)
                mx[mask_f]    = fuel.get('moisture_ext', 0.3)
                h[mask_f]     = fuel.get('heat_content', 8000)
                rho_b[mask_f] = fuel.get('bulk_density', 0.1)

        self._cached_w0 = w0.copy()

        valid_fuel = w0 > 0.0
        sigma = np.where(sigma > 0, sigma, 1e-9)
        rho_b = np.where(rho_b > 0, rho_b, 1e-9)
        mf    = self.landscape.moisture

        beta     = rho_b / _PARTICLE_DENSITY
        beta_op  = 3.348 * (sigma ** -0.8189)
        rel_beta = beta / beta_op

        rm    = np.clip(mf / mx, 0.0, 1.0)
        eta_m = 1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3

        gamma_max = (sigma ** 1.5) / (495.0 + 0.0594 * sigma ** 1.5)
        A         = 1.0 / np.maximum(4.774 * (sigma ** 0.1) - 7.27, 1e-6)
        exp_arg   = np.clip(A * (1.0 - rel_beta), -100.0, 100.0)
        with np.errstate(over="ignore", invalid="ignore"):
            gamma = gamma_max * (rel_beta ** A) * np.exp(exp_arg)
        gamma = np.where(np.isfinite(gamma), gamma, 0.0)
        wn    = w0 * (1.0 - 0.0555)
        ir    = gamma * wn * h * eta_m * _ETA_S

        xi      = np.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
        epsilon = np.exp(-138.0 / sigma)
        q_ig    = 250.0 + 1116.0 * mf
        denom   = rho_b * epsilon * q_ig

        base_ros_numerator   = ir * xi
        base_ros_denominator = np.where(denom > 0, denom, 1e-9)
        valid_cells          = (denom > 0) & (ir > 0) & valid_fuel
        self._combustible_mask = valid_cells

        wind_u_grid, wind_v_grid = compute_wind_field(self.landscape, self.config)
        canopy_mult = float(getattr(self, "_canopy_wind_mult", 1.0))
        if abs(canopy_mult - 1.0) > 1e-6:
            wind_u_grid = np.clip(wind_u_grid * canopy_mult, -100.0, 100.0).astype(np.float32)
            wind_v_grid = np.clip(wind_v_grid * canopy_mult, -100.0, 100.0).astype(np.float32)
        self._wind_u_grid = wind_u_grid
        self._wind_v_grid = wind_v_grid

        c_w = 7.47 * np.exp(-0.133 * sigma**0.55)
        b_w = 0.0252 * sigma**0.54
        e_w = 0.715  * np.exp(-3.59e-4 * sigma)

        for i, (dr, dc) in enumerate(self._neighbors):
            source_elev = shift_array(self.landscape.elevation, dr, dc)
            dz = self.landscape.elevation - source_elev

            dist_m  = self.config.CELL_SIZE_METERS * np.sqrt(dr**2 + dc**2)
            dist_ft = dist_m * _M_TO_FT

            tan_phi = np.clip((dz * _M_TO_FT) / dist_ft, 0.0, None)
            phi_s   = 5.275 * (np.maximum(beta, 1e-9) ** -0.3) * (tan_phi ** 2)

            dr_vec = np.array([dc, dr], dtype=np.float64)
            dr_vec /= np.linalg.norm(dr_vec)
            u_effective = wind_u_grid * dr_vec[0] + wind_v_grid * dr_vec[1]
            u_ft_min    = np.maximum(0.0, u_effective) * _MS_TO_FTMIN
            u_ft_min_safe = np.minimum(u_ft_min, 1500.0)
            phi_w = c_w * (u_ft_min_safe ** b_w) * (np.maximum(rel_beta, 1e-9) ** -e_w)

            r_ft_min = (base_ros_numerator * (1.0 + phi_w + phi_s)) / base_ros_denominator
            r_m_min  = r_ft_min * _FTMIN_TO_MPM
            r_m_min  = np.where(valid_cells, r_m_min, 0.0)
            self.p_spread[i] = np.minimum((r_m_min * self.dt) / dist_m, 1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Main simulation step
    # ──────────────────────────────────────────────────────────────────────────
    def step(self) -> None:
        rows, cols = self.state.shape

        # 1. Burn timer countdown
        burning_mask = (self.state == 1)
        if self.blocked_mask.any():
            self.state[self.blocked_mask] = 0
            self.burn_timer[self.blocked_mask] = 0
            self.ignition_fraction[self.blocked_mask] = 0.0
            burning_mask[self.blocked_mask] = False
        self.burn_timer[burning_mask] -= 1

        burned_out = burning_mask & (self.burn_timer <= 0)
        self.state[burned_out] = 2
        burning_mask[burned_out] = False

        # 1b. Probabilistic water extinction
        if burning_mask.any() and np.any(self.water_amount > 0.05):
            self._apply_water_extinction(burning_mask)
            burning_mask = (self.state == 1)

        # 2. Heat accumulation from burning neighbours
        unburned_mask  = (self.state == 0) & (~self.blocked_mask) & self._combustible_mask
        dry_factor     = 1.0 - self.wetness

        if np.any(self.containment_strength > 0.0):
            suppression_factor = 1.0 - self.containment_strength
        else:
            suppression_factor = None

        for i, (dr, dc) in enumerate(self._neighbors):
            burning_neighbors = shift_array(burning_mask, dr, dc)
            # Detect burning neighbors adjacent to blocked cells (firebreak hit)
            blocked_threatened = burning_neighbors & (self.state == 0) & self.blocked_mask
            if blocked_threatened.any():
                rr, cc = np.where(blocked_threatened)
                for r_hit, c_hit in zip(rr.tolist(), cc.tolist()):
                    # record encounter: cell that is blocked and direction index of source
                    self._blocked_encounters.append((int(r_hit), int(c_hit), int(i)))
            threatened = burning_neighbors & unburned_mask
            if not threatened.any():
                continue
            heat = self.p_spread[i] * dry_factor
            if suppression_factor is not None:
                heat = heat * suppression_factor
            self.ignition_fraction[threatened] += heat[threatened]

        # 3. Deterministic ignition at ≥ _ignition_threshold accumulated heat
        ignited = unburned_mask & (self.ignition_fraction >= self._ignition_threshold)
        if ignited.any():
            # Record contributing sources before changing state
            dry_fac_snap = dry_factor   # snapshot
            new_r, new_c = np.where(ignited)
            for r_ig, c_ig in zip(new_r.tolist(), new_c.tolist()):
                sources = []
                for i_nb, (dr, dc) in enumerate(self._neighbors):
                    nr, nc = r_ig - dr, c_ig - dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if self.state[nr, nc] == 1:
                        p    = float(self.p_spread[i_nb, r_ig, c_ig])
                        cont = p * float(dry_fac_snap[r_ig, c_ig])
                        if cont > 0.0:
                            sources.append({
                                'dr': int(dr), 'dc': int(dc),
                                'direction': _DIR_NAMES[i_nb],
                                'p_spread': round(p, 4),
                                'heat_contrib': round(cont, 4),
                                'fraction': 0.0,
                            })
                total_heat = sum(s['heat_contrib'] for s in sources)
                if total_heat > 0:
                    for s in sources:
                        s['fraction'] = round(s['heat_contrib'] / total_heat, 3)
                sources.sort(key=lambda x: -x['fraction'])
                self._ignition_sources[(r_ig, c_ig)] = sources

            self.state[ignited] = 1
            self.burn_timer[ignited] = self._burn_time_steps
            self.ignition_fraction[ignited] = 0.0
            self._ignition_step[ignited] = self._sim_step

        # 4. Monte Carlo spotting
        if burning_mask.any():
            self._apply_spotting(burning_mask, unburned_mask)

        # 5. Containment decay
        if np.any(self._containment_decay > 0.0):
            self.containment_strength = np.clip(
                self.containment_strength - self._containment_decay, 0.0, 1.0
            ).astype(np.float32)
            self.suppression_strength = self.containment_strength

        # 6. Containment humidity boost (applied as a moisture floor)
        if np.any(self.containment_humidity_boost > 0.0):
            self.landscape.moisture = np.clip(
                np.maximum(self.landscape.moisture,
                           self.containment_humidity_boost),
                0.01, 0.35
            ).astype(np.float32)
            # Decay the boost slowly
            self.containment_humidity_boost = np.maximum(
                0.0, self.containment_humidity_boost * 0.998
            ).astype(np.float32)

        # 7. Wetness + water decay / diffusion
        if np.any(self.wetness > 0.0):
            self.wetness *= 0.996
            self.wetness[self.wetness < 0.01] = 0.0
            self.wetness[self.blocked_mask] = 1.0

        if np.any(self.water_amount > 0.0):
            # Evaporation: ~15 sim-minute half-life (at dt=0.25 min/step) —
            # dropped water loses its power over time and eventually vanishes.
            # Deliberately NO neighbour diffusion and NO permanent moisture
            # write-back any more: both made water act far outside the brush
            # footprint and left an invisible, never-expiring suppression halo.
            self.water_amount *= 0.988
            self.water_amount[self.water_amount < 0.02] = 0.0

        self._sim_step += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Probabilistic water extinction
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_water_extinction(self, burning_mask: np.ndarray) -> None:
        if not burning_mask.any():
            return
        # Water can only put out cells it actually covers. Without this gate
        # the ground-moisture term of the sigmoid gave EVERY burning cell on
        # the map a small random chance to die whenever any droplet existed
        # anywhere — fires visibly went out far outside the wet footprint.
        burning_mask = burning_mask & (self.water_amount > 0.05)
        if not burning_mask.any():
            return
        # Restrict all math to the bounding box of the burning cells. This
        # runs once per water droplet while painting — full-grid (800×800)
        # arithmetic + RNG here used to freeze the simulation mid-stroke.
        rr = np.flatnonzero(burning_mask.any(axis=1))
        cc = np.flatnonzero(burning_mask.any(axis=0))
        sl = (slice(int(rr[0]), int(rr[-1]) + 1),
              slice(int(cc[0]), int(cc[-1]) + 1))

        w0_full = getattr(self, '_cached_w0', None)
        w0 = (w0_full[sl] if w0_full is not None
              else np.zeros(burning_mask[sl].shape, dtype=np.float32))

        water_effect  = np.minimum(self.water_amount[sl] * 4.0, 4.0)
        fuel_resist   = w0 * 16.0
        moisture_help = self.landscape.moisture[sl] * 3.0
        wind_resist   = np.hypot(self._wind_u_grid[sl], self._wind_v_grid[sl]) * 0.35

        raw   = water_effect - fuel_resist + moisture_help - wind_resist - 2.0
        p_ext = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)

        rnd        = np.random.random(p_ext.shape).astype(np.float32)
        extinguish = burning_mask[sl] & (rnd < p_ext)
        if extinguish.any():
            # Slices are views — writing through them updates the full grids.
            # Doused cells stay scorched (state 2 → the dark "burned" overlay);
            # only the orange "burning" indication is removed.
            self.state[sl][extinguish] = 2
            self.burn_timer[sl][extinguish] = 0
            self.ignition_fraction[sl][extinguish] = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Cell explanation (used by Alt+click / rect analysis)
    # ──────────────────────────────────────────────────────────────────────────
    def get_cell_explanation(self, r: int, c: int) -> dict:
        """Return a physics-based explanation dict for cell (r, c)."""
        rows, cols = self.state.shape
        if not (0 <= r < rows and 0 <= c < cols):
            return {"error": "Cell out of bounds", "row": r, "col": c}

        state_val = int(self.state[r, c])
        fuel_idx  = int(self.landscape.fuel_map[r, c])
        fuel_name = (self.landscape.fuel_names[fuel_idx]
                     if fuel_idx < len(self.landscape.fuel_names) else "Unknown")
        fuel      = self.landscape.get_fuel_at(r, c)

        wu = float(self._wind_u_grid[r, c]) if hasattr(self, '_wind_u_grid') else 0.0
        wv = float(self._wind_v_grid[r, c]) if hasattr(self, '_wind_v_grid') else 0.0
        wind_speed      = float(np.hypot(wu, wv))
        wind_dir_toward = float(np.degrees(np.arctan2(wu, wv))) % 360.0

        elev   = self.landscape.elevation
        cell_m = float(getattr(self.config, 'CELL_SIZE_METERS', 5.0))
        dz_dy  = (elev[min(r+1, rows-1), c] - elev[max(r-1, 0), c]) / (2.0 * cell_m)
        dz_dx  = (elev[r, min(c+1, cols-1)] - elev[r, max(c-1, 0)]) / (2.0 * cell_m)
        slope_deg = float(np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))))

        mf = float(self.landscape.moisture[r, c])
        mx = float(fuel.get('moisture_ext', 0.25))
        moisture_ratio = mf / max(mx, 0.01)

        # Incoming p_spread per direction
        incoming = []
        for i, (dr, dc) in enumerate(self._neighbors):
            nr, nc_idx = r - dr, c - dc
            if not (0 <= nr < rows and 0 <= nc_idx < cols):
                continue
            p  = float(self.p_spread[i, r, c])
            nb = int(self.state[nr, nc_idx])
            incoming.append({
                "direction":      _DIR_NAMES[i],
                "p_spread":       round(p, 4),
                "neighbor_state": ["unburned", "burning", "burned"][min(nb, 2)],
                "active_heat":    round(p, 4) if nb == 1 else 0.0,
            })
        incoming.sort(key=lambda x: -x["active_heat"])

        # Ignition source attribution (if available)
        ignition_sources = self._ignition_sources.get((r, c), [])

        # ── Numerical ignition probability breakdown ──────────────────────────
        max_p = float(self.p_spread[:, r, c].max()) if hasattr(self, 'p_spread') else 0.0
        active_heat_from_neighbors = 0.0
        for i_nb, (dr_nb, dc_nb) in enumerate(self._neighbors):
            nr_nb, nc_nb = r - dr_nb, c - dc_nb
            if 0 <= nr_nb < rows and 0 <= nc_nb < cols and self.state[nr_nb, nc_nb] == 1:
                active_heat_from_neighbors += float(self.p_spread[i_nb, r, c])

        # Estimate Rothermel wind factor phi_w for this cell
        phi_w_contrib = 0.0
        if fuel and max_p > 0:
            sigma_f = max(float(fuel.get('surface_ratio', 2000.0)), 1e-9)
            rho_b_f = max(float(fuel.get('bulk_density', 0.1)), 1e-9)
            beta_f  = rho_b_f / _PARTICLE_DENSITY
            beta_op_f = 3.348 * (sigma_f ** -0.8189)
            rel_b   = max(beta_f / max(beta_op_f, 1e-9), 1e-9)
            c_w_f   = 7.47 * float(np.exp(-0.133 * sigma_f**0.55))
            b_w_f   = 0.0252 * sigma_f**0.54
            e_w_f   = 0.715  * float(np.exp(-3.59e-4 * sigma_f))
            u_ft_min = min(wind_speed * _MS_TO_FTMIN, 1500.0)
            phi_w_f  = c_w_f * (max(u_ft_min, 0.0) ** b_w_f) * (rel_b ** -e_w_f)
            phi_w_contrib = (phi_w_f / max(1.0 + phi_w_f, 1e-9)) * max_p

        fuel_base_contrib = max_p - phi_w_contrib
        # Slope contribution: estimate from elevation gradient vs neighbours
        elev_r = float(self.landscape.elevation[min(r+1, rows-1), c] - self.landscape.elevation[max(r-1, 0), c])
        elev_c = float(self.landscape.elevation[r, min(c+1, cols-1)] - self.landscape.elevation[r, max(c-1, 0)])
        slope_mag = float(np.hypot(elev_r, elev_c)) / (2.0 * max(cell_m, 1.0))
        phi_s_est = min(slope_mag * 0.5 * max_p, phi_w_contrib * 0.4)  # capped estimate
        phi_w_contrib = max(0.0, phi_w_contrib - phi_s_est)

        # Moisture damping η_m
        rm_f = min(mf / max(mx, 0.01), 1.0)
        eta_m = max(0.0, 1.0 - 2.59*rm_f + 5.11*rm_f**2 - 3.52*rm_f**3)
        moisture_penalty = -(1.0 - eta_m) * max_p * 0.6

        wetness_val = float(self.wetness[r, c])
        water_val   = float(self.water_amount[r, c])
        water_penalty = -(wetness_val * 0.5 + water_val * 0.08) * max_p

        cont_strength = float(self.containment_strength[r, c])
        containment_penalty = -cont_strength * max_p

        ignition_breakdown = {
            "wind_contribution":    round(phi_w_contrib,    3),
            "slope_contribution":   round(phi_s_est,        3),
            "fuel_base":            round(fuel_base_contrib,3),
            "neighbor_heat_active": round(active_heat_from_neighbors, 3),
            "humidity_penalty":     round(moisture_penalty, 3),
            "water_penalty":        round(water_penalty,    3),
            "containment_penalty":  round(containment_penalty, 3),
            "max_spread_rate_pstep":round(max_p,            4),
            "ignition_progress":    round(float(self.ignition_fraction[r, c]), 3),
            "threshold":            round(float(self._ignition_threshold),     2),
            "eta_m":                round(eta_m, 3),
        }

        result: dict = {
            "row": r, "col": c,
            "state": ["unburned", "burning", "burned"][min(state_val, 2)],
            "fuel": {
                "name":         fuel_name,
                "load_lb_ft2":  round(fuel.get('fuel_load', 0.0), 4),
                "heat_content": int(fuel.get('heat_content', 0)),
                "moisture_ext": round(mx, 3),
            },
            "environment": {
                "moisture_fraction": round(mf, 3),
                "moisture_ratio":    round(moisture_ratio, 3),
                "wind_speed_ms":     round(wind_speed, 2),
                "wind_direction_deg":round(wind_dir_toward, 1),
                "slope_degrees":     round(slope_deg, 1),
                "elevation_m":       round(float(elev[r, c]), 1),
            },
            "containment": {
                "wetness":                  round(float(self.wetness[r, c]), 3),
                "water_amount":             round(float(self.water_amount[r, c]), 3),
                "containment_strength":     round(float(self.containment_strength[r, c]), 3),
                "containment_humidity_boost": round(float(self.containment_humidity_boost[r, c]), 3),
                "is_blocked":               bool(self.blocked_mask[r, c]),
                # compat alias
                "suppression_strength":     round(float(self.containment_strength[r, c]), 3),
            },
            "fire_state": {
                "ignition_fraction": round(float(self.ignition_fraction[r, c]), 4),
                "burn_timer":        int(self.burn_timer[r, c]),
                "burn_timer_max":    int(self._burn_time_steps),
                "ignition_step":     int(self._ignition_step[r, c]),
                "sim_step_now":      int(self._sim_step),
            },
            "incoming_spread":    incoming[:6],
            "ignition_sources":   ignition_sources[:4],   # top-4 contributors
            "ignition_breakdown": ignition_breakdown,
            "factors": [],
        }

        factors: list[str] = []

        if state_val == 0:  # unburned
            active_heat = sum(d["active_heat"] for d in incoming)
            if active_heat > 0:
                top = incoming[0]
                pct = result["fire_state"]["ignition_fraction"]
                factors.append(
                    f"Receiving heat from {top['direction']} at rate {top['p_spread']:.3f}/step. "
                    f"Ignition progress: {pct:.1%} (needs 100%)."
                )
            else:
                factors.append("No active fire in neighbouring cells — cell is not threatened.")

            if moisture_ratio > 0.75:
                factors.append(
                    f"Fuel moisture ({mf:.1%}) is {moisture_ratio:.0%} of extinction threshold "
                    f"({mx:.1%}). Spread probability substantially reduced."
                )
            w = float(self.wetness[r, c])
            if w > 0.15:
                factors.append(
                    f"Wetness ({w:.0%}) dampening incoming heat — Rothermel wetness factor applied."
                )
            cs = float(self.containment_strength[r, c])
            if cs > 0.05:
                factors.append(
                    f"Containment line at {cs:.0%} strength actively protecting this cell."
                )
            if self.blocked_mask[r, c]:
                factors.append("Hard barrier (firebreak / non-combustible) — fire cannot ignite here.")

        elif state_val == 1:  # burning
            fl  = fuel.get('fuel_load', 0.0)
            bpct = int(self.burn_timer[r, c]) / max(self._burn_time_steps, 1)
            factors.append(
                f"Burning — {fuel_name.replace('_', ' ')} "
                f"(load: {fl:.3f} lb/ft²). Burn timer: {bpct:.0%} remaining."
            )
            if wind_speed > 0.3:
                bearing = _compass_bearing(wind_dir_toward)
                factors.append(
                    f"Wind {wind_speed:.1f} m/s toward {bearing} amplifies spread "
                    f"via Rothermel φ_w (wind factor)."
                )
            if slope_deg > 4.0:
                factors.append(
                    f"Upslope {slope_deg:.0f}° accelerates fire via φ_s (slope factor)."
                )
            wa = float(self.water_amount[r, c])
            if wa > 0.2:
                p_ext_approx = 1.0 / (1.0 + np.exp(-(wa*4 - fl*16 + mf*3 - wind_speed*0.35 - 2.0)))
                factors.append(
                    f"Water accumulation ({wa:.2f} units) — "
                    f"approx. {p_ext_approx:.0%} extinction probability per step."
                )
            if moisture_ratio > 0.65:
                factors.append(
                    f"Fuel moisture {mf:.1%} at {moisture_ratio:.0%} of extinction threshold — "
                    f"η_m damping factor reducing reaction intensity."
                )
            # Ignition attribution
            if ignition_sources:
                top_src = ignition_sources[0]
                factors.append(
                    f"Primary ignition source: {top_src['direction']} neighbour "
                    f"({top_src['fraction']:.0%} of heat at ignition step {self._ignition_step[r,c]})."
                )

        else:  # burned
            factors.append("Cell fully consumed — fuel exhausted. Charred scar.")
            if ignition_sources:
                top_src = ignition_sources[0]
                factors.append(
                    f"Was ignited primarily from {top_src['direction']} ({top_src['fraction']:.0%})."
                )
            wa = float(self.water_amount[r, c])
            if wa > 0.1:
                factors.append(f"Residual water present ({wa:.2f} units).")

        result["factors"] = factors
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Regional statistics (used by rect analysis)
    # ──────────────────────────────────────────────────────────────────────────
    def get_region_stats(self, r0: int, r1: int, c0: int, c1: int) -> dict:
        """Return aggregate statistics for the rectangular region [r0:r1, c0:c1]."""
        r0 = max(0, r0);  r1 = min(self.state.shape[0] - 1, r1)
        c0 = max(0, c0);  c1 = min(self.state.shape[1] - 1, c1)

        region_state = self.state[r0:r1+1, c0:c1+1]
        total  = region_state.size
        burned  = int((region_state == 2).sum())
        burning = int((region_state == 1).sum())

        region_ps = self.p_spread[:, r0:r1+1, c0:c1+1]
        avg_ros   = float(region_ps.max(axis=0).mean()) if total > 0 else 0.0

        wu_r = self._wind_u_grid[r0:r1+1, c0:c1+1] if hasattr(self, '_wind_u_grid') else np.zeros((1,1))
        wv_r = self._wind_v_grid[r0:r1+1, c0:c1+1] if hasattr(self, '_wind_v_grid') else np.zeros((1,1))
        wind_spd = float(np.hypot(wu_r, wv_r).mean())
        wind_dir = float(np.degrees(np.arctan2(wu_r.mean(), wv_r.mean())) % 360)

        # Dominant spread direction among burning cells in region
        burning_mask = (region_state == 1)
        if burning_mask.any():
            ps_burning = self.p_spread[:, r0:r1+1, c0:c1+1][:, burning_mask]
            dom_dir_idx = int(ps_burning.mean(axis=1).argmax())
            dominant_spread_dir = _DIR_NAMES[dom_dir_idx]
        else:
            dominant_spread_dir = "—"

        return {
            "total_cells": total,
            "burned_cells": burned,
            "burning_cells": burning,
            "unburned_cells": total - burned - burning,
            "avg_moisture":    round(float(self.landscape.moisture[r0:r1+1, c0:c1+1].mean()), 3),
            "avg_elevation":   round(float(self.landscape.elevation[r0:r1+1, c0:c1+1].mean()), 1),
            "avg_water":       round(float(self.water_amount[r0:r1+1, c0:c1+1].mean()), 3),
            "avg_containment": round(float(self.containment_strength[r0:r1+1, c0:c1+1].mean()), 3),
            "avg_ros_fraction":round(avg_ros, 4),
            "wind_speed_ms":   round(wind_spd, 2),
            "wind_dir_deg":    round(wind_dir, 1),
            "dominant_spread_dir": dominant_spread_dir,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Monte Carlo spotting
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_spotting(self, burning_mask: np.ndarray,
                        unburned_mask: np.ndarray) -> None:
        """Vectorised Monte Carlo firebrand spotting (Andrews/Albini-style)."""
        if not hasattr(self, "_wind_u_grid"):
            return

        cell_m = self.config.CELL_SIZE_METERS
        rows, cols = burning_mask.shape

        br, bc = np.where(burning_mask)
        if br.size == 0:
            return

        wu = self._wind_u_grid[br, bc]
        wv = self._wind_v_grid[br, bc]
        wind_spd = np.hypot(wu, wv).clip(0.1)

        p_loft = np.clip(0.003 * wind_spd * float(getattr(self, '_spotting_rate_mult', 1.0)), 0.0, 0.20)
        lofted  = np.random.random(br.size) < p_loft

        if not lofted.any():
            return

        br  = br[lofted];  bc  = bc[lofted]
        wu  = wu[lofted];  wv  = wv[lofted]
        wind_spd = wind_spd[lofted]

        mean_dist_m = np.maximum(200.0, 150.0 * wind_spd)
        sigma_ln    = 0.45
        mu_ln       = np.log(mean_dist_m) - 0.5 * sigma_ln**2
        dist_m      = np.random.lognormal(mu_ln, sigma_ln)
        dist_cells  = dist_m / cell_m

        base_angle  = np.arctan2(wv, wu)
        scatter     = np.random.uniform(-np.pi / 9, np.pi / 9, br.size)
        angle       = base_angle + scatter

        dc_land = np.round(dist_cells * np.cos(angle)).astype(int)
        dr_land = np.round(dist_cells * np.sin(angle)).astype(int)

        land_c = bc + dc_land
        land_r = br - dr_land

        in_bounds = ((land_r >= 0) & (land_r < rows) &
                     (land_c >= 0) & (land_c < cols))
        br = br[in_bounds]
        bc = bc[in_bounds]
        land_r = land_r[in_bounds]
        land_c = land_c[in_bounds]

        combustible = unburned_mask[land_r, land_c]
        br = br[combustible]
        bc = bc[combustible]
        land_r = land_r[combustible]
        land_c = land_c[combustible]

        if self.blocked_mask.any() and land_r.size:
            not_crossing_firebreak = np.array([
                not _line_hits_mask(self.blocked_mask, int(sr), int(sc), int(lr), int(lc))
                for sr, sc, lr, lc in zip(br, bc, land_r, land_c)
            ], dtype=bool)
            land_r = land_r[not_crossing_firebreak]
            land_c = land_c[not_crossing_firebreak]

        if land_r.size == 0:
            return

        # Firefighter crews on a containment corridor catch embers that land
        # inside it — damp the deposit by the local containment strength.
        # Brands landing beyond the corridor are unaffected (ember transport
        # is how real fires defeat containment lines).
        deposit = np.full(land_r.shape, 0.5, dtype=np.float32)
        if np.any(self.containment_strength > 0.0):
            deposit *= 1.0 - self.containment_strength[land_r, land_c]

        np.add.at(self.ignition_fraction, (land_r, land_c), deposit)

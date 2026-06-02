"""
core/microclimate.py
====================
Spatial microclimate learning for Project WILSON.

Accumulates fire behaviour observations across simulation runs and produces
per-cell correction factors that improve future predictions.  Persists in
memory across WebSocket connections; can be serialised to JSON for disk caching.

Design
------
Each time a simulation finishes, ``record_simulation()`` updates rolling
averages for:
  - ignition frequency   — how often each cell has burned historically
  - ROS factor           — empirical speed-up relative to Rothermel baseline
  - wind factor          — local wind amplification (orographic, channelling)
  - suppression effect   — how effectively containment worked in this area

``apply_to_model()`` optionally boosts p_spread in historically high-ignition
corridors and adjusts the wind field before the simulation begins.

Performance notes
-----------------
All arrays are (rows, cols) float32.  For an 800×800 grid that is ~2.4 MB per
array; with 6 arrays the total is ~15 MB — negligible.
"""

from __future__ import annotations

import json
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.fire_model import CellularAutomataFire
    from core.landscape import Landscape


class MicroclimateLearner:
    """
    Accumulates spatially distributed fire behaviour observations.

    Usage
    -----
    # Once per server session (module-level singleton)
    mc = MicroclimateLearner(800, 800)

    # After each simulation completes
    mc.record_simulation(sim, land)

    # Before each new simulation starts
    mc.apply_to_model(sim)

    # For rect-selection analysis
    stats = mc.get_region_stats(r0, r1, c0, c1)
    """

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._runs = 0

        # Correction factors (multiplicative; 1.0 = no correction)
        self.ros_factor  = np.ones((rows, cols), dtype=np.float32)
        self.wind_factor = np.ones((rows, cols), dtype=np.float32)

        # Observational statistics [0..1 range after normalisation]
        self.ignition_freq     = np.zeros((rows, cols), dtype=np.float32)
        self.burn_speed_norm   = np.zeros((rows, cols), dtype=np.float32)
        self.containment_effect= np.zeros((rows, cols), dtype=np.float32)

        # Weighted counts for Welford-style rolling updates
        self._obs_weight = np.zeros((rows, cols), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_simulation(self, sim: "CellularAutomataFire",
                          land: "Landscape") -> None:
        """
        Update microclimate knowledge from a completed simulation run.

        Uses exponential moving average with alpha = 1/(runs+1) so early
        observations have lower weight than later, more data-rich ones.
        """
        self._runs += 1
        alpha = 1.0 / (self._runs + 1)          # decaying weight for new obs
        one_minus = 1.0 - alpha

        # ── ignition frequency ──────────────────────────────────────────
        burned_mask = ((sim.state == 2) | (sim.state == 1)).astype(np.float32)
        self.ignition_freq = one_minus * self.ignition_freq + alpha * burned_mask

        # ── wind factor: local wind speed / domain-mean ─────────────────
        if hasattr(sim, '_wind_u_grid') and hasattr(sim, '_wind_v_grid'):
            wind_spd = np.hypot(sim._wind_u_grid, sim._wind_v_grid).astype(np.float32)
            mean_spd = float(wind_spd.mean())
            if mean_spd > 0.01:
                local_wf = (wind_spd / mean_spd).astype(np.float32)
                self.wind_factor = np.clip(
                    one_minus * self.wind_factor + alpha * local_wf,
                    0.2, 5.0,
                )

        # ── ROS factor: compare actual burn speed to p_spread baseline ──
        # Proxy: cells that burned quickly (high p_spread neighbourhood) vs
        # cells that took many steps (sparse heat accumulation).
        if hasattr(sim, 'p_spread') and burned_mask.any():
            max_ps  = sim.p_spread.max(axis=0)          # peak p_spread per cell
            # Normalise to [0, 1] within domain
            ps_norm = np.clip(max_ps / (max_ps.mean() + 1e-9), 0.0, 3.0).astype(np.float32)
            # Only update where we have observations
            update_mask = burned_mask > 0
            new_ros  = np.where(update_mask, ps_norm, self.ros_factor)
            self.ros_factor = np.clip(
                one_minus * self.ros_factor + alpha * new_ros,
                0.3, 4.0,
            )

        # ── containment effectiveness ───────────────────────────────────
        # High containment strength that prevented burning → effective.
        if hasattr(sim, 'containment_strength'):
            cs = sim.containment_strength.astype(np.float32)
            protected_unburned = cs * (1.0 - burned_mask)   # protected AND unburned
            self.containment_effect = np.clip(
                one_minus * self.containment_effect + alpha * protected_unburned,
                0.0, 1.0,
            )

    def apply_to_model(self, sim: "CellularAutomataFire") -> None:
        """
        Apply learned corrections to a freshly built fire model.
        Call after ``_precompute_ros_grid()`` but before ignition.
        """
        if self._runs < 2:
            return   # need at least two runs for meaningful corrections

        # Boost p_spread in historically high-ignition corridors (≥30% frequency)
        high_freq = self.ignition_freq > 0.30
        if high_freq.any():
            boost = np.where(high_freq, 1.0 + 0.15 * self.ignition_freq, 1.0)
            sim.p_spread *= boost.astype(np.float32)
            sim.p_spread = np.clip(sim.p_spread, 0.0, 1.0)

        # Adjust wind field using learned local factors
        if np.any(self.wind_factor != 1.0) and hasattr(sim, '_wind_u_grid'):
            # Apply gently — cap at ±30% deviation from baseline
            wf = np.clip(self.wind_factor, 0.7, 1.3)
            sim._wind_u_grid = (sim._wind_u_grid * wf).astype(np.float32)
            sim._wind_v_grid = (sim._wind_v_grid * wf).astype(np.float32)

    def get_region_stats(self, r0: int, r1: int, c0: int, c1: int) -> dict:
        """Return microclimate statistics for a rectangular grid region."""
        r0 = max(0, r0);  r1 = min(self.rows - 1, r1)
        c0 = max(0, c0);  c1 = min(self.cols - 1, c1)

        if self._runs == 0:
            return {
                "runs": 0,
                "learned": False,
                "avg_ignition_freq": 0.0,
                "max_ignition_freq": 0.0,
                "avg_ros_factor": 1.0,
                "avg_wind_factor": 1.0,
                "avg_containment_effect": 0.0,
            }

        freq = self.ignition_freq[r0:r1+1, c0:c1+1]
        ros  = self.ros_factor[r0:r1+1, c0:c1+1]
        wf   = self.wind_factor[r0:r1+1, c0:c1+1]
        ce   = self.containment_effect[r0:r1+1, c0:c1+1]

        return {
            "runs":                    self._runs,
            "learned":                 self._runs >= 2,
            "avg_ignition_freq":       round(float(freq.mean()), 3),
            "max_ignition_freq":       round(float(freq.max()), 3),
            "avg_ros_factor":          round(float(ros.mean()), 3),
            "avg_wind_factor":         round(float(wf.mean()), 3),
            "avg_containment_effect":  round(float(ce.mean()), 3),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialise to compact JSON (arrays stored as nested lists)."""
        return json.dumps({
            "runs": self._runs,
            "rows": self.rows,
            "cols": self.cols,
            "ignition_freq":     self.ignition_freq.tolist(),
            "ros_factor":        self.ros_factor.tolist(),
            "wind_factor":       self.wind_factor.tolist(),
            "containment_effect":self.containment_effect.tolist(),
        })

    @classmethod
    def from_json(cls, data: str | dict) -> "MicroclimateLearner":
        """Deserialise from JSON string or pre-parsed dict."""
        if isinstance(data, str):
            data = json.loads(data)
        obj = cls(int(data["rows"]), int(data["cols"]))
        obj._runs             = int(data.get("runs", 0))
        obj.ignition_freq     = np.array(data.get("ignition_freq",     obj.ignition_freq.tolist()), dtype=np.float32)
        obj.ros_factor        = np.array(data.get("ros_factor",        obj.ros_factor.tolist()),    dtype=np.float32)
        obj.wind_factor       = np.array(data.get("wind_factor",       obj.wind_factor.tolist()),   dtype=np.float32)
        obj.containment_effect= np.array(data.get("containment_effect",obj.containment_effect.tolist()), dtype=np.float32)
        return obj

    def resize_to(self, rows: int, cols: int) -> "MicroclimateLearner":
        """
        Return a new learner resized to (rows, cols).
        Used when a new simulation has a different grid than the stored one.
        Resampling is nearest-neighbour (fast, acceptable for smooth fields).
        """
        if rows == self.rows and cols == self.cols:
            return self

        from scipy.ndimage import zoom
        zy = rows / self.rows
        zx = cols / self.cols

        new = MicroclimateLearner(rows, cols)
        new._runs = self._runs
        new.ignition_freq      = np.clip(zoom(self.ignition_freq,      (zy, zx), order=1), 0.0, 1.0).astype(np.float32)
        new.ros_factor         = np.clip(zoom(self.ros_factor,         (zy, zx), order=1), 0.3, 4.0).astype(np.float32)
        new.wind_factor        = np.clip(zoom(self.wind_factor,        (zy, zx), order=1), 0.2, 5.0).astype(np.float32)
        new.containment_effect = np.clip(zoom(self.containment_effect, (zy, zx), order=1), 0.0, 1.0).astype(np.float32)
        return new

"""
optimization.py  —  Synthetic-terrain Differential Evolution optimizer.

This is the original offline optimizer that works entirely with synthetic
terrain and no external APIs.  It was used for development and testing
before real terrain (COP30 + CORINE) and real FIRMS data were integrated.

For the full real-data pipeline use hindcast_optimizer.py instead.

Usage
-----
    python optimization.py
    python optimization.py --maxiter 30 --popsize 15 --plot
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution

# Ensure project root is importable when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.landscape import Landscape
from core.fire_model import CellularAutomataFire


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic landscape builder (same as simulation.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_synthetic_landscape(config) -> Landscape:
    rows, cols = config.GRID_SIZE
    land = Landscape(config)

    cy, cx = rows // 2, cols // 2
    ys, xs = np.mgrid[0:rows, 0:cols]
    elev    = 200.0 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * (rows * 0.2)**2))
    land.elevation = elev.astype(np.float32)
    land.fuel_map  = np.where((ys + xs) % 4 < 2, 0, 3).astype(np.int8)
    land.moisture  = np.full((rows, cols), 0.06, dtype=np.float32)
    land.set_wind(speed=4.0, direction=225.0)
    return land


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth generator: run with "true" params to produce an observed mask
# ──────────────────────────────────────────────────────────────────────────────

def make_truth_mask(land: Landscape, config,
                    true_moisture: float = 0.05,
                    true_wind_mult: float = 1.2,
                    steps: int = 200) -> np.ndarray:
    """Run the CA with known params and return burned cells as a truth mask."""
    land.moisture = np.full(land.shape, true_moisture, dtype=np.float32)
    land.set_wind(speed=4.0 * true_wind_mult, direction=225.0)

    rows, cols = land.shape
    sim = CellularAutomataFire(land, config)
    sim.ignite(rows // 2, cols // 2)

    for _ in range(steps):
        sim.step()

    return (sim.state >= 1)


# ──────────────────────────────────────────────────────────────────────────────
# Objective function
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticOptimizer:
    """
    Wraps the CA in a SciPy-compatible objective for synthetic-terrain tests.

    Optimises two parameters:
        params[0] : moisture_offset  in [-0.05, +0.05]
        params[1] : wind_multiplier  in [0.5, 2.0]
    """

    def __init__(self, base_landscape: Landscape, truth_mask: np.ndarray,
                 steps: int, ignition_rc: tuple):
        self.land          = base_landscape
        self.base_moisture = base_landscape.moisture.copy()
        self.base_wind     = base_landscape.wind_speed
        self.base_dir      = base_landscape.wind_dir
        self.truth_mask    = truth_mask
        self.steps         = steps
        self.ignition_rc   = ignition_rc

        self._n_evals = 0
        self._best    = float("inf")
        self._t0      = time.time()

    def __call__(self, params: np.ndarray) -> float:
        dm, kw = float(params[0]), float(params[1])

        self.land.moisture = np.clip(self.base_moisture + dm, 0.01, 0.30)
        self.land.set_wind(self.base_wind * kw, self.base_dir)

        rows, cols = self.land.shape
        sim = CellularAutomataFire(self.land, config)
        sim.ignite(*self.ignition_rc)
        for _ in range(self.steps):
            sim.step()

        pred = (sim.state >= 1)
        inter = np.logical_and(pred, self.truth_mask).sum()
        union = np.logical_or( pred, self.truth_mask).sum()
        iou   = float(inter / union) if union > 0 else 0.0
        error = 1.0 - iou

        self._n_evals += 1
        if error < self._best:
            self._best = error
        elapsed = time.time() - self._t0
        print(f"  eval={self._n_evals:3d}  error={error:.4f}  "
              f"best={self._best:.4f}  Δm={dm:+.3f}  k={kw:.2f}  {elapsed:.0f}s",
              flush=True)
        return error


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_optimization(maxiter: int = 20, popsize: int = 10,
                     steps: int = 200, plot: bool = False) -> dict:

    print("[OptimSyn] Building synthetic landscape...")
    land = build_synthetic_landscape(config)

    print("[OptimSyn] Generating ground-truth mask (true params known)...")
    truth = make_truth_mask(land, config,
                            true_moisture=0.05, true_wind_mult=1.2,
                            steps=steps)
    land.moisture = np.full(land.shape, 0.06, dtype=np.float32)
    land.set_wind(speed=4.0, direction=225.0)

    rows, cols = config.GRID_SIZE
    ignition_rc = (rows // 2, cols // 2)

    obj = SyntheticOptimizer(land, truth, steps, ignition_rc)

    print(f"[OptimSyn] Running DE  (maxiter={maxiter}, popsize={popsize})...")
    result = differential_evolution(
        obj,
        bounds  = [(-0.05, 0.05), (0.5, 2.0)],
        maxiter = maxiter,
        popsize = popsize,
        seed    = 42,
        tol     = 1e-4,
        polish  = True,
        disp    = False,
    )

    dm, kw = float(result.x[0]), float(result.x[1])
    print(f"\n[OptimSyn] Best: Δmoisture={dm:+.4f}  k_wind={kw:.3f}  "
          f"error={result.fun:.4f}  IoU={100*(1-result.fun):.1f}%")

    if plot:
        _plot_result(land, truth, obj, dm, kw, steps, ignition_rc)

    return {"moisture_offset": dm, "wind_multiplier": kw, "error": result.fun}


def _plot_result(land, truth, obj, dm, kw, steps, ignition_rc):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    land.moisture = np.clip(obj.base_moisture + dm, 0.01, 0.30)
    land.set_wind(obj.base_wind * kw, obj.base_dir)
    sim = CellularAutomataFire(land, config)
    sim.ignite(*ignition_rc)
    for _ in range(steps):
        sim.step()
    pred = (sim.state >= 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Synthetic Optimizer  Δm={dm:+.3f}  k={kw:.2f}  "
                 f"IoU={100*(1 - obj._best):.1f}%")

    axes[0].imshow(land.elevation, cmap="terrain", origin="lower")
    axes[0].set_title("Elevation")

    ov_t = np.zeros((*land.shape, 4)); ov_t[truth] = [1, 0, 0, 0.6]
    ov_p = np.zeros((*land.shape, 4)); ov_p[pred]  = [0, 0.4, 1, 0.4]
    axes[1].imshow(land.elevation, cmap="gray", origin="lower", alpha=0.4)
    axes[1].imshow(ov_t, origin="lower"); axes[1].imshow(ov_p, origin="lower")
    axes[1].set_title("Truth (red) vs Predicted (blue)")
    axes[1].legend(handles=[
        mpatches.Patch(color=[1,0,0,0.6], label=f"Truth {truth.sum()}"),
        mpatches.Patch(color=[0,0.4,1,0.6], label=f"Predicted {pred.sum()}"),
    ], fontsize=8)

    tp = pred & truth; fp = pred & ~truth; fn = ~pred & truth
    ov = np.zeros((*land.shape, 4))
    ov[tp]=[0,.8,0,.8]; ov[fp]=[0,.3,1,.6]; ov[fn]=[1,0,0,.8]
    axes[2].imshow(land.elevation, cmap="gray", origin="lower", alpha=0.4)
    axes[2].imshow(ov, origin="lower")
    axes[2].set_title("TP/FP/FN")

    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Synthetic-terrain DE optimizer")
    p.add_argument("--maxiter", type=int,   default=20)
    p.add_argument("--popsize", type=int,   default=10)
    p.add_argument("--steps",   type=int,   default=200)
    p.add_argument("--plot",    action="store_true")
    args = p.parse_args()
    run_optimization(args.maxiter, args.popsize, args.steps, args.plot)


if __name__ == "__main__":
    main()

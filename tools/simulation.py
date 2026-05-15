"""
simulation.py  —  Standalone fire simulation runner.

Runs a synthetic-terrain simulation (no real DEM) and optionally
launches the 3-D PyVista visualiser or saves a 2-D animation.

Requires PyVista for 3-D view:  pip install pyvista vtk

Usage
-----
    python simulation.py
    python simulation.py --no-3d       # matplotlib animation only
    python simulation.py --steps 500
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure project root is importable when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.landscape import Landscape
from core.fire_model import CellularAutomataFire


def build_synthetic_landscape(config) -> Landscape:
    """
    Build a Landscape with a synthetic Gaussian hill DEM and
    a simple checkerboard fuel map (pine / dry grass alternating).
    """
    rows, cols = config.GRID_SIZE
    land = Landscape(config)

    # Synthetic hill centred in the grid
    cy, cx = rows // 2, cols // 2
    ys, xs = np.mgrid[0:rows, 0:cols]
    elev    = 200.0 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * (rows * 0.2)**2))
    land.elevation = elev.astype(np.float32)

    # Alternating fuel types: Aleppo_Pine (0) and Dry_Grass (3)
    land.fuel_map  = np.where((ys + xs) % 4 < 2, 0, 3).astype(np.int8)

    # Uniform moisture and wind
    land.moisture  = np.full((rows, cols), 0.06, dtype=np.float32)
    land.set_wind(speed=4.0, direction=225.0)   # SW wind, 4 m/s

    return land


def run_simulation(land: Landscape, config,
                   ignition_rc: tuple = None,
                   steps: int = 300,
                   record_every: int = 5) -> list:
    """
    Run the Rothermel CA and return a list of state snapshots.

    Parameters
    ----------
    land         : Landscape object
    config       : config module
    ignition_rc  : (row, col) ignition cell; defaults to grid centre
    steps        : total time steps
    record_every : capture a snapshot every N steps

    Returns
    -------
    history : list of (rows, cols) int8 arrays
    """
    rows, cols = land.shape
    if ignition_rc is None:
        ignition_rc = (rows // 2, cols // 2)

    sim = CellularAutomataFire(land, config)
    sim.ignite(*ignition_rc)

    history = []
    for step in range(steps):
        sim.step()
        if step % record_every == 0:
            history.append(sim.state.copy())

    burned = int((sim.state >= 1).sum())
    cell_m = config.CELL_SIZE_METERS
    print(f"[Simulation] Done.  {steps} steps, "
          f"{burned} cells burned "
          f"({burned * cell_m**2 / 10_000:.1f} ha equivalent).")
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Standalone synthetic-terrain fire simulation"
    )
    parser.add_argument("--steps",    type=int, default=300,
                        help="Number of CA time steps")
    parser.add_argument("--no-3d",   action="store_true",
                        help="Skip PyVista; use matplotlib animation only")
    parser.add_argument("--save-gif", default="",
                        help="Save matplotlib animation to this .gif path")
    args = parser.parse_args()

    print("[Simulation] Building synthetic landscape...")
    land = build_synthetic_landscape(config)

    print(f"[Simulation] Running {args.steps} steps...")
    history = run_simulation(land, config, steps=args.steps)

    if not args.no_3d:
        try:
            from viz.visualizer_3d import Visualizer3D
            sim_dummy = CellularAutomataFire(land, config)
            print("[Simulation] Launching 3-D visualiser (PyVista)...")
            viz = Visualizer3D.from_landscape(history, land, config)
            viz.play()
        except ImportError as exc:
            print(f"[Simulation] PyVista not available ({exc}), falling back to 2-D.")
            args.no_3d = True

    if args.no_3d:
        from viz.visualizer import animate_fire
        print("[Simulation] Launching 2-D matplotlib animation...")
        ani = animate_fire(land, history)
        if args.save_gif and ani is not None:
            ani.save(args.save_gif, writer="pillow", fps=15)
            print(f"[Simulation] Saved animation to '{args.save_gif}'")


if __name__ == "__main__":
    main()

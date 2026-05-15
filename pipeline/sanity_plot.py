"""
sanity_plot.py
==============

Generates a 4-panel diagnostic figure summarising one simulation run:

    +-----------------+  +---------------------+
    | 1. Elevation    |  | 2. Fuel map (CORINE)|
    +-----------------+  +---------------------+
    | 3. Fire @ early |  | 4. Fire @ end       |
    +-----------------+  +---------------------+

Usage as a module:

    from sanity_plot import save_sanity_plot
    save_sanity_plot(land, history, fire_sim, ignite_rc=(36, 36),
                     out_path="output/sanity_check.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Same fuel→color mapping as visualizer_3d.py for consistency.
_FUEL_COLORS = {
    "Aleppo_Pine":         "#238b45",
    "Phrygana_Low_Scrub":  "#a1d99b",
    "Maquis_Dense_Shrub":  "#74c476",
    "Dry_Grass":           "#d9f0a3",
    "Oak_Forest":          "#00441b",
    "Olive_Grove":         "#addd8e",
    "Non_Combustible":     "#9c9c9c",
}


def _fuel_cmap(fuel_names):
    return ListedColormap([_FUEL_COLORS.get(n, "#888888") for n in fuel_names])


def save_sanity_plot(land, history, fire_sim, ignite_rc, out_path,
                     wind_speed=None, wind_dir=None):
    """
    land       : Landscape with elevation, fuel_map, fuel_names
    history    : list of np.ndarray state snapshots
    fire_sim   : CellularAutomataFire (only used for .dt)
    ignite_rc  : (row, col) ignition point
    out_path   : where to save the PNG
    wind_speed : float (m/s)  – optional, for text annotation
    wind_dir   : float (deg)  – optional, for text annotation + arrow
    """
    if wind_speed is None:
        wind_speed = float(getattr(land, 'wind_speed', 0.0))
    if wind_dir is None:
        wind_dir = float(getattr(land, 'wind_dir', 0.0))

    ir, ic = ignite_rc
    rows, cols = land.shape
    n_fuels = len(land.fuel_names)
    cmap_fuel = _fuel_cmap(land.fuel_names)

    # Pick two evenly-spaced snapshots
    if len(history) >= 4:
        early = history[len(history) // 3]
        late  = history[-1]
        t_early = len(history) // 3
        t_late  = len(history) - 1
    else:
        early = late = history[-1]
        t_early = t_late = len(history) - 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Hellenic Wildfire Digital Twin × WILSON — "
                 "Real-DEM Run Diagnostic", fontsize=14)

    # Panel 1: elevation
    im0 = axes[0, 0].imshow(land.elevation, cmap="terrain", origin="lower")
    axes[0, 0].set_title(
        f"Elevation [m]  ({land.elevation.min():.0f} – "
        f"{land.elevation.max():.0f})"
    )
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.7)
    axes[0, 0].plot(ic, ir, "r*", markersize=18, markeredgecolor="white",
                    label="ignition")
    axes[0, 0].legend(loc="upper right")

    # Panel 2: fuel map
    axes[0, 1].imshow(land.fuel_map, cmap=cmap_fuel, origin="lower",
                      vmin=0, vmax=n_fuels - 1)
    axes[0, 1].set_title("Fuel map (CORINE → GREEK_FUELS)")
    # Legend for fuel types actually present
    present = sorted(set(land.fuel_map.flatten().tolist()))
    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=_FUEL_COLORS.get(land.fuel_names[i], "#888"))
               for i in present]
    labels = [land.fuel_names[i] for i in present]
    axes[0, 1].legend(handles, labels, loc="upper left",
                      fontsize=7, framealpha=0.85)

    # Panels 3 & 4: fire spread
    for ax, t, state in [(axes[1, 0], t_early, early), (axes[1, 1], t_late, late)]:
        ax.imshow(land.fuel_map, cmap=cmap_fuel, origin="lower",
                  vmin=0, vmax=n_fuels - 1, alpha=0.4)
        burning = np.ma.masked_where(state != 1,
                                     np.ones_like(state, dtype=float))
        burned  = np.ma.masked_where(state != 2,
                                     np.ones_like(state, dtype=float))
        ax.imshow(burned,  cmap=ListedColormap(["#2a2a2a"]), origin="lower")
        ax.imshow(burning, cmap=ListedColormap(["#ff4500"]), origin="lower")
        ax.plot(ic, ir, "y*", markersize=12, markeredgecolor="black")
        minutes = t * fire_sim.dt
        active = int(np.sum(state == 1))
        burned_n = int(np.sum(state == 2))
        ax.set_title(f"t = {t} steps ≈ {minutes:.1f} min  "
                     f"(active={active}, burned={burned_n})")

        # Wind arrow + label
        rad = np.radians(90 - wind_dir)
        dx = np.cos(rad) * 6
        dy = np.sin(rad) * 6
        ax.annotate("", xy=(cols - 8, rows - 4),
                    xytext=(cols - 8 - dx, rows - 4 - dy),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.text(cols - 14, rows - 10,
                f"wind\n{wind_speed} m/s\n@ {wind_dir:g}°",
                fontsize=8, color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path

"""
visualizer.py  —  Simple 2-D matplotlib fire animation stub.

Replays a list of CA state snapshots as a looping animation.
Superseded by the full GUI in gui_launcher.py (Hillshade Fire tab),
but kept here as a lightweight standalone script.

Usage
-----
    from visualizer import animate_fire
    animate_fire(landscape, history, interval_ms=150)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_fire(landscape, history: list, interval_ms: int = 150,
                 title: str = "Fire Spread") -> None:
    """
    Animate a list of CA state arrays over the terrain.

    Parameters
    ----------
    landscape   : Landscape object with .elevation attribute
    history     : list of (rows, cols) int8 arrays  (0=unburned, 1=burning, 2=burned)
    interval_ms : milliseconds between frames
    title       : figure title
    """
    if not history:
        print("[Visualizer] history is empty — run the simulation first.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#0a0a0a")
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="#7f849c", labelsize=7)

    # Base terrain layer (drawn once)
    ax.imshow(landscape.elevation, cmap="terrain", origin="lower", alpha=0.55)

    # Fire overlay updated each frame
    fire_rgba = np.zeros((*history[0].shape, 4), dtype=np.float32)
    fire_img  = ax.imshow(fire_rgba, origin="lower", animated=True)

    frame_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        color="white", fontsize=8, va="top")

    def _update(idx):
        snap = history[idx]
        rgba = np.zeros((*snap.shape, 4), dtype=np.float32)
        rgba[snap == 1] = [1.0, 0.3, 0.0, 0.9]
        rgba[snap == 2] = [0.3, 0.1, 0.0, 0.6]
        fire_img.set_data(rgba)
        frame_txt.set_text(f"frame {idx + 1}/{len(history)}")
        return fire_img, frame_txt

    ani = animation.FuncAnimation(
        fig, _update,
        frames=len(history),
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return ani

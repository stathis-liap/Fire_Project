"""
visualizer_3d.py  —  PyVista 3-D terrain-draped fire animation for Project WILSON.

Features
--------
  • Real COP30 terrain mesh (StructuredGrid) with smooth shading
  • Custom fuel-type colormap + fire-state overlay (burning / burned)
  • Wind-vector arrows over terrain
  • Ignition-point marker sphere
  • Animated playback with VTK timer (no Python for-loops over grid points)
  • P = pause/resume,  R = reset,  drag slider = seek to frame
  • Clock + status text overlays

Standalone use (launched by gui_launcher.py "Launch 3D View" button)
---------------------------------------------------------------------
    python visualizer_3d.py viz3d_data.npz

Direct Python use
-----------------
    from visualizer_3d import Visualizer3D
    viz = Visualizer3D.from_landscape(snapshots, landscape, config)
    viz.play()

Wind diagnostic
---------------
    from visualizer_3d import inspect_wind_model
    inspect_wind_model(landscape, wind_u_grid, wind_v_grid)
"""

import sys
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path so this file can be launched as a subprocess
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import pyvista as pv
    from matplotlib.colors import ListedColormap
except ImportError as _e:
    raise ImportError(
        "PyVista not installed.  Run:  pip install pyvista\n"
        f"Original error: {_e}"
    )



# ── Custom colormap: 0-6 fuel types, 7 burning, 8 burned ─────────────────────
_FUEL_COLORS = [
    "#228B22",   # 0 Aleppo Pine
    "#8B4513",   # 1 Phrygana
    "#556B2F",   # 2 Maquis
    "#DAA520",   # 3 Dry Grass
    "#90EE90",   # 4 Oak Forest
    "#D3D3D3",   # 5 Olive Grove
    "#4682B4",   # 6 Non-Combustible
    "#FF3800",   # 7 BURNING
    "#3F1400",   # 8 BURNED
    "#87CEEB",   # 9 sky (padding)
]
_CMAP = ListedColormap(_FUEL_COLORS)

_IDX_BURNING = 1
_IDX_BURNED  = 2


class Visualizer3D:
    """
    3-D terrain-draped fire animation using PyVista.

    All NumPy arrays — no Python for-loops over grid points.

    Controls
    --------
    P  — pause / resume
    R  — reset to frame 0
    Drag slider to seek.

    Typical use (from gui_launcher button)
    --------------------------------------
        viz = Visualizer3D.from_npz("viz3d_data.npz")
        viz.play()

    Direct use
    ----------
        viz = Visualizer3D.from_landscape(snapshots, landscape, config)
        viz.play()
    """

    def __init__(self,
                 snapshots: list,
                 elevation: np.ndarray,
                 fuel_map: np.ndarray,
                 wind_speed: float,
                 wind_dir: float,
                 cell_size_m: float,
                 dt_minutes: float = 1.0,
                 start_time: str = "14:00",
                 fps: int = 8,
                 ignition_rc: tuple = None,
                 title: str = "Project WILSON — Fire Spread",
                 texture_array: np.ndarray = None):

        if not snapshots:
            print("[Visualizer3D] snapshots list is empty — nothing to show.")
            return

        self._snapshots   = snapshots
        self._elevation   = elevation.astype(float)
        self._fuel_map    = fuel_map.astype(np.float32)
        self._wind_speed  = float(wind_speed)
        self._wind_dir    = float(wind_dir)
        self._cell_m      = float(cell_size_m)
        self._dt_min      = float(dt_minutes)
        self._fps         = max(1, int(fps))
        self._ignition_rc = ignition_rc
        self._title       = title
        self._paused      = False
        self._frame_idx   = 0

        # Satellite / CORINE texture: (H, W, 3) uint8, already south-up flipped
        self._texture_array = texture_array  # None or np.ndarray

        # Parse start time "HH:MM" → total minutes
        hh, mm = (int(x) for x in start_time.split(":"))
        self._start_min   = hh * 60 + mm

        rows, cols = elevation.shape
        self._rows  = rows
        self._cols  = cols

        # Build PyVista StructuredGrid from DEM — pure NumPy, no loops
        x = np.arange(cols, dtype=float) * self._cell_m
        y = np.arange(rows, dtype=float) * self._cell_m
        xx, yy = np.meshgrid(x, y)
        self._grid = pv.StructuredGrid(xx, yy, self._elevation)
        self._grid.point_data["state"] = self._fuel_map.ravel()

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_landscape(cls, snapshots: list, landscape, config,
                       fps: int = 8, start_time: str = "14:00"):
        """Create from Landscape + config objects (direct pipeline use)."""
        cell_m = getattr(landscape.config if hasattr(landscape, "config") else config,
                         "CELL_SIZE_METERS", 280.0)
        dt     = getattr(config, "dt", 1.0)
        return cls(
            snapshots   = snapshots,
            elevation   = landscape.elevation,
            fuel_map    = landscape.fuel_map,
            wind_speed  = getattr(landscape, "wind_speed", 0.0),
            wind_dir    = getattr(landscape, "wind_dir", 0.0),
            cell_size_m = cell_m,
            dt_minutes  = dt,
            start_time  = start_time,
            fps         = fps,
        )

    @classmethod
    def from_npz(cls, npz_path: str, fps: int = 8):
        """Create from an npz export produced by gui_launcher._launch_3d_view."""
        d = np.load(npz_path, allow_pickle=False)
        snaps = [d["snapshots"][i] for i in range(d["snapshots"].shape[0])]
        ign   = tuple(d["ignition_rc"].tolist()) if "ignition_rc" in d else None
        # Load satellite texture if present (H, W, 3) uint8, south-up already
        texture = d["satellite_texture"] if "satellite_texture" in d else None
        return cls(
            snapshots    = snaps,
            elevation    = d["elevation"],
            fuel_map     = d["fuel_map"],
            wind_speed   = float(d["wind_speed"]),
            wind_dir     = float(d["wind_dir"]),
            cell_size_m  = float(d["cell_size_m"]),
            dt_minutes   = float(d.get("dt_minutes", np.array([1.0]))),
            start_time   = str(d["start_time"]) if "start_time" in d else "14:00",
            fps          = fps,
            ignition_rc  = ign,
            texture_array = texture,
        )

    # ── Playback ──────────────────────────────────────────────────────────────

    def play(self):
        """Open the PyVista window and start animated playback."""
        if not self._snapshots:
            print("[Visualizer3D] No snapshots to play.")
            return

        plotter = pv.Plotter(title=self._title, window_size=[1280, 800])
        plotter.background_color = "#1a2a3a"   # dark blue sky

        # ── Terrain mesh: satellite texture or fuel colormap ─────────────
        if self._texture_array is not None:
            # Build manual UV texture coordinates (u=East, v=North)
            rows, cols = self._rows, self._cols
            u_coords = np.linspace(0.0, 1.0, cols)
            v_coords = np.linspace(0.0, 1.0, rows)
            uu, vv   = np.meshgrid(u_coords, v_coords)
            tcoords  = np.column_stack([uu.ravel(), vv.ravel()])
            self._grid.active_texture_coordinates = tcoords

            # Resize texture to terrain resolution and create pv.Texture
            try:
                from PIL import Image as _PIL
                _img = _PIL.fromarray(self._texture_array)   # already south-up
                _img = _img.resize((cols, rows), _PIL.LANCZOS)
                tex  = pv.Texture(np.array(_img, dtype=np.uint8))
                plotter.add_mesh(
                    self._grid,
                    texture         = tex,
                    smooth_shading  = True,
                    show_scalar_bar = False,
                )
                print("[Visualizer3D] Satellite texture applied to terrain.")
            except Exception as _te:
                print(f"[Visualizer3D] Texture apply failed ({_te}); falling back to colormap.")
                self._texture_array = None   # trigger fallback below

        if self._texture_array is None:
            plotter.add_mesh(
                self._grid,
                scalars         = "state",
                cmap            = _CMAP,
                clim            = [0, 9],
                smooth_shading  = True,
                show_scalar_bar = False,
            )
            # Scalar bar legend (only shown in colormap mode)
            plotter.add_scalar_bar(
                title      = "Fuel / State",
                n_labels   = 9,
                color      = "white",
                font_family= "courier",
                vertical   = True,
                position_x = 0.88,
                position_y = 0.05,
                width      = 0.08,
                height     = 0.65,
            )

        # Wind arrows (subsampled grid, terrain surface + 50 m offset)
        self._add_wind_arrows(plotter)

        # Ignition marker
        if self._ignition_rc is not None:
            r0, c0 = self._ignition_rc
            z0     = float(self._elevation[r0, c0]) + 120.0
            sphere = pv.Sphere(
                radius = self._cell_m * 0.7,
                center = (c0 * self._cell_m, r0 * self._cell_m, z0)
            )
            plotter.add_mesh(sphere, color="yellow", opacity=0.95)

        # Text overlays
        plotter.add_text(
            "P = pause/resume     R = reset     drag slider = seek",
            position   = "lower_left",
            color      = "white",
            font_size  = 8,
            name       = "hint",
        )
        self._update_status(plotter, "Ready")
        self._update_clock(plotter, 0)

        # Time slider
        n = len(self._snapshots)
        slider = plotter.add_slider_widget(
            callback = lambda value: self._on_slider(plotter, int(round(value))),
            rng      = [0, n - 1],
            value    = 0,
            title    = f"Frame  (0 → {n - 1})",
            style    = "modern",
            fmt      = "%.0f",
            color    = "white",
        )
        self._slider = slider

        # Keyboard events
        plotter.add_key_event("p", lambda: self._toggle_pause(plotter))
        plotter.add_key_event("r", lambda: self._reset(plotter))

        # Open window (non-blocking)
        plotter.show(interactive=False, auto_close=False)

        # VTK repeating timer drives animation
        interval_ms = max(1, int(1000 / self._fps))
        iren = plotter.iren.interactor
        iren.CreateRepeatingTimer(interval_ms)

        def _on_timer(caller, event):
            if self._paused or self._frame_idx >= n:
                return
            self._render_frame(plotter, self._frame_idx)
            self._frame_idx += 1
            try:
                slider.GetRepresentation().SetValue(float(self._frame_idx))
                slider.GetRepresentation().Modified()
            except Exception:
                pass

        iren.AddObserver("TimerEvent", _on_timer)
        iren.Start()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _render_frame(self, plotter, idx: int):
        snap    = self._snapshots[idx]
        display = self._fuel_map.copy()
        display[snap == _IDX_BURNING] = 7.0
        display[snap == _IDX_BURNED]  = 8.0
        self._grid.point_data["state"] = display.ravel()
        self._update_clock(plotter, idx)
        burned_cells = int((snap > 0).sum())
        burned_ha    = burned_cells * self._cell_m ** 2 / 10_000
        self._update_status(
            plotter,
            f"{'PAUSED' if self._paused else 'Playing'}  · "
            f"  {self._frame_to_clock(idx)}  ·  "
            f"frame {idx}/{len(self._snapshots) - 1}  ·  "
            f"{burned_ha:.1f} ha burned"
        )
        plotter.render()

    def _on_slider(self, plotter, idx: int):
        self._frame_idx = max(0, min(idx, len(self._snapshots) - 1))
        self._render_frame(plotter, self._frame_idx)

    def _toggle_pause(self, plotter):
        self._paused = not self._paused
        label = "PAUSED" if self._paused else "Playing"
        self._update_status(plotter, f"{label}  —  P to toggle")

    def _reset(self, plotter):
        self._frame_idx = 0
        self._paused    = False
        self._render_frame(plotter, 0)
        self._update_status(plotter, "Reset — playing from start")

    def _frame_to_clock(self, idx: int) -> str:
        total = self._start_min + int(idx * self._dt_min)
        return f"{(total // 60) % 24:02d}:{total % 60:02d}"

    def _update_clock(self, plotter, idx: int):
        plotter.add_text(
            self._frame_to_clock(idx),
            position    = "upper_right",
            color       = "yellow",
            font        = "courier",
            font_size   = 18,
            name        = "clock",
        )

    def _update_status(self, plotter, msg: str):
        plotter.add_text(
            msg,
            position  = "upper_left",
            color     = "white",
            font_size = 9,
            name      = "status",
        )

    def _add_wind_arrows(self, plotter):
        """Subsampled wind vectors shown as arrows above the terrain surface."""
        rows, cols = self._rows, self._cols
        step = max(1, rows // 15)
        ys_s = np.arange(step // 2, rows, step)
        xs_s = np.arange(step // 2, cols, step)
        yy_s, xx_s = np.meshgrid(ys_s, xs_s, indexing="ij")
        z_s = self._elevation[yy_s, xx_s] + 80.0   # 80 m above surface

        pts = np.column_stack([
            xx_s.ravel() * self._cell_m,
            yy_s.ravel() * self._cell_m,
            z_s.ravel(),
        ])

        # Meteorological convention → Cartesian (East=+x, North=+y)
        angle_rad = math.radians(270.0 - self._wind_dir)
        scale     = self._cell_m * 2.5
        wu = self._wind_speed * math.cos(angle_rad) * scale
        wv = self._wind_speed * math.sin(angle_rad) * scale
        n_pts = len(pts)
        vecs  = np.column_stack([
            np.full(n_pts, wu),
            np.full(n_pts, wv),
            np.zeros(n_pts),
        ])

        plotter.add_arrows(pts, vecs, color="cyan", opacity=0.55,
                           label=f"Wind {self._wind_speed:.1f} m/s @ {self._wind_dir:.0f}°")


# ──────────────────────────────────────────────────────────────────────────────
# Wind-field diagnostic (standalone, uses matplotlib only)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_wind_model(landscape, wind_u_grid: np.ndarray,
                       wind_v_grid: np.ndarray, step: int = 10):
    """
    3-panel matplotlib diagnostic for the terrain-modified midflame wind field.

    Left   — elevation contourf + quiver arrows
    Centre — speed magnitude heatmap
    Right  — midflame/open-air ratio (WAF + Poisson correction)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Terrain-Modified Midflame Wind  "
        f"(base: {landscape.wind_speed:.1f} m/s @ {landscape.wind_dir:.0f}°)",
        fontweight="bold"
    )

    elev       = landscape.elevation
    rows, cols = elev.shape
    ys, xs     = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    speed_mag  = np.hypot(wind_u_grid, wind_v_grid)

    ax = axes[0]
    cf = ax.contourf(elev, cmap="terrain")
    fig.colorbar(cf, ax=ax).set_label("Elevation (m)")
    ax.quiver(xs[::step, ::step], ys[::step, ::step],
              wind_u_grid[::step, ::step], wind_v_grid[::step, ::step],
              speed_mag[::step, ::step], cmap="hot_r",
              scale=None, width=0.003, alpha=0.85)
    ax.set_title("Midflame Wind Vectors over Terrain")
    ax.set_xlabel("Grid column"); ax.set_ylabel("Grid row")

    ax2 = axes[1]
    im  = ax2.imshow(speed_mag, cmap="hot_r", origin="upper")
    fig.colorbar(im, ax=ax2).set_label("Midflame wind speed (m/s)")
    ax2.set_title(f"Speed Magnitude\n({speed_mag.min():.2f}–{speed_mag.max():.2f} m/s)")

    ax3    = axes[2]
    ratio  = speed_mag / (landscape.wind_speed + 1e-9)
    im3    = ax3.imshow(ratio, cmap="RdYlGn", origin="upper", vmin=0, vmax=2)
    fig.colorbar(im3, ax=ax3).set_label("Midflame / Open-air  (WAF + terrain)")
    ax3.set_title("Sheltering & Acceleration\n(green = exposed, red = sheltered)")
    ax3.set_xlabel(
        f"min {ratio.min():.2f}   mean {ratio.mean():.2f}   max {ratio.max():.2f}\n"
        "Grid column"
    )

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess entry point — launched by gui_launcher._launch_3d_view()
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer_3d.py <viz3d_data.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    if not Path(npz_path).exists():
        print(f"[Visualizer3D] File not found: {npz_path}")
        sys.exit(1)

    print(f"[Visualizer3D] Loading {npz_path} ...")
    viz = Visualizer3D.from_npz(npz_path, fps=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    print("[Visualizer3D] Opening 3D window …  P=pause  R=reset  slider=seek")
    viz.play()

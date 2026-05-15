"""
viz/fire_viewer_3d.py
=====================
Standalone PyVista fire animation viewer.

Usage (launched by gui_launcher.py as a subprocess):
    python viz/fire_viewer_3d.py /tmp/fire_data.npz

NPZ payload expected keys:
    elevation      – (R, C) float32 DEM in metres
    snapshots      – (N, R, C) uint8  0=empty 1=burning 2=burned
    ignition_rc    – (2,)    [row, col]
    cell_size_m    – scalar  cell size in metres
    wind_u         – (R, C) float32 or scalar
    wind_v         – (R, C) float32 or scalar
    texture_path   – bytes   path string (optional)

Controls:
    Space / P  – play / pause
    ← →        – step one frame back / forward
    Q / Esc    – quit
"""

import sys
import os
import time
import numpy as np
import pyvista as pv
from PIL import Image

# ── Load payload ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("[3D] Usage: python viz/fire_viewer_3d.py <payload.npz>")
    sys.exit(1)

data = np.load(sys.argv[1], allow_pickle=True)

elevation   = data["elevation"].astype(float)       # (R, C)
snapshots   = data["snapshots"].astype(np.uint8)    # (N, R, C)
ign_rc      = data["ignition_rc"]                   # [row, col]
cell_size_m = float(data["cell_size_m"])
texture_path = data["texture_path"].tobytes().decode() if "texture_path" in data else ""

rows, cols = elevation.shape
N_frames   = len(snapshots)

# Wind (may be 2-D arrays or scalars)
wu = data["wind_u"] if "wind_u" in data else np.zeros_like(elevation)
wv = data["wind_v"] if "wind_v" in data else np.zeros_like(elevation)
if wu.ndim == 0:
    wu = np.full_like(elevation, float(wu))
if wv.ndim == 0:
    wv = np.full_like(elevation, float(wv))

print(f"[3D] Terrain {rows}×{cols}  |  {N_frames} frames  |  cell={cell_size_m:.0f} m")

# ── Build metric coordinate grids ─────────────────────────────────────────────
Z_EXG = 1.5   # vertical exaggeration
xx, yy = np.meshgrid(
    np.arange(cols, dtype=float) * cell_size_m,
    np.arange(rows, dtype=float) * cell_size_m,
)
zz = elevation * Z_EXG

# ── PyVista structured grid ───────────────────────────────────────────────────
mesh = pv.StructuredGrid(xx, yy, zz)

# ── Terrain texture: satellite JPG → vertex RGB ───────────────────────────────
terrain_actor_name = "terrain"
texture_loaded = False
if texture_path and os.path.exists(texture_path):
    try:
        img = Image.open(texture_path).convert("RGB")
        img = img.resize((cols, rows), Image.LANCZOS)
        rgb = np.flipud(np.array(img)).reshape(-1, 3).astype(np.uint8)
        mesh.point_data["RGB"] = rgb
        texture_loaded = True
        print(f"[3D] Satellite texture loaded: {texture_path}")
    except Exception as e:
        print(f"[3D] Texture load failed: {e}")

# ── Fire overlay mesh (slightly above terrain) ────────────────────────────────
fire_mesh = pv.StructuredGrid(xx, yy, zz + 2.0)
fire_colors = np.zeros((fire_mesh.n_points, 4), dtype=np.uint8)
fire_mesh.point_data["fire_rgba"] = fire_colors

# ── Wind quivers ──────────────────────────────────────────────────────────────
q_step = max(1, min(rows, cols) // 20)
qi = np.arange(0, rows, q_step)
qj = np.arange(0, cols, q_step)
qI, qJ = np.meshgrid(qi, qj, indexing="ij")
q_x = (qJ * cell_size_m).ravel()
q_y = (qI * cell_size_m).ravel()
q_z = zz[qI, qJ].ravel() + 80.0
q_u = wu[qI, qJ].ravel()
q_v = wv[qI, qJ].ravel()
vec_pts = np.column_stack((q_x, q_y, q_z))
vec_dir = np.column_stack((q_u, q_v, np.zeros_like(q_u)))
wind_cloud = pv.PolyData(vec_pts)
wind_cloud["wind"] = vec_dir
arrows = wind_cloud.glyph(orient="wind", factor=cell_size_m * 3, scale=True)

# ── Ignition star ─────────────────────────────────────────────────────────────
ig_r, ig_c = int(ign_rc[0]), int(ign_rc[1])
ig_x = ig_c * cell_size_m
ig_y = ig_r * cell_size_m
ig_z = zz[ig_r, ig_c] + 80.0
star_pt = pv.PolyData(np.array([[ig_x, ig_y, ig_z]]))

# ── Plotter setup ─────────────────────────────────────────────────────────────
pl = pv.Plotter(title="Project WILSON — 3D Fire Terrain", window_size=[1400, 900])
pl.set_background("#0d0d1a")

if texture_loaded:
    pl.add_mesh(mesh, scalars="RGB", rgb=True, lighting=True, name=terrain_actor_name,
                smooth_shading=True)
else:
    pl.add_mesh(mesh, cmap="terrain", show_scalar_bar=False, lighting=True,
                name=terrain_actor_name, smooth_shading=True)

pl.add_mesh(fire_mesh, scalars="fire_rgba", rgba=True, show_scalar_bar=False,
            name="fire")
pl.add_mesh(arrows, color="cyan", opacity=0.7, name="wind")
pl.add_mesh(star_pt, color="yellow", point_size=18, render_points_as_spheres=True,
            name="ignition")
pl.add_text("★ Ignition", position=(ig_x, ig_y, ig_z + 120), font_size=10,
            color="yellow", name="ign_label")

# ── State ─────────────────────────────────────────────────────────────────────
state = {"frame": 0, "paused": False, "last_t": time.time(), "fps": 6}

def _render_frame(idx):
    snap = snapshots[idx].ravel()
    c = np.zeros((len(snap), 4), dtype=np.uint8)
    c[snap == 1] = [255, 50, 0, 230]    # burning — red-orange
    c[snap == 2] = [30, 20, 10, 200]    # burned  — near-black char
    fire_mesh.point_data["fire_rgba"] = c
    burned_ha = int((snap == 2).sum()) * cell_size_m**2 / 10_000
    active_ha  = int((snap == 1).sum()) * cell_size_m**2 / 10_000
    pl.add_text(
        f"Frame {idx+1}/{N_frames}  |  "
        f"🔥 {active_ha:.1f} ha active  |  🖤 {burned_ha:.1f} ha burned  "
        f"|  [Space]=pause  [←→]=step  [Q]=quit",
        position="lower_edge", font_size=11, color="white", name="status"
    )

def _toggle_pause():
    state["paused"] = not state["paused"]

def _step_back():
    state["frame"] = max(0, state["frame"] - 1)
    _render_frame(state["frame"])
    state["paused"] = True

def _step_fwd():
    state["frame"] = min(N_frames - 1, state["frame"] + 1)
    _render_frame(state["frame"])
    state["paused"] = True

pl.add_key_event("space", _toggle_pause)
pl.add_key_event("p",     _toggle_pause)
pl.add_key_event("Left",  _step_back)
pl.add_key_event("Right", _step_fwd)

# Camera: isometric view looking from SW
pl.camera_position = "iso"
pl.camera.elevation = 35
pl.camera.azimuth   = -45

pl.show(interactive_update=True, auto_close=False)

# ── Animation loop ────────────────────────────────────────────────────────────
_render_frame(0)
while pl.iren.initialized:
    now = time.time()
    if not state["paused"] and (now - state["last_t"]) >= (1.0 / state["fps"]):
        _render_frame(state["frame"])
        state["frame"] = (state["frame"] + 1) % N_frames
        state["last_t"] = now
    pl.update()

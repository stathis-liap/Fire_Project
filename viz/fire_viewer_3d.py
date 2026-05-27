"""
viz/fire_viewer_3d.py
=====================
Standalone PyVista fire animation viewer.

Usage (launched by gui_launcher.py as a subprocess):
    python viz/fire_viewer_3d.py /tmp/fire_data.npz

NPZ payload expected keys:
    elevation      – (R, C) float32 DEM in metres
    snapshots      – (N, R, C) uint8  0=unburned 1=burning 2=burned
    ignition_rc    – (2,)    [row, col]  primary seed (backwards compat)
    ignition_rcs   – (K, 2)  int array  all ignition seeds (optional)
    cell_size_m    – scalar  cell size in metres
    wind_u         – (R, C) float32 or scalar
    wind_v         – (R, C) float32 or scalar
    texture_path   – bytes   path string (optional)
    fuel_map       – (R, C) int16   fuel-type index per cell (optional)
    fuel_names     – object array   fuel-type name strings (optional)

Visual layers (bottom → top):
    1. Terrain mesh  — satellite texture or elevation colormap
    2. Fuel overlay  — semi-transparent per-fuel-type colour mesh (static)
    3. Fire mesh     — burn-age coloured RGBA overlay (animated)
    4. Wind arrows   — cyan glyphs sampled at coarse resolution
    5. Ignition star — yellow sphere at ignition point(s)

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

# Import canonical fuel colours from fuels.py (single source of truth).
# fire_viewer_3d.py is launched as a subprocess so we add the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.fuels import fuel_color_rgba as _fuel_color_rgba, FUEL_DISPLAY_COLOR_DEFAULT

# Keep local alias for inline use
_FUEL_RGBA         = {}   # populated lazily from fuels.py via _fuel_color_rgba()
_DEFAULT_FUEL_RGBA = FUEL_DISPLAY_COLOR_DEFAULT

# ── Load payload ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("[3D] Usage: python viz/fire_viewer_3d.py <payload.npz>")
    sys.exit(1)

data = np.load(sys.argv[1], allow_pickle=True)

elevation    = data["elevation"].astype(float)       # (R, C)
snapshots    = data["snapshots"].astype(np.uint8)    # (N, R, C)
ign_rc       = data["ignition_rc"]                   # [row, col]  primary seed
# All ignition seeds (multi-fire support): shape (K, 2). Falls back to
# single primary seed when ignition_rcs key is absent.
if "ignition_rcs" in data:
    ign_rcs = [tuple(int(x) for x in rc) for rc in data["ignition_rcs"]]
else:
    ign_rcs = [(int(ign_rc[0]), int(ign_rc[1]))]
cell_size_m  = float(data["cell_size_m"])
texture_path = data["texture_path"].tobytes().decode() if "texture_path" in data else ""

# Optional fuel map (new key)
fuel_map_raw   = data["fuel_map"].astype(int)  if "fuel_map"   in data else None
fuel_names_raw = data["fuel_names"].tolist()   if "fuel_names" in data else []

rows, cols = elevation.shape
N_frames   = len(snapshots)

# Minutes of simulated fire time per animation frame.
# Provided by gui_launcher: dt_fine * snap_every.  Falls back to 1 min.
frame_dt_min = float(data["frame_dt_min"]) if "frame_dt_min" in data else 1.0

# Wind (may be 2-D arrays or scalars)
wu = data["wind_u"] if "wind_u" in data else np.zeros_like(elevation)
wv = data["wind_v"] if "wind_v" in data else np.zeros_like(elevation)
if wu.ndim == 0:
    wu = np.full_like(elevation, float(wu))
if wv.ndim == 0:
    wv = np.full_like(elevation, float(wv))

print(f"[3D] Terrain {rows}×{cols}  |  {N_frames} frames  |  cell={cell_size_m:.0f} m")
if fuel_map_raw is not None:
    print(f"[3D] Fuel map loaded  |  {len(fuel_names_raw)} fuel types")

# ── Use the actual simulation grid directly — no artificial zoom ──────────────
# The fine-resolution run uses an 800×800 grid (~69 m cells for ±0.25° bbox),
# which is real COP30 data.  We display it as-is.  A small 2× nearest-neighbour
# upsample is applied only when the incoming grid is coarser than 400 px/side
# (e.g. a legacy 200×200 snapshot) to avoid PyVista rendering a blocky mesh.
MAX_DISPLAY = 1600
SCALE = 1
if max(rows, cols) < 400:
    SCALE = min(2, MAX_DISPLAY // max(rows, cols))

if SCALE > 1:
    from scipy.ndimage import zoom as _zoom
    elev_disp = _zoom(elevation, SCALE, order=1)   # bilinear terrain
    wu_disp   = _zoom(wu,        SCALE, order=1)
    wv_disp   = _zoom(wv,        SCALE, order=1)
    disp_r, disp_c = elev_disp.shape
    print(f"[3D] Display mesh: {disp_r}×{disp_c} (×{SCALE} bilinear, source <400px)")
else:
    elev_disp, wu_disp, wv_disp = elevation, wu, wv
    disp_r, disp_c = rows, cols
    print(f"[3D] Display mesh: {disp_r}×{disp_c} (native resolution)")

# Effective cell size on the display grid
disp_cell = cell_size_m / SCALE

# ── Build metric coordinate grids ─────────────────────────────────────────────
Z_EXG = 1.5   # vertical exaggeration
xx, yy = np.meshgrid(
    np.arange(disp_c, dtype=float) * disp_cell,
    np.arange(disp_r, dtype=float) * disp_cell,
)
zz = elev_disp * Z_EXG

# ── PyVista structured grid ───────────────────────────────────────────────────
mesh = pv.StructuredGrid(xx, yy, zz)

# ── Terrain texture: satellite JPG → vertex RGB at display resolution ────────
terrain_actor_name = "terrain"
texture_loaded = False
if texture_path and os.path.exists(texture_path):
    try:
        img = Image.open(texture_path).convert("RGB")
        img = img.resize((disp_c, disp_r), Image.LANCZOS)
        rgb = np.flipud(np.array(img)).reshape(-1, 3).astype(np.uint8)
        mesh.point_data["RGB"] = rgb
        texture_loaded = True
        print(f"[3D] Satellite texture loaded: {texture_path}")
    except Exception as e:
        print(f"[3D] Texture load failed: {e}")

# ── Fuel-type colour overlay (semi-transparent, static) ──────────────────────
# Each pixel is coloured by its fuel type.  This layer sits between the terrain
# and the fire overlay so the user can see what vegetation is burning.
# When a satellite texture is loaded, this blends with it at reduced alpha.
fuel_mesh = pv.StructuredGrid(xx, yy, zz + 1.0)
if fuel_map_raw is not None:
    if SCALE > 1:
        from scipy.ndimage import zoom as _zoom
        fuel_disp = np.round(_zoom(fuel_map_raw.astype(float), SCALE, order=0)).astype(int)
    else:
        fuel_disp = fuel_map_raw
    flat_fuel = fuel_disp.ravel()
    fuel_rgba_flat = np.zeros((len(flat_fuel), 4), dtype=np.uint8)
    for fi, fname in enumerate(fuel_names_raw):
        rgba = _fuel_color_rgba(fname)
        mask = flat_fuel == fi
        fuel_rgba_flat[mask] = rgba
    fuel_mesh.point_data["fuel_rgba"] = fuel_rgba_flat
    print(f"[3D] Fuel overlay built  ({len(fuel_names_raw)} types)")
else:
    # No fuel data — transparent placeholder
    fuel_rgba_flat = np.zeros((fuel_mesh.n_points, 4), dtype=np.uint8)
    fuel_mesh.point_data["fuel_rgba"] = fuel_rgba_flat

# ── Pre-compute burn-frame map for age-based fire colouring ──────────────────
# burn_frame[r,c] = frame index when cell first reached state==2 (burned).
# This allows colouring the scar from hot orange (fresh) to dark ash (old).
burn_frame = np.full((rows, cols), N_frames, dtype=int)  # default = never burned
for fi in range(N_frames):
    newly = (snapshots[fi] == 2) & (burn_frame == N_frames)
    burn_frame[newly] = fi
if SCALE > 1:
    burn_frame_disp = np.repeat(np.repeat(burn_frame, SCALE, axis=0), SCALE, axis=1)
else:
    burn_frame_disp = burn_frame

# ── Fire overlay mesh (slightly above terrain, at display resolution) ─────────
fire_mesh = pv.StructuredGrid(xx, yy, zz + 3.0)
fire_colors = np.zeros((fire_mesh.n_points, 4), dtype=np.uint8)
fire_mesh.point_data["fire_rgba"] = fire_colors

# ── Wind quivers (use display grid, sampled at coarse step) ───────────────────
q_step = max(1, min(disp_r, disp_c) // 20)
qi = np.arange(0, disp_r, q_step)
qj = np.arange(0, disp_c, q_step)
qI, qJ = np.meshgrid(qi, qj, indexing="ij")
q_x = (qJ * disp_cell).ravel()
q_y = (qI * disp_cell).ravel()
q_z = zz[qI, qJ].ravel() + 80.0
q_u = wu_disp[qI, qJ].ravel()
q_v = wv_disp[qI, qJ].ravel()
vec_pts = np.column_stack((q_x, q_y, q_z))
vec_dir = np.column_stack((q_u, q_v, np.zeros_like(q_u)))
wind_cloud = pv.PolyData(vec_pts)
wind_cloud["wind"] = vec_dir
arrows = wind_cloud.glyph(orient="wind", factor=disp_cell * 3, scale=True)

# ── Ignition stars (all seeds — supports multi-fire events) ──────────────────
# Build PolyData objects now; add to plotter after pl is created (below).
_ign_star_meshes = []   # list of (PolyData, label_str, x, y, z)
for _si, (_s_r, _s_c) in enumerate(ign_rcs):
    _sg_r = min(_s_r * SCALE, disp_r - 1)
    _sg_c = min(_s_c * SCALE, disp_c - 1)
    _sx   = _sg_c * disp_cell
    _sy   = _sg_r * disp_cell
    _sz   = zz[_sg_r, _sg_c] + 80.0
    _pt   = pv.PolyData(np.array([[_sx, _sy, _sz]]))
    _lbl  = f"★ Seed {_si + 1}" if len(ign_rcs) > 1 else "★ Ignition"
    _ign_star_meshes.append((_pt, _lbl, _sx, _sy, _sz))


# ── Plotter setup ─────────────────────────────────────────────────────────────
pl = pv.Plotter(title="Project WILSON — 3D Fire Terrain", window_size=[1400, 900])
pl.set_background("#0d0d1a")

if texture_loaded:
    pl.add_mesh(mesh, scalars="RGB", rgb=True, lighting=True, name=terrain_actor_name,
                smooth_shading=True)
else:
    pl.add_mesh(mesh, cmap="terrain", show_scalar_bar=False, lighting=True,
                name=terrain_actor_name, smooth_shading=True)

# Fuel overlay: show at reduced opacity so terrain texture remains visible
fuel_alpha_factor = 0.55 if texture_loaded else 0.85
pl.add_mesh(fuel_mesh, scalars="fuel_rgba", rgba=True, show_scalar_bar=False,
            opacity=fuel_alpha_factor, name="fuel_overlay", lighting=False)

pl.add_mesh(fire_mesh, scalars="fire_rgba", rgba=True, show_scalar_bar=False,
            name="fire", lighting=False)
pl.add_mesh(arrows, color="cyan", opacity=0.7, name="wind")

# Add all ignition seed markers (built before pl was created)
_star_points = np.array([[_sx, _sy, _sz] for _, _, _sx, _sy, _sz in _ign_star_meshes])
_star_labels = [_lbl for _, _lbl, _, _, _ in _ign_star_meshes]
for _si, (_pt, _lbl, _sx, _sy, _sz) in enumerate(_ign_star_meshes):
    pl.add_mesh(_pt, color="yellow", point_size=18, render_points_as_spheres=True,
                name=f"ignition_{_si}")
# Use add_point_labels for 3D world-space text (add_text only accepts pixel/screen coords)
if len(_star_points) > 0:
    _label_pts = _star_points.copy()
    _label_pts[:, 2] += 150   # raise label above the star sphere
    pl.add_point_labels(_label_pts, _star_labels,
                        font_size=12, text_color="yellow",
                        shape_opacity=0.0, always_visible=True,
                        name="ign_labels")


# ── Fuel legend (top-right text block) ───────────────────────────────────────
if fuel_names_raw:
    _legend_entries = []
    for fname in fuel_names_raw:
        if fname in ("Non_Combustible",):
            continue
        rgba = _fuel_color_rgba(fname)
        hex_col = "#{:02x}{:02x}{:02x}".format(rgba[0], rgba[1], rgba[2])
        _legend_entries.append([fname.replace("_", " "), hex_col])
    if _legend_entries:
        pl.add_legend(_legend_entries, bcolor=(0.05, 0.05, 0.1),
                      border=False, size=(0.10, 0.24), loc="lower right",
                      face="rectangle")

# ── State ─────────────────────────────────────────────────────────────────────
state = {"frame": 0, "paused": False, "last_t": time.time(), "fps": 6}

def _render_frame(idx):
    snap = snapshots[idx]                    # (rows, cols) uint8
    if SCALE > 1:
        snap_d = np.repeat(np.repeat(snap, SCALE, axis=0), SCALE, axis=1)
    else:
        snap_d = snap
    flat = snap_d.ravel()

    # Base: transparent (unburned cells show fuel overlay and terrain below)
    c = np.zeros((disp_r * disp_c, 4), dtype=np.uint8)

    # ── Fire-front detection: interior burning cells → charcoal ──────────────
    # Reshape to 2D for morphological ops, then flatten back.
    snap2d   = snap_d.astype(np.int8)
    consumed = snap2d > 0
    active2d = snap2d == 1
    from scipy.ndimage import binary_erosion
    interior_2d = active2d & binary_erosion(consumed, structure=np.ones((3, 3)))
    front_2d    = active2d & ~interior_2d

    front_flat    = front_2d.ravel()
    interior_flat = interior_2d.ravel()

    # Active fire FRONT — bright yellow-orange glow
    c[front_flat]    = [255, 160, 10, 250]
    # Interior burning (no unburned neighbours) — dark ember/charcoal
    c[interior_flat] = [60,  15,  5,  230]

    # ── Burned scar: age-based colour gradient ────────────────────────────────
    # Age 0 (just burned) → deep red-orange   (180, 40, 0)
    # Age ½ sim           → dark ember         (70, 15, 5)
    # Age full sim        → charcoal black     (20, 10, 8)
    burned_mask = flat == 2
    if burned_mask.any():
        age = (idx - burn_frame_disp.ravel()[burned_mask]).clip(0)
        max_age = max(N_frames // 3, 1)
        t = np.clip(age / max_age, 0.0, 1.0)
        r_ch = np.clip(180*(1-t) + 20*t, 0, 255).astype(np.uint8)
        g_ch = np.clip(40*(1-t)  + 10*t, 0, 255).astype(np.uint8)
        b_ch = np.clip(0*(1-t)   +  8*t, 0, 255).astype(np.uint8)
        a_ch = np.clip(235*(1-t) + 210*t, 0, 255).astype(np.uint8)
        c[burned_mask] = np.column_stack([r_ch, g_ch, b_ch, a_ch])

    fire_mesh.point_data["fire_rgba"] = c

    burned_ha = int((flat == 2).sum()) * cell_size_m**2 / 10_000
    active_ha  = int((flat == 1).sum()) * cell_size_m**2 / 10_000
    elapsed_min = idx * frame_dt_min
    h, m = divmod(int(elapsed_min), 60)
    time_str = f"{h:02d}h{m:02d}m" if h > 0 else f"{elapsed_min:.0f} min"
    pl.add_text(
        f"t = {time_str}  (frame {idx+1}/{N_frames})  |  "
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

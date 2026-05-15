"""
gui_launcher.py  -  Project WILSON  |  Wildfire Hindcast GUI
=============================================================
Step-by-step pipeline GUI: FIRMS -> ERA5 -> Terrain -> Optimize -> Animate

Run with:
    python gui_launcher.py
or:
    conda run -n computational_geometry python gui_launcher.py
"""

import sys
import os
import io
import platform
import threading
import datetime
import subprocess
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cross-platform file/folder opener
_OS_OPENER = {"Darwin": "open", "Windows": "explorer", "Linux": "xdg-open"}
_OPEN_CMD  = _OS_OPENER.get(platform.system(), "xdg-open")

def _open_path(path: str):
    """Open a file or folder with the OS default application."""
    subprocess.Popen([_OPEN_CMD, path])

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
BG_PANEL  = "#2a2a3e"
BG_ENTRY  = "#313150"
ACCENT    = "#f38ba8"
ACCENT2   = "#89b4fa"
GREEN     = "#a6e3a1"
YELLOW    = "#f9e2af"
TEXT      = "#cdd6f4"
TEXT_DIM  = "#7f849c"
FONT_MAIN = ("Segoe UI", 10)
FONT_MONO = ("Courier New", 9)
FONT_HEAD = ("Segoe UI Semibold", 11)

STEP_NAMES = [
    "Fetch NASA FIRMS data",
    "Identify ignition point",
    "Fetch ERA5 weather",
    "Fetch terrain (DEM + CORINE)",
    "Extract ground truth mask",
    "Run Differential Evolution",
]
_ICON     = {"pending": "o", "running": "@", "done": "v", "error": "X"}
_ICON_CLR = {"pending": TEXT_DIM, "running": YELLOW, "done": GREEN, "error": ACCENT}

FUEL_COLORS = ["#228B22", "#8B4513", "#556B2F", "#DAA520", "#90EE90", "#D3D3D3", "#4682B4"]


# ──────────────────────────────────────────────────────────────────────────────
# stdout redirect to Tk Text widget
# ──────────────────────────────────────────────────────────────────────────────

class _TextRedirector(io.TextIOBase):
    def __init__(self, widget, after_fn):
        self._widget = widget
        self._after  = after_fn

    def write(self, text: str) -> int:
        self._after(0, self._append, text)
        return len(text)

    def _append(self, text: str):
        self._widget.configure(state="normal")
        if "NEW BEST" in text:
            self._widget.insert(tk.END, text, "best")
        elif text.strip().startswith("[") or text.strip().startswith("="):
            self._widget.insert(tk.END, text, "head")
        elif "Traceback" in text or "Error" in text:
            self._widget.insert(tk.END, text, "err")
        else:
            self._widget.insert(tk.END, text)
        self._widget.see(tk.END)
        self._widget.configure(state="disabled")

    def flush(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Main application window
# ──────────────────────────────────────────────────────────────────────────────

class WilsonGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Project WILSON -- Wildfire Digital Twin")
        self.configure(bg=BG)
        self.geometry("1300x880")
        self.resizable(True, True)

        self._result      = None
        self._running     = False
        self._snapshots   = []
        self._anim_job    = None
        self._anim_frame  = 0
        self._hill_anim_job   = None
        self._hill_anim_frame = 0
        self._hillshade_cache = None
        self._conv_evals  = []
        self._conv_errors = []
        self._conv_best   = []
        # Clock / fire-time display
        self._wall_clock_var = tk.StringVar(value="")
        self._fire_time_var  = tk.StringVar(value="")

        self._setup_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── ttk styles ─────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",        background=BG,       borderwidth=0)
        s.configure("TNotebook.Tab",    background=BG_PANEL, foreground=TEXT_DIM,
                    font=FONT_MAIN, padding=[10, 4])
        s.map("TNotebook.Tab",          background=[("selected", BG_ENTRY)],
                                        foreground=[("selected", TEXT)])
        s.configure("TProgressbar",     troughcolor=BG_PANEL, background=ACCENT2)
        s.configure("Dark.TCheckbutton", background=BG_PANEL, foreground=TEXT,
                    font=FONT_MAIN)

    # ── Top-level layout ───────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(hdr, text="Project WILSON",
                 font=("Segoe UI Semibold", 18), fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hdr,
                 text="  Drone-Based Wildfire Hindcast  |  Rothermel CA + ERA5 + FIRMS",
                 font=FONT_MAIN, fg=TEXT_DIM, bg=BG).pack(side="left")
        # ── Wall clock (real time) ──────────────────────────────────────────
        tk.Label(hdr, textvariable=self._wall_clock_var,
                 font=FONT_MONO, fg=ACCENT2, bg=BG).pack(side="right", padx=(0, 8))
        # ── Fire simulation time (updates during animation) ─────────────────
        tk.Label(hdr, textvariable=self._fire_time_var,
                 font=FONT_MONO, fg=ACCENT, bg=BG).pack(side="right", padx=(0, 16))
        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x", padx=16, pady=(4, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_canvas(body)
        self._tick_wall_clock()   # start 1-second real-time ticker

    def _tick_wall_clock(self):
        """Update the wall clock label every second."""
        self._wall_clock_var.set("🕐 " + datetime.datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._tick_wall_clock)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=BG_PANEL, width=290)
        sidebar.pack(side="left", fill="y", padx=(8, 0), pady=8)
        sidebar.pack_propagate(False)

        def section(title):
            tk.Label(sidebar, text=title, font=FONT_HEAD,
                     fg=ACCENT2, bg=BG_PANEL, anchor="w").pack(fill="x", padx=10, pady=(10, 2))
            tk.Frame(sidebar, bg=ACCENT2, height=1).pack(fill="x", padx=10, pady=(0, 4))

        def entry_row(label, default, tip=""):
            row = tk.Frame(sidebar, bg=BG_PANEL)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, font=FONT_MAIN, fg=TEXT,
                     bg=BG_PANEL, width=16, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            e = tk.Entry(row, textvariable=var, font=FONT_MAIN,
                         bg=BG_ENTRY, fg=TEXT, insertbackground=TEXT,
                         relief="flat", bd=3, width=10)
            e.pack(side="left", fill="x", expand=True)
            if tip:
                _ToolTip(e, tip)
            return var

        section("FIRMS / Study Area")
        self.v_key     = entry_row("API Key",    "2694cf69813b733c8e75e2fc038bac40", "NASA FIRMS map key")
        self.v_lat_min = entry_row("Lat Min N",  "35.8",  "Southern edge of search box")
        self.v_lat_max = entry_row("Lat Max N",  "36.5",  "Northern edge of search box")
        self.v_lon_min = entry_row("Lon Min E",  "27.0",  "Western edge of search box")
        self.v_lon_max = entry_row("Lon Max E",  "28.5",  "Eastern edge of search box")

        row = tk.Frame(sidebar, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=4)
        tk.Label(row, text="Quick Preset", font=FONT_MAIN, fg=TEXT_DIM,
                 bg=BG_PANEL, width=16, anchor="w").pack(side="left")
        self._preset_var = tk.StringVar(value="Rhodes 2023")
        cb = ttk.Combobox(row, textvariable=self._preset_var,
                          values=["Rhodes 2023", "Evoia 2021", "Custom"],
                          state="readonly", width=12)
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", self._apply_preset)

        section("Date & Window")
        self.v_date_start   = entry_row("Date Start",    "2023-07-19", "YYYY-MM-DD")
        self.v_date_end     = entry_row("Date End",      "2023-07-21", "YYYY-MM-DD")
        self.v_hindcast_hrs = entry_row("Hindcast h",    "6",          "Min window (auto-extended)")

        section("Terrain & Optimiser")
        self.v_terrain_buf  = entry_row("Terrain buf",   "0.25",  "Half-width degrees (~28km)")
        self.v_maxiter      = entry_row("Max Gens",      "20",    "DE generations")
        self.v_popsize      = entry_row("Pop Size",      "12",    "Candidates/generation (>=5)")

        section("Output")
        self.v_save_plot = tk.BooleanVar(value=True)
        ttk.Checkbutton(sidebar, text="Save result figure",
                        variable=self.v_save_plot,
                        style="Dark.TCheckbutton").pack(anchor="w", padx=12)
        self.v_open_plot = tk.BooleanVar(value=True)
        ttk.Checkbutton(sidebar, text="Auto-open figure on done",
                        variable=self.v_open_plot,
                        style="Dark.TCheckbutton").pack(anchor="w", padx=12)

        section("Ground Truth")
        # ── Copernicus EMS shapefile (Priority 1) ─────────────────────────
        shp_row = tk.Frame(sidebar, bg=BG_PANEL)
        shp_row.pack(fill="x", padx=10, pady=4)
        tk.Label(shp_row, text="Copernicus .shp", font=FONT_MAIN, fg=TEXT,
                 bg=BG_PANEL, width=16, anchor="w").pack(side="left")
        self.v_truth_shp = tk.StringVar(value="")
        shp_entry = tk.Entry(shp_row, textvariable=self.v_truth_shp,
                             font=("Segoe UI", 8), bg=BG_ENTRY, fg=TEXT_DIM,
                             insertbackground=TEXT, relief="flat", bd=3, width=8)
        shp_entry.pack(side="left", fill="x", expand=True)
        tk.Button(shp_row, text="Browse", font=("Segoe UI", 8),
                  bg=BG_ENTRY, fg=ACCENT2, relief="flat", bd=0, cursor="hand2",
                  command=self._browse_shapefile).pack(side="left", padx=(4, 0))
        _ToolTip(shp_entry, "Optional Copernicus EMS fire perimeter .shp or .geojson.\n"
                            "When set, replaces NASA FIRMS as ground truth.")

        # ── FIRED daily GeoPackage (Priority 2) ───────────────────────────
        fired_row = tk.Frame(sidebar, bg=BG_PANEL)
        fired_row.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(fired_row, text="FIRED .gpkg", font=FONT_MAIN, fg=TEXT,
                 bg=BG_PANEL, width=16, anchor="w").pack(side="left")
        self.v_fired_gpkg = tk.StringVar(value="")
        fired_entry = tk.Entry(fired_row, textvariable=self.v_fired_gpkg,
                               font=("Segoe UI", 8), bg=BG_ENTRY, fg=TEXT_DIM,
                               insertbackground=TEXT, relief="flat", bd=3, width=8)
        fired_entry.pack(side="left", fill="x", expand=True)
        tk.Button(fired_row, text="Browse", font=("Segoe UI", 8),
                  bg=BG_ENTRY, fg="#a6e3a1", relief="flat", bd=0, cursor="hand2",
                  command=self._browse_fired_gpkg).pack(side="left", padx=(4, 0))
        _ToolTip(fired_entry, "FIRED daily perimeter GeoPackage (.gpkg).\n"
                              "Used when no Copernicus shapefile is set.\n"
                              "Download: https://scholar.colorado.edu/collections/pz50gx05h")

        shp_lbl_row = tk.Frame(sidebar, bg=BG_PANEL)
        shp_lbl_row.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(shp_lbl_row, text="Leave blank to use NASA FIRMS (auto-fetched)",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL,
                 wraplength=250, anchor="w").pack(side="left")

        section("Pipeline Steps")
        self._step_labels = []
        self._step_status = []
        for i, name in enumerate(STEP_NAMES):
            frm = tk.Frame(sidebar, bg=BG_PANEL)
            frm.pack(fill="x", padx=10, pady=1)
            icon_var = tk.StringVar(value=_ICON["pending"])
            icon_lbl = tk.Label(frm, textvariable=icon_var, font=FONT_MONO,
                                fg=TEXT_DIM, bg=BG_PANEL, width=2)
            icon_lbl.pack(side="left")
            tk.Label(frm, text=f"{i+1}. {name}",
                     font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_PANEL,
                     anchor="w").pack(side="left")
            self._step_labels.append((icon_var, icon_lbl))
            self._step_status.append("pending")

        tk.Frame(sidebar, bg=BG_PANEL, height=8).pack()
        self.btn_run = tk.Button(
            sidebar, text="RUN ALL STEPS",
            font=("Segoe UI Semibold", 11),
            bg=ACCENT, fg="#1e1e2e", activebackground="#e06c94",
            relief="flat", bd=0, pady=8, cursor="hand2",
            command=self._run
        )
        self.btn_run.pack(fill="x", padx=10, pady=(4, 2))

        btn_row = tk.Frame(sidebar, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=10, pady=2)
        tk.Button(btn_row, text="Clear Log", font=FONT_MAIN,
                  bg=BG_ENTRY, fg=TEXT_DIM, relief="flat", bd=0,
                  cursor="hand2", command=self._clear_all
                  ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(btn_row, text="Open Folder", font=FONT_MAIN,
                  bg=BG_ENTRY, fg=TEXT_DIM, relief="flat", bd=0,
                  cursor="hand2",
                  command=lambda: _open_path(str(PROJECT_ROOT))
                  ).pack(side="left", expand=True, fill="x", padx=(2, 0))

        tk.Frame(sidebar, bg=BG_PANEL, height=6).pack()
        self.progress = ttk.Progressbar(sidebar, mode="indeterminate", length=250)
        self.progress.pack(fill="x", padx=10, pady=2)
        self.status_var = tk.StringVar(value="Ready -- fill in parameters and press RUN ALL STEPS.")
        tk.Label(sidebar, textvariable=self.status_var, font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_PANEL, wraplength=250,
                 anchor="w", justify="left").pack(fill="x", padx=10, pady=2)

    # ── Right canvas with tabs ─────────────────────────────────────────────────
    def _build_canvas(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        self._tab_log      = self._make_tab("Log")
        self._tab_terrain  = self._make_tab("Terrain")
        self._tab_fire     = self._make_tab("Fire Animation")
        self._tab_hillfire = self._make_tab("Hillshade Fire")
        self._tab_compare  = self._make_tab("Comparison")
        self._tab_fired_tl = self._make_tab("FIRED Timeline")
        self._tab_earth3d  = self._make_tab("3D Earth View")
        self._tab_analysis = self._make_tab("Analysis")

        self._build_log_tab(self._tab_log)
        self._build_terrain_tab(self._tab_terrain)
        self._build_fire_tab(self._tab_fire)
        self._build_hillfire_tab(self._tab_hillfire)
        self._build_compare_tab(self._tab_compare)
        self._build_fired_tab(self._tab_fired_tl)
        self._build_earth3d_tab(self._tab_earth3d)
        self._build_analysis_tab(self._tab_analysis)

    def _make_tab(self, title):
        frm = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(frm, text=title)
        return frm

    # ── Log tab ────────────────────────────────────────────────────────────────
    def _build_log_tab(self, frm):
        self.log = scrolledtext.ScrolledText(
            frm, font=FONT_MONO, bg="#11111b", fg=GREEN,
            insertbackground=TEXT, state="disabled", wrap="word", bd=0
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_config("best", foreground=ACCENT)
        self.log.tag_config("head", foreground=ACCENT2)
        self.log.tag_config("err",  foreground="#f38ba8")

    # ── Terrain tab ────────────────────────────────────────────────────────────
    def _build_terrain_tab(self, frm):
        self._fig_terrain, self._ax_terrain = plt.subplots(2, 2, figsize=(9, 7))
        self._fig_terrain.patch.set_facecolor("#1e1e2e")
        for ax in self._ax_terrain.flat:
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
        self._fig_terrain.tight_layout(pad=2.0)
        self._canvas_terrain = FigureCanvasTkAgg(self._fig_terrain, master=frm)
        self._canvas_terrain.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas_terrain, frm).pack(fill="x")

    # ── Fire animation tab ─────────────────────────────────────────────────────
    def _build_fire_tab(self, frm):
        self._fig_fire, self._ax_fire = plt.subplots(1, 1, figsize=(7, 6))
        self._fig_fire.patch.set_facecolor("#1e1e2e")
        self._ax_fire.set_facecolor("#11111b")
        self._ax_fire.tick_params(colors=TEXT_DIM, labelsize=7)
        self._canvas_fire = FigureCanvasTkAgg(self._fig_fire, master=frm)
        self._canvas_fire.get_tk_widget().pack(fill="both", expand=True)
        # Zoom/pan toolbar
        NavigationToolbar2Tk(self._canvas_fire, frm).pack(fill="x")

        ctrl = tk.Frame(frm, bg=BG_PANEL)
        ctrl.pack(fill="x", padx=4, pady=2)

        self.btn_play = tk.Button(ctrl, text="▶ Play", font=FONT_MAIN,
                                  bg=BG_ENTRY, fg=ACCENT2, relief="flat", bd=0,
                                  cursor="hand2", command=self._play_anim, state="disabled")
        self.btn_play.pack(side="left", padx=4, pady=4)

        self.btn_pause = tk.Button(ctrl, text="⏸ Pause", font=FONT_MAIN,
                                   bg=BG_ENTRY, fg=YELLOW, relief="flat", bd=0,
                                   cursor="hand2", command=self._pause_anim, state="disabled")
        self.btn_pause.pack(side="left", padx=4)

        tk.Label(ctrl, text="Speed (ms/frame):", font=FONT_MAIN,
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side="left", padx=(8, 2))
        self._anim_speed = tk.IntVar(value=80)
        tk.Scale(ctrl, from_=20, to=500, orient="horizontal",
                 variable=self._anim_speed,
                 bg=BG_PANEL, fg=TEXT_DIM, troughcolor=BG_ENTRY,
                 highlightthickness=0, length=120).pack(side="left")

        # Fire-time display (right side of control bar)
        tk.Label(ctrl, textvariable=self._fire_time_var,
                 font=FONT_MONO, fg=ACCENT, bg=BG_PANEL).pack(side="right", padx=8)

        self._frame_var = tk.IntVar(value=0)
        self._frame_slider = tk.Scale(
            frm, from_=0, to=1, orient="horizontal",
            variable=self._frame_var, label="Frame",
            bg=BG_PANEL, fg=TEXT, troughcolor=BG_ENTRY,
            highlightthickness=0, length=600,
            command=self._seek_frame
        )
        self._frame_slider.pack(fill="x", padx=4, pady=2)

    # ── Hillshade fire tab ─────────────────────────────────────────────────────
    def _build_hillfire_tab(self, frm):
        """
        Terrain-draped fire animation using hillshading.
        Shows the fire moving across the actual topography of the region —
        ridges, valleys and slopes clearly visible via shaded-relief rendering.
        """
        self._fig_hill, self._ax_hill = plt.subplots(1, 1, figsize=(8, 7))
        self._fig_hill.patch.set_facecolor("#1e1e2e")
        self._ax_hill.set_facecolor("#0a0a0a")
        self._ax_hill.tick_params(colors=TEXT_DIM, labelsize=7)
        self._canvas_hill = FigureCanvasTkAgg(self._fig_hill, master=frm)
        self._canvas_hill.get_tk_widget().pack(fill="both", expand=True)
        # Zoom/pan toolbar so user can interact with the map
        NavigationToolbar2Tk(self._canvas_hill, frm).pack(fill="x")

        ctrl = tk.Frame(frm, bg=BG_PANEL)
        ctrl.pack(fill="x", padx=4, pady=2)

        self.btn_hplay = tk.Button(ctrl, text="▶ Play", font=FONT_MAIN,
                                   bg=BG_ENTRY, fg=ACCENT2, relief="flat", bd=0,
                                   cursor="hand2", command=self._play_hill, state="disabled")
        self.btn_hplay.pack(side="left", padx=4, pady=4)
        self.btn_hpause = tk.Button(ctrl, text="⏸ Pause", font=FONT_MAIN,
                                    bg=BG_ENTRY, fg=YELLOW, relief="flat", bd=0,
                                    cursor="hand2", command=self._pause_hill, state="disabled")
        self.btn_hpause.pack(side="left", padx=4)
        tk.Label(ctrl, text="Speed (ms/frame):", font=FONT_MAIN,
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side="left", padx=(8, 2))
        self._hill_speed = tk.IntVar(value=100)
        tk.Scale(ctrl, from_=20, to=600, orient="horizontal",
                 variable=self._hill_speed,
                 bg=BG_PANEL, fg=TEXT_DIM, troughcolor=BG_ENTRY,
                 highlightthickness=0, length=120).pack(side="left")

        # Fire simulation time (right side of control bar)
        tk.Label(ctrl, textvariable=self._fire_time_var,
                 font=("Courier New", 11, "bold"), fg=ACCENT, bg=BG_PANEL).pack(side="right", padx=8)

        self._hill_frame_var = tk.IntVar(value=0)
        self._hill_slider = tk.Scale(
            frm, from_=0, to=1, orient="horizontal",
            variable=self._hill_frame_var, label="Frame",
            bg=BG_PANEL, fg=TEXT, troughcolor=BG_ENTRY,
            highlightthickness=0, length=600,
            command=self._seek_hill
        )
        self._hill_slider.pack(fill="x", padx=4, pady=2)

        # 3-D PyVista launch button (enabled after run completes)
        self.btn_3d = tk.Button(
            frm, text="🌋  Launch 3D View (PyVista)",
            font=FONT_MAIN, bg="#313244", fg="#cba6f7",
            relief="flat", bd=0, cursor="hand2",
            command=self._launch_3d_view, state="disabled"
        )
        self.btn_3d.pack(pady=(4, 6))

        self._hill_anim_job   = None
        self._hill_anim_frame = 0
        self._hillshade_cache = None   # pre-computed once from DEM

    def _compute_hillshade(self, elevation: np.ndarray,
                            azimuth_deg: float = 315.0,
                            altitude_deg: float = 45.0) -> np.ndarray:
        """
        Compute a true hillshade using matplotlib LightSource (vert_exag=2).
        Returns a float32 array in [0, 1].
        """
        from matplotlib.colors import LightSource
        ls    = LightSource(azdeg=azimuth_deg, altdeg=altitude_deg)
        shade = ls.hillshade(elevation.astype(float), vert_exag=2,
                             dx=1.0, dy=1.0)
        return shade.astype(np.float32)

    def _draw_hill_frame(self, idx):
        if not self._snapshots or idx >= len(self._snapshots):
            return
        land   = self._result["landscape"]
        r0, c0 = self._result["ignition_rc"]
        snap   = self._snapshots[idx]

        # Build hillshade once and cache it
        if self._hillshade_cache is None:
            self._hillshade_cache = self._compute_hillshade(land.elevation)

        ax = self._ax_hill
        ax.cla()
        ax.set_facecolor("#0a0a0a")

        # Layer 1: hillshaded terrain (grey)
        ax.imshow(self._hillshade_cache, cmap="gray", origin="lower",
                  vmin=0, vmax=1, alpha=1.0)

        # Layer 2: elevation tint (terrain colours, semi-transparent)
        ax.imshow(land.elevation, cmap="terrain", origin="lower", alpha=0.45)

        # Layer 3: fire state — glowing embers
        fire_rgba = np.zeros((*snap.shape, 4), dtype=np.float32)
        fire_rgba[snap == 1] = [1.0, 0.22, 0.0, 0.90]   # active front — bright orange-red
        fire_rgba[snap == 2] = [0.25, 0.08, 0.0, 0.65]  # burned area — dark char
        # Add a faint glow halo around active cells (dilation by 1)
        from scipy.ndimage import binary_dilation
        active = snap == 1
        glow   = binary_dilation(active, iterations=2) & ~active
        fire_rgba[glow] = [1.0, 0.5, 0.0, 0.20]
        ax.imshow(fire_rgba, origin="lower")

        # Ignition marker
        ax.plot(c0, r0, "w*", ms=10, markeredgecolor="yellow", markeredgewidth=0.8)

        hindcast_steps = self._result.get("hindcast_steps", len(self._snapshots))
        snap_every     = max(1, hindcast_steps // 80)
        import config as _cfg
        t_min = idx * snap_every * getattr(_cfg, "dt", 1.0)

        burned_cells = int(np.sum(snap > 0))
        import config as _cfg2
        cell_m       = getattr(land, "cell_size_m",
                               getattr(land.config if hasattr(land, "config") else _cfg2,
                                       "CELL_SIZE_METERS", 417))
        burned_ha    = burned_cells * cell_m**2 / 10_000

        # ── Update fire-time clock in header ─────────────────────────────────
        ign_time = self._result.get("ignition_time")
        if ign_time is not None:
            fire_dt = ign_time + datetime.timedelta(minutes=t_min)
            self._fire_time_var.set(f"🔥 {fire_dt.strftime('%Y-%m-%d %H:%M UTC')}  (+{t_min:.0f} min)")
        else:
            self._fire_time_var.set(f"🔥 t = {t_min:.0f} min  ({burned_ha:.1f} ha)")

        ax.set_title(
            f"Fire on terrain  |  t ~ {t_min:.0f} min  |  "
            f"{burned_ha:.1f} ha burned  |  frame {idx+1}/{len(self._snapshots)}",
            color=TEXT, fontsize=9
        )
        ax.tick_params(colors=TEXT_DIM, labelsize=7)

        # Contour lines for topography
        try:
            elev = land.elevation
            levels = np.arange(elev.min(), elev.max(), max(50, (elev.max()-elev.min())/8))
            ax.contour(elev, levels=levels, colors="white", linewidths=0.3,
                       alpha=0.25, origin="lower")
        except Exception:
            pass

        self._canvas_hill.draw()

    def _update_hillfire_tab(self):
        if not self._snapshots:
            return
        self._hillshade_cache = None   # force recompute with new landscape
        self._hill_slider.configure(to=len(self._snapshots) - 1)
        self._hill_frame_var.set(0)
        # Draw frame 0 first, then force canvas flush before switching tab
        self._draw_hill_frame(0)
        self._canvas_hill.draw()          # <-- fix: flush so canvas isn't black
        self._canvas_hill.get_tk_widget().update()
        self.btn_hplay.configure(state="normal")
        self.btn_hpause.configure(state="normal")
        # Select tab AFTER the canvas is fully painted
        self.after(50, lambda: self.notebook.select(self._tab_hillfire))

    def _seek_hill(self, val):
        if self._hill_anim_job:
            self.after_cancel(self._hill_anim_job)
            self._hill_anim_job = None
        self._draw_hill_frame(int(float(val)))

    def _play_hill(self):
        if self._hill_anim_job:
            return
        self._hill_anim_frame = int(self._hill_frame_var.get())
        self._tick_hill()

    def _pause_hill(self):
        if self._hill_anim_job:
            self.after_cancel(self._hill_anim_job)
            self._hill_anim_job = None

    def _tick_hill(self):
        if not self._snapshots:
            return
        if self._hill_anim_frame >= len(self._snapshots):
            self._hill_anim_frame = 0
        self._hill_frame_var.set(self._hill_anim_frame)
        self._draw_hill_frame(self._hill_anim_frame)
        self._hill_anim_frame += 1
        self._hill_anim_job = self.after(self._hill_speed.get(), self._tick_hill)

    # ── Comparison tab ─────────────────────────────────────────────────────────
    def _build_compare_tab(self, frm):
        """
        Multi-window hindcast comparison tab.
        Shows predicted fire spread at 1/4, 1/2 and full hindcast time alongside
        the FIRMS truth mask and per-window statistics.
        """
        self._fig_compare, self._ax_compare = plt.subplots(2, 3, figsize=(14, 7))
        self._fig_compare.patch.set_facecolor("#1e1e2e")
        for ax in self._ax_compare.flat:
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(TEXT_DIM)
        self._fig_compare.tight_layout(pad=2.5)
        self._canvas_compare = FigureCanvasTkAgg(self._fig_compare, master=frm)
        self._canvas_compare.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas_compare, frm).pack(fill="x")

    def _update_compare_tab(self):
        """Populate the Comparison tab after a successful run."""
        if self._result is None or not self._snapshots:
            return

        land      = self._result["landscape"]
        truth     = self._result["truth_mask"]
        r0, c0    = self._result["ignition_rc"]
        ign_time  = self._result.get("ignition_time")
        weather   = self._result["weather"]
        iou_final = self._result.get("iou", 0) * 100
        best      = self._result["best_params"]
        import config as _cfg
        hindcast_steps = self._result.get("hindcast_steps", len(self._snapshots))
        snap_every     = max(1, hindcast_steps // 80)
        dt             = getattr(_cfg, "dt", 1.0)
        cell_m         = getattr(land.config if hasattr(land, "config") else _cfg,
                                 "CELL_SIZE_METERS", 417)

        n_snaps  = len(self._snapshots)
        quarter  = max(0, n_snaps // 4 - 1)
        half     = max(0, n_snaps // 2 - 1)
        full_idx = n_snaps - 1
        windows  = [quarter, half, full_idx]
        labels   = ["1/4 window", "1/2 window", "Full window"]

        for ax in self._ax_compare.flat:
            ax.cla()
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=7)

        for col, (snap_idx, label) in enumerate(zip(windows, labels)):
            snap      = self._snapshots[snap_idx]
            t_min     = snap_idx * snap_every * dt
            pred_mask = snap > 0

            # Top row: predicted (blue) + truth (red) overlay on elevation
            ax = self._ax_compare[0][col]
            ax.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.4)
            ov_t = np.zeros((*land.shape, 4)); ov_t[truth]     = [1, 0.1, 0.1, 0.7]
            ov_p = np.zeros((*land.shape, 4)); ov_p[pred_mask] = [0.2, 0.5, 1.0, 0.55]
            ax.imshow(ov_t, origin="lower")
            ax.imshow(ov_p, origin="lower")
            ax.plot(c0, r0, "y*", ms=9, markeredgecolor="white", markeredgewidth=0.6)

            # ── FIRED daily perimeter contour overlay ─────────────────────
            fired_tl = self._result.get("fired_timeline")
            if fired_tl is not None and len(fired_tl):
                try:
                    from pipeline.fired_loader import fired_polygon_to_grid_mask as _f2g
                    terrain_bbox = self._result.get("terrain_bbox")
                    if terrain_bbox is not None:
                        # Pick the FIRED snapshot closest in time to this panel's t_min
                        import pandas as _pd
                        ign_t = self._result.get("ignition_time")
                        if ign_t is not None:
                            target_dt = ign_t + _pd.Timedelta(minutes=t_min)
                            diffs = (fired_tl["burn_date"] - target_dt).abs()
                            closest_row = fired_tl.iloc[[diffs.argmin()]]
                        else:
                            closest_row = fired_tl
                        fired_mask = _f2g(closest_row, terrain_bbox, land.shape)
                        # Resize to match display if needed
                        if fired_mask.shape == land.shape and fired_mask.any():
                            ax.contour(fired_mask.astype(float),
                                       levels=[0.5], colors=["lime"],
                                       linewidths=1.2, origin="lower",
                                       linestyles="solid", alpha=0.85)
                except Exception as _fe:
                    pass  # FIRED overlay is cosmetic — never block the tab

            iou_w   = 0.0
            union_w = int(np.logical_or(pred_mask, truth).sum())
            if union_w > 0:
                iou_w = float(np.logical_and(pred_mask, truth).sum()) / union_w * 100
            burned_ha = float(pred_mask.sum()) * cell_m**2 / 10_000
            if ign_time is not None:
                fire_dt  = ign_time + datetime.timedelta(minutes=t_min)
                time_str = fire_dt.strftime("%Y-%m-%d %H:%M UTC")
            else:
                time_str = f"t={t_min:.0f} min"
            ax.set_title(
                f"{label}  [{time_str}]\nPred {burned_ha:.1f} ha  IoU={iou_w:.1f}%",
                color=TEXT, fontsize=8
            )
            legend_handles = [
                mpatches.Patch(color=[1, 0.1, 0.1, 0.7],    label="Truth"),
                mpatches.Patch(color=[0.2, 0.5, 1.0, 0.55], label="Predicted"),
            ]
            if fired_tl is not None and len(fired_tl):
                legend_handles.append(
                    mpatches.Patch(color="lime", label="FIRED perimeter")
                )
            ax.legend(handles=legend_handles, fontsize=6, loc="lower right")

            # Bottom row: TP / FP / FN coloured overlay
            ax2 = self._ax_compare[1][col]
            tp  = pred_mask & truth
            fp  = pred_mask & ~truth
            fn  = ~pred_mask & truth
            ov2 = np.zeros((*land.shape, 4))
            ov2[tp] = [0.0, 0.85, 0.0, 0.85]
            ov2[fp] = [0.2, 0.4,  1.0, 0.6 ]
            ov2[fn] = [1.0, 0.0,  0.0, 0.85]
            ax2.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.35)
            ax2.imshow(ov2, origin="lower")
            ax2.plot(c0, r0, "y*", ms=9, markeredgecolor="white", markeredgewidth=0.6)
            ax2.set_title(f"TP={tp.sum()}  FP={fp.sum()}  FN={fn.sum()}",
                          color=TEXT, fontsize=8)
            ax2.legend(handles=[
                mpatches.Patch(color=[0, 0.85, 0, 0.85], label=f"TP {tp.sum()}"),
                mpatches.Patch(color=[0.2, 0.4, 1, 0.6], label=f"FP {fp.sum()}"),
                mpatches.Patch(color=[1, 0, 0, 0.85],    label=f"FN {fn.sum()}"),
            ], fontsize=6, loc="lower right")

        # Bottom-right panel: burn area growth over time vs FIRMS truth
        frames_t     = [i * snap_every * dt for i in range(n_snaps)]
        burned_ha_ts = [float(np.sum(s > 0)) * cell_m**2 / 10_000
                        for s in self._snapshots]
        truth_ha     = float(truth.sum()) * cell_m**2 / 10_000
        ax_grow = self._ax_compare[1][2]
        ax_grow.cla(); ax_grow.set_facecolor("#11111b")
        ax_grow.tick_params(colors=TEXT_DIM, labelsize=8)
        ax_grow.plot(frames_t, burned_ha_ts, color=ACCENT, lw=1.8, label="Model")
        ax_grow.axhline(truth_ha, color=ACCENT2, lw=1.2, ls="--",
                        label=f"FIRMS truth  ({truth_ha:.1f} ha)")
        for snap_idx, label in zip(windows, labels):
            t = snap_idx * snap_every * dt
            ax_grow.axvline(t, color=YELLOW, lw=0.8, ls=":", alpha=0.7)
            ax_grow.text(t, truth_ha * 0.05, label, color=YELLOW, fontsize=6, rotation=90)
        ax_grow.fill_between(frames_t, burned_ha_ts, alpha=0.2, color=ACCENT)
        ax_grow.set_xlabel("Minutes after ignition", color=TEXT_DIM, fontsize=8)
        ax_grow.set_ylabel("Burn area (ha)", color=TEXT_DIM, fontsize=8)
        ax_grow.set_title("Burn Area vs FIRMS Truth", color=TEXT, fontsize=9)
        ax_grow.legend(fontsize=7)

        self._fig_compare.suptitle(
            f"Multi-Window Comparison  |  "
            f"ERA5 {weather['wind_speed_ms']:.1f} m/s x {best['wind_multiplier']:.2f}  "
            f"dm={best['fuel_moisture_offset']:+.3f}  Final IoU={iou_final:.1f}%",
            color=TEXT_DIM, fontsize=9
        )
        self._fig_compare.tight_layout(pad=2.0)
        self._canvas_compare.draw()

    # ── FIRED Timeline tab ────────────────────────────────────────────────────

    def _build_fired_tab(self, frm):
        """
        FIRED daily perimeter growth visualiser.

        Left panel  — cumulative burn area (ha) over time vs model prediction.
        Right panel — daily perimeter polygons rendered as a colour gradient
                      (yellow = first day, red = last day) on the terrain grid.
        """
        self._fig_fired, self._ax_fired = plt.subplots(1, 2, figsize=(14, 6))
        self._fig_fired.patch.set_facecolor("#1e1e2e")
        for ax in self._ax_fired:
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(TEXT_DIM)
        # Placeholder message until data arrives
        for ax in self._ax_fired:
            ax.text(0.5, 0.5, "Run the pipeline with a FIRED .gpkg file\nto see the timeline.",
                    ha="center", va="center", color=TEXT_DIM, fontsize=10,
                    transform=ax.transAxes)
        self._fig_fired.tight_layout(pad=2.5)
        self._canvas_fired = FigureCanvasTkAgg(self._fig_fired, master=frm)
        self._canvas_fired.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas_fired, frm).pack(fill="x")

    def _update_fired_tab(self):
        """Populate the FIRED Timeline tab after a successful run."""
        if self._result is None:
            return

        fired_tl = self._result.get("fired_timeline")
        land     = self._result["landscape"]
        r0, c0   = self._result["ignition_rc"]
        ign_time = self._result.get("ignition_time")
        import config as _cfg
        cell_m   = getattr(land.config if hasattr(land, "config") else _cfg,
                            "CELL_SIZE_METERS", 417)

        ax_area, ax_map = self._ax_fired
        ax_area.cla(); ax_map.cla()
        for ax in (ax_area, ax_map):
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=8)

        # ── If no FIRED data: show informative placeholder ─────────────────
        if fired_tl is None or len(fired_tl) == 0:
            msg = (
                "No FIRED daily polygons found for this run.\n\n"
                "Possible reasons:\n"
                "  • No FIRED .gpkg file was selected (sidebar is blank)\n"
                "  • The FIRED dataset does not cover this region/date\n"
                "    (the default FIRED dataset covers Western USA fires)\n"
                "  • The bbox or date range produced no matching events\n\n"
                "For Mediterranean fires, use a Copernicus EMS .shp\n"
                "or leave blank to fall back to NASA FIRMS auto-fetch."
            )
            for ax in (ax_area, ax_map):
                ax.text(0.5, 0.5, msg,
                        ha="center", va="center", color=TEXT_DIM, fontsize=9,
                        transform=ax.transAxes, linespacing=1.6)
            self._fig_fired.tight_layout(pad=2.0)
            self._canvas_fired.draw()
            return

        terrain_bbox = self._result.get("terrain_bbox")  # (lat_min, lat_max, lon_min, lon_max)

        # ── Left panel: cumulative burn area over time ─────────────────────
        from pipeline.fired_loader import fired_polygon_to_grid_mask as _f2g
        import pandas as _pd

        dates    = fired_tl["burn_date"].dt.normalize().unique()
        dates    = sorted(dates)
        cum_ha   = []
        cum_mask = np.zeros(land.shape, dtype=bool)
        for d in dates:
            day_rows  = fired_tl[fired_tl["burn_date"].dt.normalize() == d]
            if terrain_bbox is not None:
                day_mask  = _f2g(day_rows, terrain_bbox, land.shape)
                cum_mask |= day_mask
            cum_ha.append(float(cum_mask.sum()) * cell_m**2 / 10_000)

        # Convert to hours after ignition for x-axis
        if ign_time is not None:
            x_hours = [(d.tz_localize("UTC") if d.tzinfo is None else d - ign_time).total_seconds() / 3600
                        if ign_time is not None else i * 24
                        for i, d in enumerate(dates)]
            try:
                x_hours = [(_pd.Timestamp(d).tz_localize("UTC") if _pd.Timestamp(d).tzinfo is None
                             else _pd.Timestamp(d) - ign_time).total_seconds() / 3600
                            for d in dates]
            except Exception:
                x_hours = list(range(len(dates)))
            x_label = "Hours after ignition"
        else:
            x_hours = list(range(len(dates)))
            x_label = "Day index"

        ax_area.plot(x_hours, cum_ha, color="#a6e3a1", lw=2.0,
                     marker="o", ms=5, label="FIRED cumulative")
        ax_area.fill_between(x_hours, cum_ha, alpha=0.25, color="#a6e3a1")

        # Overlay model prediction growth
        if self._snapshots:
            import config as _cfg2
            dt = getattr(_cfg2, "dt", 1.0)
            hindcast_steps = self._result.get("hindcast_steps", len(self._snapshots))
            snap_every = max(1, hindcast_steps // 80)
            model_t  = [i * snap_every * dt / 60.0 for i in range(len(self._snapshots))]
            model_ha = [float(np.sum(s > 0)) * cell_m**2 / 10_000 for s in self._snapshots]
            ax_area.plot(model_t, model_ha, color=ACCENT, lw=1.5, ls="--",
                         alpha=0.8, label="Model prediction")

        ax_area.set_xlabel(x_label, color=TEXT_DIM, fontsize=9)
        ax_area.set_ylabel("Burn area (ha)", color=TEXT_DIM, fontsize=9)
        ax_area.set_title("FIRED Cumulative Burn Area", color=TEXT, fontsize=10)
        ax_area.legend(fontsize=8)
        ax_area.spines["bottom"].set_color(TEXT_DIM)
        ax_area.spines["left"].set_color(TEXT_DIM)

        # ── Right panel: daily perimeter growth as colour gradient ─────────
        ax_map.imshow(land.elevation, cmap="gray", origin="lower", alpha=0.45)
        ax_map.plot(c0, r0, "y*", ms=12, markeredgecolor="white",
                    markeredgewidth=0.7, label="Ignition", zorder=5)

        n_days = len(dates)
        cmap_fire = plt.get_cmap("YlOrRd")   # yellow (early) → red (late)

        for day_i, d in enumerate(dates):
            if terrain_bbox is None:
                break
            day_rows = fired_tl[fired_tl["burn_date"].dt.normalize() == d]
            try:
                day_mask = _f2g(day_rows, terrain_bbox, land.shape)
            except Exception:
                continue
            if not day_mask.any():
                continue
            colour = cmap_fire(day_i / max(n_days - 1, 1))
            # Draw boundary contour for each day's polygon
            ax_map.contour(day_mask.astype(float), levels=[0.5],
                           colors=[colour], linewidths=1.5,
                           origin="lower", linestyles="solid", alpha=0.9)

        # Colour bar legend for day progression
        sm = plt.cm.ScalarMappable(cmap=cmap_fire,
                                   norm=plt.Normalize(vmin=0, vmax=n_days - 1))
        sm.set_array([])
        cbar = self._fig_fired.colorbar(sm, ax=ax_map, fraction=0.046, pad=0.02)
        cbar.set_label("Day index (0 = earliest)", color=TEXT_DIM, fontsize=8)
        cbar.ax.tick_params(colors=TEXT_DIM, labelsize=7)

        # Final FIRED mask filled in semi-transparent orange
        if terrain_bbox is not None and cum_mask.any():
            ov_fired = np.zeros((*land.shape, 4))
            ov_fired[cum_mask] = [1.0, 0.5, 0.0, 0.30]
            ax_map.imshow(ov_fired, origin="lower", zorder=2)

        date_str_start = _pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
        date_str_end   = _pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
        ax_map.set_title(
            f"FIRED Daily Perimeter Growth\n"
            f"{date_str_start} → {date_str_end}  ({n_days} days)",
            color=TEXT, fontsize=10
        )
        ax_map.legend(fontsize=8, loc="lower right")
        ax_map.set_xlabel("Grid column (W→E)", color=TEXT_DIM, fontsize=8)
        ax_map.set_ylabel("Grid row (S→N)", color=TEXT_DIM, fontsize=8)

        truth_ha = float(cum_mask.sum()) * cell_m**2 / 10_000
        self._fig_fired.suptitle(
            f"FIRED Event  |  Total burned area: {truth_ha:.1f} ha  |  {n_days} daily snapshots",
            color=TEXT_DIM, fontsize=9
        )
        self._fig_fired.tight_layout(pad=2.0)
        self._canvas_fired.draw()

    # ── 3D Earth View tab ─────────────────────────────────────────────────────
    def _build_earth3d_tab(self, frm):
        """
        Launches a standalone PyVista window: high-res terrain mesh +
        animated red/black fire overlay.  No in-process matplotlib rendering
        so the main GUI never lags.
        """
        # Centre card
        card = tk.Frame(frm, bg=BG_PANEL)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(card, text="🌋  Interactive 3D Fire Terrain",
                 font=("Segoe UI", 16, "bold"), fg=ACCENT2, bg=BG_PANEL
                 ).pack(pady=(30, 6))
        tk.Label(card,
                 text=(
                     "Opens a separate PyVista window with the full-resolution\n"
                     "terrain mesh and animated fire spread.\n\n"
                     "Controls:  Space = play/pause   ← → = step   Q = quit"
                 ),
                 font=FONT_MAIN, fg=TEXT_DIM, bg=BG_PANEL, justify="center"
                 ).pack(pady=(0, 20))

        self._btn_earth3d_render = tk.Button(
            card, text="🌍  Launch 3D Viewer",
            font=("Segoe UI", 13, "bold"), bg="#313244", fg=ACCENT2,
            relief="flat", bd=0, cursor="hand2", padx=24, pady=10,
            command=self._launch_earth3d_pyvista, state="disabled"
        )
        self._btn_earth3d_render.pack(pady=4)

        tk.Label(card, text="(Run the hindcast first to enable)",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_PANEL).pack(pady=(4, 0))

    def _launch_earth3d_pyvista(self):
        """Save terrain + fire snapshots to a temp .npz and launch fire_viewer_3d.py."""
        if self._result is None:
            return

        import tempfile, subprocess, sys as _sys
        import config as _cfg

        land   = self._result["landscape"]
        r0, c0 = self._result["ignition_rc"]
        cell_m = getattr(land.config if hasattr(land, "config") else _cfg,
                         "CELL_SIZE_METERS", 278)

        snaps = np.stack(self._snapshots, axis=0).astype(np.uint8) if self._snapshots \
                else np.zeros((1, *land.elevation.shape), dtype=np.uint8)

        # Wind: use per-cell arrays if available, else scalar broadcast
        if hasattr(land, "wind_u") and getattr(land, "wind_u", None) is not None \
                and np.asarray(land.wind_u).shape == land.elevation.shape:
            wu = land.wind_u.astype(np.float32)
            wv = land.wind_v.astype(np.float32)
        else:
            _w   = self._result.get("weather", {})
            _spd = float(_w.get("wind_speed_ms", 0.0))
            _dir = float(_w.get("wind_direction", 0.0))
            _ang = np.radians(270.0 - _dir)
            wu = np.float32(_spd * np.cos(_ang))
            wv = np.float32(_spd * np.sin(_ang))

        texture_path = self._result.get("texture_path", "")
        if texture_path is None:
            texture_path = ""

        # Write payload
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez_compressed(
            tmp.name,
            elevation=land.elevation.astype(np.float32),
            snapshots=snaps,
            ignition_rc=np.array([r0, c0]),
            cell_size_m=np.float32(cell_m),
            wind_u=wu,
            wind_v=wv,
            texture_path=np.bytes_(str(texture_path).encode()),
        )
        tmp.close()

        viewer_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "viz", "fire_viewer_3d.py"
        )
        subprocess.Popen([_sys.executable, viewer_script, tmp.name])

    # ── Analysis tab ──────────────────────────────────────────────────────────
    def _build_analysis_tab(self, frm):
        self._fig_analysis, self._ax_analysis = plt.subplots(1, 3, figsize=(13, 4))
        self._fig_analysis.patch.set_facecolor("#1e1e2e")
        for ax in self._ax_analysis:
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(TEXT_DIM)
        self._fig_analysis.tight_layout(pad=2.5)
        self._canvas_analysis = FigureCanvasTkAgg(self._fig_analysis, master=frm)
        self._canvas_analysis.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas_analysis, frm).pack(fill="x")

    # ── Presets ────────────────────────────────────────────────────────────────
    _PRESETS = {
        # Rhodes July 2023 — VIIRS ignition, 6-hour hindcast
        "Rhodes 2023": dict(
            lat_min="35.80", lat_max="36.50",
            lon_min="27.00", lon_max="28.50",
            date_start="2023-07-19", date_end="2023-07-22",
        ),
        # Northern Evia megafire August 2021
        # FIRED event id = 2871   centre ≈ (38.83°N, 23.20°E)
        # FIRED window: 2021-08-01 → 2021-08-13  (12 days, ~50 000 ha)
        "Evoia 2021":  dict(
            lat_min="38.50", lat_max="39.10",
            lon_min="22.70", lon_max="23.70",
            date_start="2021-08-01", date_end="2021-08-13",
        ),
    }

    def _apply_preset(self, _e=None):
        p = self._PRESETS.get(self._preset_var.get())
        if not p:
            return
        self.v_lat_min.set(p["lat_min"]); self.v_lat_max.set(p["lat_max"])
        self.v_lon_min.set(p["lon_min"]); self.v_lon_max.set(p["lon_max"])
        self.v_date_start.set(p["date_start"]); self.v_date_end.set(p["date_end"])

    def _browse_shapefile(self):
        """Open a file-chooser for a Copernicus EMS .shp or .geojson file."""
        path = filedialog.askopenfilename(
            title="Select Copernicus EMS fire perimeter",
            filetypes=[
                ("Shapefiles & GeoJSON", "*.shp *.geojson *.json"),
                ("Shapefile", "*.shp"),
                ("GeoJSON", "*.geojson *.json"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.v_truth_shp.set(path)

    def _browse_fired_gpkg(self):
        """Open a file-chooser for a FIRED daily GeoPackage (.gpkg)."""
        path = filedialog.askopenfilename(
            title="Select FIRED daily perimeter GeoPackage",
            filetypes=[
                ("GeoPackage", "*.gpkg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.v_fired_gpkg.set(path)

    # ── Step status helpers ────────────────────────────────────────────────────
    def _set_step(self, idx, status):
        self._step_status[idx] = status
        icon_var, icon_lbl = self._step_labels[idx]
        self.after(0, lambda iv=icon_var, il=icon_lbl, s=status: (
            iv.set(_ICON[s]),
            il.configure(fg=_ICON_CLR[s])
        ))

    def _reset_steps(self):
        for i in range(len(STEP_NAMES)):
            self._set_step(i, "pending")

    # ── Validation ─────────────────────────────────────────────────────────────
    def _validate(self):
        errs = []
        try:
            if float(self.v_lat_min.get()) >= float(self.v_lat_max.get()):
                errs.append("Lat Min must be < Lat Max")
            if float(self.v_lon_min.get()) >= float(self.v_lon_max.get()):
                errs.append("Lon Min must be < Lon Max")
        except ValueError:
            errs.append("Lat/Lon must be numbers")
        try:
            datetime.datetime.strptime(self.v_date_start.get(), "%Y-%m-%d")
            datetime.datetime.strptime(self.v_date_end.get(),   "%Y-%m-%d")
        except ValueError:
            errs.append("Dates must be YYYY-MM-DD")
        try:
            if float(self.v_hindcast_hrs.get()) <= 0:
                errs.append("Hindcast hours must be > 0")
            if float(self.v_terrain_buf.get()) <= 0:
                errs.append("Terrain buffer must be > 0")
            if int(self.v_maxiter.get()) < 1:
                errs.append("Max generations must be >= 1")
            if int(self.v_popsize.get()) < 5:
                errs.append("Population size must be >= 5")
        except ValueError:
            errs.append("Numeric fields are invalid")
        if errs:
            messagebox.showerror("Invalid Input", "\n".join(f"- {e}" for e in errs))
            return False
        return True

    # ── Run ────────────────────────────────────────────────────────────────────
    def _run(self):
        if self._running or not self._validate():
            return
        self._running = True
        self._result  = None
        self._reset_steps()
        self._clear_log_widget()
        self._conv_evals.clear()
        self._conv_errors.clear()
        self._conv_best.clear()
        self._snapshots.clear()

        self.btn_run.configure(state="disabled", text="Running...")
        self.btn_play.configure(state="disabled")
        self.btn_pause.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Pipeline running...")
        self.notebook.select(0)

        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        redir = _TextRedirector(self.log, self.after)
        sys.stdout = redir
        sys.stderr = redir

        def _eval_cb(eval_num, error, best_error, params):
            self._conv_evals.append(eval_num)
            self._conv_errors.append(error)
            self._conv_best.append(best_error)
            if eval_num % 10 == 0:
                self.after(0, self._update_convergence_chart)

        def worker():
            try:
                from pipeline.hindcast_optimizer import run_hindcast, plot_results

                # Ground-truth priority: Copernicus .shp > FIRED .gpkg > FIRMS
                shp_path  = self.v_truth_shp.get().strip()
                truth_shp = shp_path if shp_path and os.path.exists(shp_path) else None

                gpkg_path  = self.v_fired_gpkg.get().strip()
                fired_gpkg = gpkg_path if gpkg_path and os.path.exists(gpkg_path) else None

                if truth_shp:
                    print(f"[GUI] Ground truth: Copernicus shapefile → {truth_shp}")
                elif fired_gpkg:
                    print(f"[GUI] Ground truth: FIRED GeoPackage → {fired_gpkg}")
                else:
                    print("[GUI] Ground truth: NASA FIRMS (auto-fetched)")

                result = run_hindcast(
                    map_key         = self.v_key.get().strip(),
                    lat_min         = float(self.v_lat_min.get()),
                    lat_max         = float(self.v_lat_max.get()),
                    lon_min         = float(self.v_lon_min.get()),
                    lon_max         = float(self.v_lon_max.get()),
                    date_start      = self.v_date_start.get().strip(),
                    date_end        = self.v_date_end.get().strip(),
                    hindcast_hours  = float(self.v_hindcast_hrs.get()),
                    maxiter         = int(self.v_maxiter.get()),
                    popsize         = int(self.v_popsize.get()),
                    terrain_buffer  = float(self.v_terrain_buf.get()),
                    eval_callback   = _eval_cb,
                    truth_shapefile = truth_shp,
                    fired_gpkg      = fired_gpkg,
                )
                self._result = result

                if self.v_save_plot.get():
                    out = str(PROJECT_ROOT / "hindcast_result.png")
                    plot_results(result, output_path=out)

                self._build_snapshots(result)
                self.after(0, self._on_success)

            except Exception as exc:
                traceback.print_exc()
                self.after(0, self._on_error, str(exc))
            finally:
                sys.stdout = self._old_stdout
                sys.stderr = self._old_stderr

        threading.Thread(target=worker, daemon=True).start()

    # ── Replay the best simulation and capture frame snapshots ────────────────
    def _build_snapshots(self, result):
        """
        Re-run the best-fit simulation at HIGH resolution for smooth animation.
        Uses land_snap.config.CELL_SIZE_METERS (dynamically set by load_real_terrain)
        rather than the stale global cfg.CELL_SIZE_METERS.
        """
        try:
            from core.fire_model import CellularAutomataFire
            import copy as _copy

            land         = result["landscape"]
            best         = result["best_params"]
            ignition_rc  = result["ignition_rc"]
            hindcast_steps = result.get("hindcast_steps", 600)

            # Shallow copy avoids the "cannot pickle module" error
            # (Landscape.config is a live module object).
            # We deep-copy only the mutable numpy arrays we will modify.
            land_snap = _copy.copy(land)
            land_snap.elevation   = land.elevation.copy()
            land_snap.fuel_map    = land.fuel_map.copy()
            land_snap.moisture    = np.clip(
                land.moisture + best["fuel_moisture_offset"], 0.01, 0.30
            )
            land_snap.set_wind(land.wind_speed * best["wind_multiplier"],
                               land.wind_dir)

            sim  = CellularAutomataFire(land_snap, land_snap.config)
            r0, c0 = ignition_rc
            # Use land_snap.config — correctly reflects the resampled cell size
            cell_m = land_snap.config.CELL_SIZE_METERS
            radius = max(1, int(375 / cell_m / 2))
            rows_g, cols_g = land_snap.shape
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < rows_g and 0 <= cc < cols_g:
                        sim.ignite(rr, cc)

            snaps = []
            snap_every = max(1, hindcast_steps // 80)
            for step in range(hindcast_steps):
                sim.step()
                if step % snap_every == 0:
                    snaps.append(sim.state.copy())

            self._snapshots = snaps
            burned_cells = int(np.sum(sim.state > 0))
            burned_ha    = burned_cells * cell_m**2 / 10_000
            print(f"[Animation] Captured {len(snaps)} frames  "
                  f"({burned_ha:.1f} ha burned at {cell_m:.0f} m/cell).")
        except Exception as exc:
            print(f"[Animation] Skipped: {exc}")
            import traceback; traceback.print_exc()

    # ── Success / error callbacks ──────────────────────────────────────────────
    def _on_success(self):
        self._running = False
        self.progress.stop()
        self.btn_run.configure(state="normal", text="RUN ALL STEPS")

        iou = self._result.get("iou", 0) * 100
        truth_src = self._result.get("truth_source", "firms_viirs")
        src_tag = {"copernicus": "Copernicus", "fired": "FIRED", "firms_viirs": "FIRMS"}.get(
            truth_src, "FIRMS"
        )
        self.status_var.set(f"Done!  IoU = {iou:.1f}%  [{src_tag}]")

        for i in range(len(STEP_NAMES)):
            self._set_step(i, "done")

        self._update_terrain_tab()
        self._update_analysis_tab()
        self._update_fire_tab()
        self._update_hillfire_tab()
        self._update_compare_tab()
        self._update_fired_tab()

        # Enable 3D render button now that data is available
        if hasattr(self, "_btn_earth3d_render"):
            self._btn_earth3d_render.configure(state="normal")

        # Enable 3D view button now that snapshots are ready
        if hasattr(self, "btn_3d"):
            self.btn_3d.configure(state="normal")

        if self.v_open_plot.get():
            path = str(PROJECT_ROOT / "hindcast_result.png")
            if os.path.exists(path):
                _open_path(path)

    def _on_error(self, msg: str):
        self._running = False
        self.progress.stop()
        self.btn_run.configure(state="normal", text="RUN ALL STEPS")
        self.status_var.set(f"Error: {msg[:100]}")
        messagebox.showerror("Run Failed", f"{msg}\n\nSee Log tab for full traceback.")

    # ── 3-D PyVista view ───────────────────────────────────────────────────────
    def _launch_3d_view(self):
        """Save animation data to .npz and launch visualizer_3d.py in a subprocess."""
        if not self._snapshots or self._result is None:
            messagebox.showinfo("3D View", "Run the pipeline first to generate snapshots.")
            return

        import subprocess
        land     = self._result["landscape"]
        ign_rc   = self._result.get("ignition_rc", None)
        cell_m   = land.config.CELL_SIZE_METERS if hasattr(land, "config") else 280.0
        ws       = getattr(land, "wind_speed", 0.0)
        wd       = getattr(land, "wind_dir",   0.0)
        import config as _cfg
        dt_min   = getattr(_cfg, "dt", 1.0)

        snaps_arr = np.array(self._snapshots, dtype=np.int8)

        # ── Optional satellite/CORINE texture ────────────────────────────────
        texture_path = self._result.get("texture_path")
        texture_arr  = None
        if texture_path and os.path.exists(str(texture_path)):
            try:
                from PIL import Image as _PIL
                _img = _PIL.open(texture_path).convert("RGB")
                # Flip top-to-bottom so row-0 = South (matches landscape convention)
                _img = _img.transpose(_PIL.FLIP_TOP_BOTTOM)
                texture_arr = np.array(_img, dtype=np.uint8)   # (H, W, 3)
                print(f"[3D View] Texture loaded: {Path(texture_path).name} "
                      f"({texture_arr.shape[1]}×{texture_arr.shape[0]} px)")
            except Exception as _e:
                print(f"[3D View] Texture load failed ({_e}) — skipping.")

        npz_path = str(PROJECT_ROOT / "viz3d_data.npz")
        save_kwargs = dict(
            elevation    = land.elevation,
            fuel_map     = land.fuel_map.astype(np.float32),
            snapshots    = snaps_arr,
            cell_size_m  = np.array([cell_m]),
            wind_speed   = np.array([ws]),
            wind_dir     = np.array([wd]),
            dt_minutes   = np.array([dt_min]),
        )
        if ign_rc is not None:
            save_kwargs["ignition_rc"] = np.array(ign_rc)
        if texture_arr is not None:
            save_kwargs["satellite_texture"] = texture_arr

        np.savez(npz_path, **save_kwargs)
        print(f"[3D View] Saved {snaps_arr.shape[0]} frames → {npz_path}")

        # Fix 1: look for visualizer_3d.py in the project root first, then viz/
        viz_script_root = PROJECT_ROOT / "visualizer_3d.py"
        viz_script_sub  = PROJECT_ROOT / "viz" / "visualizer_3d.py"
        if viz_script_root.exists():
            viz_script = str(viz_script_root)
        elif viz_script_sub.exists():
            viz_script = str(viz_script_sub)
        else:
            messagebox.showerror("3D Viewer", "visualizer_3d.py not found in project root or viz/")
            return
        subprocess.Popen([sys.executable, viz_script, npz_path])
        self.status_var.set("3D viewer launched — check new window")

    # ── Terrain tab ────────────────────────────────────────────────────────────
    def _update_terrain_tab(self):
        if self._result is None:
            return
        land    = self._result["landscape"]
        truth   = self._result["truth_mask"]
        pred    = self._result["predicted_mask"]
        r0, c0  = self._result["ignition_rc"]
        weather = self._result["weather"]
        iou     = self._result.get("iou", 0) * 100

        axes = self._ax_terrain
        for ax in axes.flat:
            ax.cla()
            ax.set_facecolor("#11111b")
            ax.tick_params(colors=TEXT_DIM, labelsize=7)

        # DEM
        im = axes[0, 0].imshow(land.elevation, cmap="terrain", origin="lower")
        axes[0, 0].plot(c0, r0, "r*", ms=12, label="Ignition")
        axes[0, 0].set_title("Elevation (m)", color=TEXT, fontsize=9)
        axes[0, 0].legend(fontsize=7)
        self._fig_terrain.colorbar(im, ax=axes[0, 0], fraction=0.046)

        # Fuel / CORINE
        n = len(land.fuel_names)
        cmap_f = ListedColormap(FUEL_COLORS[:n])
        axes[0, 1].imshow(land.fuel_map, cmap=cmap_f, vmin=0, vmax=n - 1, origin="lower")
        axes[0, 1].plot(c0, r0, "r*", ms=12)
        axes[0, 1].set_title("CORINE Fuel Map", color=TEXT, fontsize=9)
        patches = [mpatches.Patch(color=FUEL_COLORS[i], label=land.fuel_names[i])
                   for i in range(n)]
        axes[0, 1].legend(handles=patches, fontsize=5, loc="lower right")

        # Predicted vs truth
        axes[1, 0].imshow(land.elevation, cmap="gray", origin="lower", alpha=0.4)
        ov_t = np.zeros((*land.shape, 4)); ov_t[truth] = [1, 0, 0, 0.6]
        ov_p = np.zeros((*land.shape, 4)); ov_p[pred]  = [0, 0.5, 1, 0.4]
        axes[1, 0].imshow(ov_t, origin="lower")
        axes[1, 0].imshow(ov_p, origin="lower")
        axes[1, 0].plot(c0, r0, "y*", ms=10)
        axes[1, 0].set_title("Predicted (blue) vs Truth (red)", color=TEXT, fontsize=9)
        axes[1, 0].set_xlabel(f"IoU = {iou:.1f}%", color=TEXT_DIM, fontsize=8)

        # TP/FP/FN breakdown
        tp = pred & truth; fp = pred & ~truth; fn = ~pred & truth
        ov = np.zeros((*land.shape, 4))
        ov[tp] = [0, 0.8, 0, 0.8]
        ov[fp] = [0, 0.3, 1, 0.6]
        ov[fn] = [1, 0,   0, 0.8]
        axes[1, 1].imshow(land.elevation, cmap="gray", origin="lower", alpha=0.4)
        axes[1, 1].imshow(ov, origin="lower")
        axes[1, 1].plot(c0, r0, "y*", ms=10)
        axes[1, 1].legend(handles=[
            mpatches.Patch(color=[0, 0.8, 0, 0.8], label=f"True Pos  {tp.sum()}"),
            mpatches.Patch(color=[0, 0.3, 1, 0.6], label=f"False Pos {fp.sum()}"),
            mpatches.Patch(color=[1, 0,   0, 0.8], label=f"False Neg {fn.sum()}"),
        ], fontsize=7)
        axes[1, 1].set_title(f"TP/FP/FN  (IoU={iou:.1f}%)", color=TEXT, fontsize=9)

        self._fig_terrain.suptitle(
            f"ERA5: {weather['wind_speed_ms']:.1f} m/s @ {weather['wind_direction']:.0f} deg  "
            f"T={weather['temperature_c']:.1f}C  RH={weather['relative_humidity']:.0f}%",
            color=TEXT_DIM, fontsize=9
        )
        self._fig_terrain.tight_layout(pad=2.0)
        self._canvas_terrain.draw()
        self.notebook.select(self._tab_terrain)

    # ── Fire animation tab ─────────────────────────────────────────────────────
    def _update_fire_tab(self):
        if not self._snapshots:
            return
        self._frame_slider.configure(to=len(self._snapshots) - 1)
        self._frame_var.set(0)
        self._draw_fire_frame(0)
        self.btn_play.configure(state="normal")
        self.btn_pause.configure(state="normal")

    def _draw_fire_frame(self, idx):
        if not self._snapshots or idx >= len(self._snapshots):
            return
        land    = self._result["landscape"]
        r0, c0  = self._result["ignition_rc"]
        snap    = self._snapshots[idx]

        ax = self._ax_fire
        ax.cla()
        ax.set_facecolor("#11111b")

        # Use LightSource hillshade (same as Hillshade tab) instead of flat colormap
        from matplotlib.colors import LightSource
        ls    = LightSource(azdeg=315, altdeg=45)
        shade = ls.hillshade(land.elevation.astype(float), vert_exag=2, dx=1.0, dy=1.0)
        ax.imshow(shade, cmap="gray", origin="lower", vmin=0, vmax=1, alpha=0.85)
        ax.imshow(land.elevation, cmap="terrain", origin="lower", alpha=0.35)

        fire_rgba = np.zeros((*snap.shape, 4))
        fire_rgba[snap == 1] = [1.0, 0.3, 0.0, 0.85]
        fire_rgba[snap == 2] = [0.3, 0.1, 0.0, 0.60]
        ax.imshow(fire_rgba, origin="lower")
        ax.plot(c0, r0, "y*", ms=10)

        hindcast_steps = self._result.get("hindcast_steps", len(self._snapshots))
        snap_every = max(1, hindcast_steps // 80)
        import config as _cfg
        t_min = idx * snap_every * getattr(_cfg, "dt", 1.0)

        # Update fire-time clock
        ign_time = self._result.get("ignition_time")
        if ign_time is not None:
            fire_dt = ign_time + datetime.timedelta(minutes=t_min)
            self._fire_time_var.set(f"🔥 {fire_dt.strftime('%Y-%m-%d %H:%M UTC')}  (+{t_min:.0f} min)")
        else:
            self._fire_time_var.set(f"🔥 t = {t_min:.0f} min")

        ax.set_title(
            f"Fire Spread  t ~ {t_min:.0f} min  |  frame {idx + 1}/{len(self._snapshots)}",
            color=TEXT, fontsize=9
        )
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        self._canvas_fire.draw()

    def _seek_frame(self, val):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None
        self._draw_fire_frame(int(float(val)))

    def _play_anim(self):
        if self._anim_job:
            return
        self._anim_frame = int(self._frame_var.get())
        self._tick_anim()

    def _pause_anim(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _tick_anim(self):
        if not self._snapshots:
            return
        if self._anim_frame >= len(self._snapshots):
            self._anim_frame = 0
        self._frame_var.set(self._anim_frame)
        self._draw_fire_frame(self._anim_frame)
        self._anim_frame += 1
        self._anim_job = self.after(self._anim_speed.get(), self._tick_anim)

    # ── Analysis tab ──────────────────────────────────────────────────────────
    def _update_analysis_tab(self):
        self._update_convergence_chart()
        if self._result is None:
            return
        pred  = self._result["predicted_mask"]
        truth = self._result["truth_mask"]

        # Panel 1: TP/FP/FN bar chart
        ax = self._ax_analysis[1]
        ax.cla(); ax.set_facecolor("#11111b"); ax.tick_params(colors=TEXT_DIM, labelsize=8)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        bars = ax.bar(["True Pos", "False Pos", "False Neg"],
                      [tp, fp, fn],
                      color=[GREEN, ACCENT2, ACCENT])
        ax.set_title("Prediction Breakdown", color=TEXT, fontsize=9)
        ax.set_ylabel("Grid cells", color=TEXT_DIM, fontsize=8)
        for bar, val in zip(bars, [tp, fp, fn]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", color=TEXT, fontsize=8)

        # Panel 2: Burn area growth from snapshots
        ax2 = self._ax_analysis[2]
        ax2.cla(); ax2.set_facecolor("#11111b"); ax2.tick_params(colors=TEXT_DIM, labelsize=8)
        if self._snapshots:
            frames = list(range(len(self._snapshots)))
            burned = [int(np.sum(s > 0)) for s in self._snapshots]
            ax2.plot(frames, burned, color=ACCENT, linewidth=1.5)
            ax2.fill_between(frames, burned, alpha=0.2, color=ACCENT)
            ax2.set_xlabel("Frame", color=TEXT_DIM, fontsize=8)
            ax2.set_ylabel("Burned cells", color=TEXT_DIM, fontsize=8)
            ax2.set_title("Burn Area Growth", color=TEXT, fontsize=9)

        self._fig_analysis.tight_layout(pad=2.5)
        self._canvas_analysis.draw()

    def _update_convergence_chart(self):
        if not self._conv_evals:
            return
        ax = self._ax_analysis[0]
        ax.cla(); ax.set_facecolor("#11111b"); ax.tick_params(colors=TEXT_DIM, labelsize=8)
        ax.plot(self._conv_evals, self._conv_errors, ".",
                color=ACCENT2, markersize=3, alpha=0.5, label="Eval")
        ax.plot(self._conv_evals, self._conv_best, "-",
                color=ACCENT, linewidth=1.5, label="Best")
        ax.set_xlabel("Evaluation", color=TEXT_DIM, fontsize=8)
        ax.set_ylabel("Error", color=TEXT_DIM, fontsize=8)
        ax.set_title("Optimizer Convergence", color=TEXT, fontsize=9)
        ax.legend(fontsize=7)
        self._fig_analysis.tight_layout(pad=2.5)
        self._canvas_analysis.draw()

    # ── Utility ────────────────────────────────────────────────────────────────
    def _clear_log_widget(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")

    def _clear_all(self):
        self._clear_log_widget()
        self._reset_steps()
        self.status_var.set("Ready")
        for ax in self._ax_terrain.flat:
            ax.cla()
        self._canvas_terrain.draw()
        self._ax_fire.cla()
        self._canvas_fire.draw()
        self._ax_hill.cla()
        self._hillshade_cache = None
        self._canvas_hill.draw()
        for ax in self._ax_analysis:
            ax.cla()
        self._canvas_analysis.draw()

    def _on_close(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
        if self._hill_anim_job:
            self.after_cancel(self._hill_anim_job)
        plt.close("all")
        self.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Tooltip widget
# ──────────────────────────────────────────────────────────────────────────────

class _ToolTip:
    def __init__(self, widget, text):
        self._w   = widget
        self._t   = text
        self._tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e=None):
        x = self._w.winfo_rootx() + 24
        y = self._w.winfo_rooty() + 24
        self._tip = tw = tk.Toplevel(self._w)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._t, font=("Segoe UI", 9),
                 bg="#313150", fg="#cdd6f4",
                 relief="solid", bd=1, padx=6, pady=4,
                 wraplength=260).pack()

    def _hide(self, _e=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = WilsonGUI()
    app.mainloop()

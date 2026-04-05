import pyvista as pv
import numpy as np
from matplotlib.colors import ListedColormap
import vtk
import math


class Visualizer3D:
    """
    3-D fire-spread visualiser built on PyVista.

    Controls
    --------
    P  — pause / resume
    R  — reset to frame 0 and play again
    """

    _IDX_BURNING = 6
    _IDX_BURNED  = 7

    def __init__(self, model):
        self.model   = model
        self.history = []

        rows, cols = self.model.landscape.shape
        cs = self.model.config.CELL_SIZE_METERS

        xx, yy = np.meshgrid(
            np.arange(cols, dtype=float) * cs,
            np.arange(rows, dtype=float) * cs,
        )
        elev = self.model.landscape.elevation.astype(float)

        self._base_display = self.model.landscape.fuel_map.copy().astype(float)

        # ----------------------------------------------------------------
        # Build vtkStructuredGrid directly in VTK
        # ----------------------------------------------------------------
        # CRITICAL FIX: Change from order="F" to order="C"
        flat_x = xx.flatten(order="C")
        flat_y = yy.flatten(order="C")
        flat_z = elev.flatten(order="C")

        pts = vtk.vtkPoints()
        for x_, y_, z_ in zip(flat_x, flat_y, flat_z):
            pts.InsertNextPoint(x_, y_, z_)

        self._vtk_grid = vtk.vtkStructuredGrid()
        self._vtk_grid.SetDimensions(cols, rows, 1)
        self._vtk_grid.SetPoints(pts)

        self._scalars = vtk.vtkFloatArray()
        self._scalars.SetName("DisplayColor")
        
        # CRITICAL FIX: Change order="F" to order="C"
        for v in self._base_display.flatten(order="C"):
            self._scalars.InsertNextValue(float(v))
        self._vtk_grid.GetPointData().SetScalars(self._scalars)

        # ----------------------------------------------------------------
        # Colormap → vtkLookupTable
        # ----------------------------------------------------------------
        cmap = ListedColormap([
            "#d9f0a3",  # 0  sparse grass
            "#238b45",  # 1  dense forest
            "#a1d99b",  # 2  light shrub
            "#74c476",  # 3  medium shrub
            "#00441b",  # 4  heavy canopy
            "#addd8e",  # 5  mixed / other
            "#ff4500",  # 6  burning
            "#2a2a2a",  # 7  ash
        ])

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(8)
        lut.SetRange(0, 7)
        for i in range(8):
            r, g, b, a = cmap(i / 7.0)
            lut.SetTableValue(i, r, g, b, a)
        lut.Build()

        self._mapper = vtk.vtkDataSetMapper()
        self._mapper.SetInputData(self._vtk_grid)
        self._mapper.SetScalarRange(0, 7)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()

        terrain_actor = vtk.vtkActor()
        terrain_actor.SetMapper(self._mapper)

        # ----------------------------------------------------------------
        # Plotter
        # ----------------------------------------------------------------
        self.plotter = pv.Plotter()
        self.plotter.background_color = "#87CEEB"
        self.plotter.renderer.AddActor(terrain_actor)

        # ----------------------------------------------------------------
        # Wind compass overlay (2-D renderer on top)
        # ----------------------------------------------------------------
        self._build_compass()

        # ----------------------------------------------------------------
        # Playback state
        # ----------------------------------------------------------------
        self._frame_idx   = 0
        self._paused      = False
        self._interval_ms = 33

        self.plotter.add_key_event("p", self._toggle_pause)
        self.plotter.add_key_event("r", self._reset_playback)

        self.plotter.add_text(
            "P = pause/resume     R = reset",
            position="lower_left",
            font_size=10,
            color="white",
            name="hint",
        )
        self._update_status("Ready — opening window...")

    # ------------------------------------------------------------------
    # Compass — built as a 2-D overlay renderer so it always faces the
    # camera and sits in the corner regardless of how the user orbits.
    # ------------------------------------------------------------------

    def _build_compass(self):
        """
        Builds the compass using absolute pixel coordinates.
        Fixed VTK list-to-integer TypeError.
        """
        wind_u = float(self.model.landscape.wind_u)
        wind_v = float(self.model.landscape.wind_v)
        speed  = math.sqrt(wind_u**2 + wind_v**2)
        angle_rad = -math.atan2(wind_v, wind_u)

        # 1. Setup the Layered Renderer
        self._compass_renderer = vtk.vtkRenderer()
        self._compass_renderer.SetViewport(0.78, 0.0, 1.0, 0.30)
        self._compass_renderer.SetLayer(1) 
        self._compass_renderer.InteractiveOff()
        self._compass_renderer.SetBackground(0.08, 0.08, 0.12)
        self._compass_renderer.SetBackgroundAlpha(0.75)

        win = self.plotter.renderer.GetRenderWindow()
        win.SetNumberOfLayers(2)
        win.AddRenderer(self._compass_renderer)

        # --- PIXEL CONSTANTS ---
        CX, CY = 80, 80   
        R = 58            
        tr = 74           
        tick_in = R - 8   
        
        coord_sys = self._viewport_coord() 

        # 2. Outer Circle
        circle_pts = vtk.vtkPoints()
        circle_lines = vtk.vtkCellArray()
        N_SEG = 64
        for i in range(N_SEG):
            a = 2 * math.pi * i / N_SEG
            circle_pts.InsertNextPoint(CX + R * math.cos(a), CY + R * math.sin(a), 0.0)
        
        circle_lines.InsertNextCell(N_SEG + 1)
        for i in range(N_SEG): 
            circle_lines.InsertCellPoint(i)
        circle_lines.InsertCellPoint(0)

        circle_pd = vtk.vtkPolyData()
        circle_pd.SetPoints(circle_pts)
        circle_pd.SetLines(circle_lines)

        circle_mapper = vtk.vtkPolyDataMapper2D()
        circle_mapper.SetInputData(circle_pd)
        circle_mapper.SetTransformCoordinate(coord_sys)

        circle_actor = vtk.vtkActor2D()
        circle_actor.SetMapper(circle_mapper)
        circle_actor.GetProperty().SetColor(0.6, 0.6, 0.6)
        self._compass_renderer.AddActor2D(circle_actor)

        # 3. Cardinal Ticks & Labels
        directions = {"N": math.pi/2, "S": -math.pi/2, "E": 0.0, "W": math.pi}
        for label, ang in directions.items():
            # Ticks (Fixing the TypeError here)
            t_pts = vtk.vtkPoints()
            t_pts.InsertNextPoint(CX + tick_in * math.cos(ang), CY + tick_in * math.sin(ang), 0.0)
            t_pts.InsertNextPoint(CX + R * math.cos(ang), CY + R * math.sin(ang), 0.0)
            
            t_cells = vtk.vtkCellArray()
            t_cells.InsertNextCell(2) # Define a line with 2 points
            t_cells.InsertCellPoint(0) # Point 1
            t_cells.InsertCellPoint(1) # Point 2
            
            t_pd = vtk.vtkPolyData()
            t_pd.SetPoints(t_pts)
            t_pd.SetLines(t_cells)
            
            t_mapper = vtk.vtkPolyDataMapper2D()
            t_mapper.SetInputData(t_pd)
            t_mapper.SetTransformCoordinate(coord_sys)
            
            t_actor = vtk.vtkActor2D()
            t_actor.SetMapper(t_mapper)
            t_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
            self._compass_renderer.AddActor2D(t_actor)

            # Labels
            txt = vtk.vtkTextActor()
            txt.SetInput(label)
            txt.GetTextProperty().SetFontSize(14)
            txt.GetTextProperty().SetBold(True)
            txt.GetTextProperty().SetJustificationToCentered()
            txt.GetTextProperty().SetVerticalJustificationToCentered()
            txt.GetPositionCoordinate().SetCoordinateSystemToViewport()
            txt.GetPositionCoordinate().SetValue(CX + tr * math.cos(ang), CY + tr * math.sin(ang))
            self._compass_renderer.AddActor2D(txt)

        # 4. Wind Arrow (Fixing the TypeError here too)
        arrow_len = R * 0.8
        tip_x, tip_y = CX + arrow_len * math.cos(angle_rad), CY + arrow_len * math.sin(angle_rad)
        tail_x, tail_y = CX - arrow_len * 0.3 * math.cos(angle_rad), CY - arrow_len * 0.3 * math.sin(angle_rad)

        h_len, h_wid = 12, 7
        perp = angle_rad + math.pi/2
        hx1 = tip_x - h_len * math.cos(angle_rad) + h_wid * math.cos(perp)
        hy1 = tip_y - h_len * math.sin(angle_rad) + h_wid * math.sin(perp)
        hx2 = tip_x - h_len * math.cos(angle_rad) - h_wid * math.cos(perp)
        hy2 = tip_y - h_len * math.sin(angle_rad) - h_wid * math.sin(perp)

        arr_pts = vtk.vtkPoints()
        arr_pts.InsertNextPoint(tail_x, tail_y, 0) # 0
        arr_pts.InsertNextPoint(tip_x, tip_y, 0)   # 1
        arr_pts.InsertNextPoint(hx1, hy1, 0)       # 2
        arr_pts.InsertNextPoint(hx2, hy2, 0)       # 3

        arr_cells = vtk.vtkCellArray()
        # Shaft line
        arr_cells.InsertNextCell(2)
        arr_cells.InsertCellPoint(0)
        arr_cells.InsertCellPoint(1)
        # Head triangle
        arr_cells.InsertNextCell(3)
        arr_cells.InsertCellPoint(1)
        arr_cells.InsertCellPoint(2)
        arr_cells.InsertCellPoint(3)

        arr_pd = vtk.vtkPolyData()
        arr_pd.SetPoints(arr_pts)
        arr_pd.SetLines(arr_cells)
        
        arr_mapper = vtk.vtkPolyDataMapper2D()
        arr_mapper.SetInputData(arr_pd)
        arr_mapper.SetTransformCoordinate(coord_sys)
        
        arr_actor = vtk.vtkActor2D()
        arr_actor.SetMapper(arr_mapper)
        arr_actor.GetProperty().SetColor(1.0, 0.3, 0.1)
        self._compass_renderer.AddActor2D(arr_actor)

        # 5. Fixed Center Dot
        dot_pts = vtk.vtkPoints()
        for i in range(16):
            a = 2 * math.pi * i / 16
            dot_pts.InsertNextPoint(CX + 4*math.cos(a), CY + 4*math.sin(a), 0)
        dot_pd = vtk.vtkPolyData()
        dot_pd.SetPoints(dot_pts)
        dot_cells = vtk.vtkCellArray()
        dot_cells.InsertNextCell(16)
        for i in range(16): 
            dot_cells.InsertCellPoint(i)
        dot_pd.SetPolys(dot_cells)

        dot_mapper = vtk.vtkPolyDataMapper2D()
        dot_mapper.SetInputData(dot_pd)
        dot_mapper.SetTransformCoordinate(coord_sys)
        dot_actor = vtk.vtkActor2D()
        dot_actor.SetMapper(dot_mapper)
        dot_actor.GetProperty().SetColor(1, 1, 1)
        self._compass_renderer.AddActor2D(dot_actor)

        # 6. Titles
        speed_txt = vtk.vtkTextActor()
        speed_txt.SetInput(f"{speed:.1f} m/s")
        speed_txt.GetTextProperty().SetFontSize(12)
        speed_txt.GetTextProperty().SetJustificationToCentered()
        speed_txt.GetPositionCoordinate().SetCoordinateSystemToViewport()
        speed_txt.GetPositionCoordinate().SetValue(CX, 15) 
        self._compass_renderer.AddActor2D(speed_txt)

    def _viewport_coord(self):
        """Helper to set mapping to Pixel Viewport coordinates."""
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToViewport()
        return coord

    @staticmethod
    def _norm_coord():
        """Returns a normalised-viewport coordinate object for 2-D mappers."""
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        return coord

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def play_history(self, history: list, fps: float = 30) -> None:
        if not history:
            raise ValueError("history is empty — run the simulation first.")

        self.history      = history
        self._frame_idx   = 0
        self._paused      = False
        self._interval_ms = max(1, int(1000.0 / fps))

        self.plotter.show(interactive_update=True, auto_close=False)

        iren = self.plotter.iren.interactor
        iren.CreateRepeatingTimer(self._interval_ms)
        iren.AddObserver("TimerEvent", self._on_vtk_timer)

        self._update_status(f"Playing  0/{len(self.history)}")
        iren.Start()

    # ------------------------------------------------------------------
    # Timer observer
    # ------------------------------------------------------------------

    def _on_vtk_timer(self, caller, event) -> None:
        if self._paused:
            return
        if self._frame_idx >= len(self.history):
            if self._frame_idx == len(self.history):
                self._update_status("Playback complete  —  R = restart")
                self._frame_idx += 1
            return
        self._render_frame(self._frame_idx)
        self._frame_idx += 1

    # ------------------------------------------------------------------
    # Key callbacks
    # ------------------------------------------------------------------

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._update_status(
                f"PAUSED at frame {self._frame_idx}/{len(self.history)}  —  P = resume"
            )
        else:
            self._update_status(f"Playing  {self._frame_idx}/{len(self.history)}")

    def _reset_playback(self) -> None:
        self._frame_idx = 0
        self._paused    = False
        self._write_scalars(self._base_display.flatten(order="F"))
        self._update_status("Reset — playing from frame 0")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_frame(self, idx: int) -> None:
        state   = self.history[idx]
        display = self._base_display.copy()
        display[state == 1] = self._IDX_BURNING
        display[state == 2] = self._IDX_BURNED
        self._write_scalars(display.flatten(order="F"))
        if idx % 50 == 0:
            self._update_status(f"Playing  frame {idx}/{len(self.history)}")

    def _write_scalars(self, flat: np.ndarray) -> None:
        self._scalars.Reset()
        for v in flat:
            self._scalars.InsertNextValue(float(v))
        self._scalars.Modified()
        self._vtk_grid.GetPointData().SetScalars(self._scalars)
        self._vtk_grid.Modified()
        self._mapper.SetInputData(self._vtk_grid)
        self._mapper.Update()
        self.plotter.render()

    def _update_status(self, text: str) -> None:
        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=11,
            color="white",
            name="status",
        )
        print(text)
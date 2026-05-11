import pyvista as pv
import numpy as np
import os
from PIL import Image

class Visualizer3D:
    def __init__(self, model):
        self.model = model
        self.history = []
        self._frame_idx = 0
        self._paused = False
        
        rows, cols = self.model.landscape.shape
        cs = self.model.config.CELL_SIZE_METERS
        
        # Construct the Grid
        xx, yy = np.meshgrid(
            np.arange(cols, dtype=float) * cs,
            np.arange(rows, dtype=float) * cs,
        )
        
        Z_EXAGGERATION = 0.5 
        zz = self.model.landscape.elevation.astype(float) * Z_EXAGGERATION
        
        self.plotter = pv.Plotter(title="Figure 4: 3D Digital Twin (Ultra-Safe Hardware Mode)")
        self.plotter.set_background('white')

        self.mesh = pv.StructuredGrid(xx, yy, zz)
        texture_path = "texture.jpg"

        # ====================================================================
        # 1. BASE TOPOGRAPHY: RAW VERTEX COLORS (FIXES OPENGL CRASH)
        # ====================================================================
        # Αντί για vtkTexture, βάφουμε κάθε σημείο του βουνού με το χρώμα του δορυφόρου!
        if os.path.exists(texture_path):
            try:
                img = Image.open(texture_path).convert('RGB')
                # Κάνουμε resize την εικόνα ώστε να έχει ακριβώς 1 pixel για κάθε σημείο του βουνού
                img = img.resize((cols, rows)) 
                # Μετατρέπουμε την εικόνα σε μια τεράστια λίστα με [Red, Green, Blue]
                img_data = np.flipud(np.array(img)).reshape(-1, 3) 
                
                # Περνάμε τα χρώματα κατευθείαν στα σημεία! (Μηδενικό ζόρι για την κάρτα γραφικών)
                self.mesh.point_data['RGB'] = img_data
                self.plotter.add_mesh(self.mesh, scalars='RGB', rgb=True, lighting=True)
            except Exception as e:
                print(f"[Warning] Vertex mapping failed: {e}")
                self.plotter.add_mesh(self.mesh, cmap="terrain")
        else:
            self.plotter.add_mesh(self.mesh, cmap="terrain")

        # ====================================================================
        # 2. FIRE LAYER: RAW RGBA
        # ====================================================================
        self.fire_mesh = pv.StructuredGrid(xx, yy, zz + 1.0) 
        self.fire_colors = np.zeros((self.fire_mesh.n_points, 4), dtype=np.uint8)
        self.fire_mesh.point_data['fire_color'] = self.fire_colors
        
        self.plotter.add_mesh(
            self.fire_mesh, 
            scalars='fire_color', 
            rgba=True, 
            show_scalar_bar=False
        )

        # 3. WORLD FRAME QUIVER (Wind Vectors)
        wind_u = self.model.landscape.wind_u
        wind_v = self.model.landscape.wind_v
        
        step_r, step_c = max(1, rows // 15), max(1, cols // 15)
        arr_x = xx[::step_r, ::step_c].ravel()
        arr_y = yy[::step_r, ::step_c].ravel()
        arr_z = zz[::step_r, ::step_c].ravel() + 35 
        
        vec_pts = np.column_stack((arr_x, arr_y, arr_z))
        vec_dir = np.column_stack((
            np.full_like(arr_x, wind_u),
            np.full_like(arr_y, wind_v),
            np.zeros_like(arr_z)
        ))
        
        quiver_cloud = pv.PolyData(vec_pts)
        quiver_cloud['wind'] = vec_dir
        arrows = quiver_cloud.glyph(orient='wind', factor=15, scale=False)
        self.plotter.add_mesh(arrows, color="cyan", line_width=2)
        
        # UI Labels
        self.plotter.add_text("Cyan Arrows: Wind Vector (Quiver)", position="lower_right", font_size=10, color="black")
        self.plotter.add_key_event("p", self._toggle_pause)

    def _render_frame(self, idx):
        state = self.history[idx].ravel()
        
        colors = np.zeros((len(state), 4), dtype=np.uint8)
        colors[state == 1] = [255, 30, 0, 210]  # Φωτιά
        colors[state == 2] = [30, 30, 30, 180]  # Στάχτη
        
        self.fire_mesh.point_data['fire_color'] = colors
        self.plotter.add_text(f"Simulating: Frame {idx}/{len(self.history)} | P: Pause/Play", position="upper_left", font_size=12, color="black", name="status")

    def play_history(self, history, fps=30):
        self.history = history
        self.plotter.show(interactive_update=True)
        
        while self.plotter.iren.initialized:
            if not self._paused and self._frame_idx < len(self.history):
                self._render_frame(self._frame_idx)
                self._frame_idx += 1
            self.plotter.update()

    def _toggle_pause(self): 
        self._paused = not self._paused
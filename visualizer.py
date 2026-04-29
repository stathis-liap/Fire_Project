# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

class Visualizer:
    def __init__(self, model):
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Colors: Green (0), Red (1), Black (2)
        self.cmap = ListedColormap(['#2ca02c', '#d62728', '#1f77b4'])
        self.img = self.ax.imshow(self.model.state, cmap=self.cmap, vmin=0, vmax=2)
        plt.title("Wildfire Digital Twin Simulation")

    def update(self, frame):
        self.model.step()
        self.img.set_data(self.model.state)
        return [self.img]

    def render(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=200, interval=100, blit=True)
        plt.show()
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def visualize_terrain_technical(dem_file):
    """ Technical visualization with 3 subplots: 2D, Point Cloud, and Mesh in Meters. """
    print(f"\n[Visualizer-Matplotlib] Loading: {dem_file}")

    with rasterio.open(dem_file) as src:
        dem_data = src.read(1)
        dem_data = np.where(dem_data < 0, 0, dem_data) 

    # Flip the data to align North with the top of the Y-axis
    dem_data = np.flipud(dem_data)

    # Resolution of Copernicus DEM in meters
    resolution = 30 
    
    # Calculate total dimensions in meters for the 2D plot
    max_x_meters = dem_data.shape[1] * resolution
    max_y_meters = dem_data.shape[0] * resolution

    # Downsample for Matplotlib performance
    step = 1                                      # Adjust this step for larger/smaller datasets
    dem_small = dem_data[::step, ::step]

    # --- FIX: Convert indices to real-world METERS ---
    # We multiply by the resolution (30m) and the downsample step
    x = np.arange(dem_small.shape[1]) * (resolution * step)
    y = np.arange(dem_small.shape[0]) * (resolution * step)
    X, Y = np.meshgrid(x, y)
    
    points_x, points_y, points_z = X.flatten(), Y.flatten(), dem_small.flatten()

    # Perform Delaunay Triangulation
    triangulation = mtri.Triangulation(points_x, points_y)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(f"Technical Terrain Analysis ({max_x_meters}m x {max_y_meters}m)", fontsize=16)

    chosen_cmap = 'viridis' 

    # 1. 2D Map
    ax1 = fig.add_subplot(131)
    # The 'extent' explicitly maps the pixels to meters: [xmin, xmax, ymin, ymax]
    # 'origin=lower' ensures the flipped data matches standard cartesian coordinates
    im = ax1.imshow(dem_data, cmap=chosen_cmap, extent=[0, max_x_meters, 0, max_y_meters], origin='lower')
    ax1.set_title("1. 2D Elevation Map")
    ax1.set_xlabel("Distance X (meters)")
    ax1.set_ylabel("Distance Y (meters)")
    fig.colorbar(im, ax=ax1, label='Altitude (meters)', shrink=0.5)

    # 2. 3D Point Cloud
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points_x, points_y, points_z, c=points_z, cmap=chosen_cmap, s=5, marker='.')
    ax2.set_title("2. 3D Point Cloud")
    ax2.set_xlabel("Distance X (m)")
    ax2.set_ylabel("Distance Y (m)")
    ax2.set_zlabel("Altitude (m)")
    ax2.view_init(elev=45, azim=-45)

    # 3. 3D Triangulated Mesh
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_trisurf(points_x, points_y, points_z, 
                            triangles=triangulation.triangles, 
                            cmap=chosen_cmap, edgecolor='black', linewidth=0.1)
    ax3.set_title("3. Delaunay Mesh")
    ax3.set_xlabel("Distance X (m)")
    ax3.set_ylabel("Distance Y (m)")
    ax3.set_zlabel("Altitude (m)")
    ax3.view_init(elev=45, azim=-45)

    plt.tight_layout()
    print("[Visualizer-Matplotlib] Displaying technical plots with metric axes...")
    plt.show()

if __name__ == "__main__":
    visualize_terrain_technical("auto_downloaded_terrain.tif")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from PIL import Image

CORINE_TO_FUEL = {
    (0, 166, 0): "Aleppo_Pine", (77, 255, 0): "Aleppo_Pine", (128, 255, 0): "Oak_Forest",
    (166, 230, 77): "Maquis_Dense_Shrub", (166, 242, 0): "Phrygana_Low_Scrub",
    (230, 166, 0): "Olive_Grove", (230, 230, 77): "Olive_Grove", (255, 230, 166): "Olive_Grove", (242, 205, 166): "Olive_Grove",
    (204, 242, 77): "Dry_Grass", (230, 230, 0): "Dry_Grass", (255, 255, 168): "Dry_Grass", (255, 255, 0): "Dry_Grass",
    (230, 0, 77): "Non_Combustible", (255, 0, 0): "Non_Combustible", (204, 77, 242): "Non_Combustible", 
    (166, 166, 166): "Non_Combustible", (204, 204, 204): "Non_Combustible", (0, 204, 242): "Non_Combustible", 
    (0, 0, 230): "Non_Combustible", (255, 255, 255): "Non_Combustible"
}

def visualize_corine_vegetation(corine_file="corine_cover.png", texture_file="texture.jpg"):
    try:
        img = Image.open(corine_file).convert('RGB')
    except Exception as e:
        return

    corine_data = np.flipud(np.array(img))
    rows, cols, _ = corine_data.shape
    
    try:
        # Load high-res texture but DO NOT resize it!
        tex_data = np.flipud(np.array(Image.open(texture_file).convert('RGB')))
    except Exception as e:
        tex_data = np.ones((rows, cols, 3), dtype=np.uint8) * 255

    profiles = {
        0: [(0, 166, 0), (77, 255, 0)],
        1: [(128, 255, 0)],
        2: [(166, 230, 77), (166, 242, 0)],
        3: [(230, 166, 0), (230, 230, 77), (255, 230, 166), (242, 205, 166)],
        4: [(204, 242, 77), (230, 230, 0), (255, 255, 168), (255, 255, 0)],
        5: [(230, 0, 77), (255, 0, 0), (166, 166, 166), (0, 204, 242), (0, 0, 230), (255, 255, 255)]
    }
    
    pixels = corine_data.reshape(-1, 3).astype(int)
    min_dists = np.full(pixels.shape[0], np.inf)
    best_cats = np.full(pixels.shape[0], 5)
    
    for cat_idx, colors in profiles.items():
        for pr, pg, pb in colors:
            dist = (pixels[:, 0] - pr)**2 + (pixels[:, 1] - pg)**2 + (pixels[:, 2] - pb)**2
            mask = dist < min_dists
            min_dists[mask] = dist[mask]
            best_cats[mask] = cat_idx
            
    category_map = best_cats.reshape(rows, cols)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title('Figure 2: Vegetation Mapping')
    cmap = ListedColormap(['#006400', '#90EE90', '#6B8E23', '#DAA520', '#F0E68C', '#808080'])
    
    # Use extent=[0, cols, 0, rows] to perfectly align different resolution images
    map_extent = [0, cols, 0, rows]

    axes[0].imshow(tex_data, extent=map_extent, origin='lower')
    axes[0].set_title("i. Original Satellite Texture (High Res)", fontsize=13, weight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(category_map, cmap=cmap, vmin=0, vmax=5, extent=map_extent, origin='lower')
    axes[1].set_title("ii. CORINE Vegetation Classes", fontsize=13, weight='bold')
    axes[1].axis('off')

    axes[2].imshow(tex_data, extent=map_extent, origin='lower')
    axes[2].imshow(category_map, cmap=cmap, vmin=0, vmax=5, extent=map_extent, origin='lower', alpha=0.45)
    axes[2].set_title("iii. Spatial Overlay (Perfect Alignment)", fontsize=13, weight='bold')
    axes[2].axis('off')
    
    labels = ['Aleppo Pine', 'Oak Forest', 'Maquis/Shrub', 'Olive/Agri', 'Dry Grass', 'Non-Combustible']
    colors = ['#006400', '#90EE90', '#6B8E23', '#DAA520', '#F0E68C', '#808080']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    
    fig.legend(handles=patches, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
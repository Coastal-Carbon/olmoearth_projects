import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
import random

# Base directory containing chip folders
BASE_DIR = Path("dataset_pastis/windows/predict")

# Find all chip directories (e.g., pastis_10103)
chip_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]

# Randomly sample 10 chips
selected_chips = random.sample(chip_dirs, 10)
print("Selected chips:")
for c in selected_chips:
    print(" -", c.name)

def visualize_chip(chip_path):
    chip_name = chip_path.name
    base = chip_path / "layers"

    prediction_path = base / "output/output/geotiff.tif"
    label_path = base / "label/B1/geotiff.tif"

    # Load label and prediction
    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)

    # Calculate accuracy
    valid_mask = (label >= 0) & (label < 20) & (prediction >= 0) & (prediction < 20)
    accuracy = (prediction[valid_mask] == label[valid_mask]).sum() / valid_mask.sum() * 100

    # Sentinel-1 and Sentinel-2 layer names
    s1_layer_names = ["sentinel1"] + [f"sentinel1.{i}" for i in range(1, 12)]
    s2_layer_names = ["sentinel2_l2a"] + [f"sentinel2_l2a.{i}" for i in range(1, 12)]

    s1_data = []
    for layer_name in s1_layer_names:
        geotiff_path = base / layer_name / "vv_vh" / "geotiff.tif"
        if geotiff_path.exists():
            with rasterio.open(geotiff_path) as src:
                s1_data.append(src.read(1))

    s2_data = []
    for layer_name in s2_layer_names:
        layer_dir = base / layer_name
        geotiff_files = list(layer_dir.glob("*/geotiff.tif"))
        if geotiff_files:
            geotiff_path = geotiff_files[0]
            with rasterio.open(geotiff_path) as src:
                rgb = np.dstack([
                    np.clip((src.read(i) - np.percentile(src.read(i), 2)) /
                            (np.percentile(src.read(i), 98) - np.percentile(src.read(i), 2)), 0, 1)
                    for i in [3, 2, 1]
                ])
                s2_data.append(rgb)

    # Colormap
    mask_cmap = plt.cm.get_cmap('tab20', 20)

    # Create figure
    n_rows = max(len(s1_data), len(s2_data)) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))

    # S1 time series
    for i, vv in enumerate(s1_data):
        axes[i, 0].imshow(vv, cmap='gray',
                          vmin=np.percentile(vv, 2),
                          vmax=np.percentile(vv, 98))
        axes[i, 0].set_title(f"S1 VV - T{i+1}")
        axes[i, 0].axis('off')

    # S2 time series
    for i, rgb in enumerate(s2_data):
        axes[i, 1].imshow(rgb)
        axes[i, 1].set_title(f"S2 RGB - T{i+1}")
        axes[i, 1].axis('off')

    # Ground truth
    last = n_rows - 1
    axes[last, 0].imshow(label, cmap=mask_cmap, vmin=0, vmax=19)
    axes[last, 0].set_title("Ground Truth")
    axes[last, 0].axis('off')

    # Prediction
    axes[last, 1].imshow(prediction, cmap=mask_cmap, vmin=0, vmax=19)
    axes[last, 1].set_title(f"Prediction\nAccuracy: {accuracy:.2f}%")
    axes[last, 1].axis('off')

    # Legend
    legend_elems = [Patch(facecolor=mask_cmap(i), label=str(i)) for i in range(20)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=10)

    plt.suptitle(chip_name, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    out_path = f"{chip_name}_temporal_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path} (Accuracy: {accuracy:.2f}%)")

# Run visualization for all selected chips
print("\nGenerating plots...")
for chip in selected_chips:
    visualize_chip(chip)

"""
Visualize regression predictions vs ground truth for soiling index.
Shows Sentinel-2 RGB, label, prediction, and error for sampled chips.
Computes per-chip and aggregate regression metrics.
"""

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

# === CONFIG ===
BASE_DIR = Path("dataset_soiling/windows/predict")
N_SAMPLES = 10
SEED = 42

random.seed(SEED)

# Find all chip directories
chip_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]

# Sample chips
n_samples = min(N_SAMPLES, len(chip_dirs))
selected_chips = random.sample(chip_dirs, n_samples)

print(f"Selected {n_samples} chips:")
for c in selected_chips:
    print(f"  - {c.name}")


def load_chip_data(chip_path):
    """Load label, prediction, and S2 RGB for a chip."""
    base = chip_path / "layers"
    
    label_path = base / "label/B1/geotiff.tif"
    prediction_path = base / "output/output/geotiff.tif"
    
    # Load label and prediction
    with rasterio.open(label_path) as src:
        label = src.read(1)
    
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    
    # Load S2 RGB from single layer
    s2_rgb = None
    s2_path = base / "sentinel2_l2a/B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif"
    
    if s2_path.exists():
        with rasterio.open(s2_path) as src:
            # Bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
            # RGB = B04 (band 4), B03 (band 3), B02 (band 2)
            bands = []
            for i in [4, 3, 2]:  # B04, B03, B02 for RGB
                band = src.read(i).astype(float)
                p2, p98 = np.percentile(band, [2, 98])
                band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
                bands.append(band)
            s2_rgb = np.dstack(bands)
    
    return label, prediction, s2_rgb


def compute_metrics(labels, predictions):
    """Compute regression metrics."""
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    errors = predictions - labels
    
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    
    # R² calculation
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def visualize_chip(chip_path, label, prediction, s2_rgb, idx):
    """Create visualization for a single chip."""
    chip_name = chip_path.name
    
    # Get mean values (labels are uniform, predictions may vary slightly)
    label_val = np.mean(label[label >= 0])
    pred_val = np.mean(prediction[prediction >= 0])
    error = pred_val - label_val
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # S2 RGB
    if s2_rgb is not None:
        axes[0].imshow(s2_rgb)
        axes[0].set_title("Sentinel-2 RGB")
    else:
        axes[0].text(0.5, 0.5, "No S2 data", ha="center", va="center")
        axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis("off")
    
    # Label
    im1 = axes[1].imshow(label, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title(f"Label\n(mean: {label_val:.4f})")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Prediction
    im2 = axes[2].imshow(prediction, cmap="RdYlGn", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction\n(mean: {pred_val:.4f})")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Error map
    error_map = prediction - label
    im3 = axes[3].imshow(error_map, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[3].set_title(f"Error (Pred - Label)\n(mean: {error:+.4f})")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{chip_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    out_path = f"soiling_prediction_{idx:02d}_{chip_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return label_val, pred_val


def plot_scatter(labels, predictions, metrics):
    """Create scatter plot of predictions vs labels."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(labels, predictions, alpha=0.7, s=100, edgecolors="black")
    
    # Perfect prediction line
    min_val = min(min(labels), min(predictions))
    max_val = max(max(labels), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
    
    ax.set_xlabel("Actual (Label)", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Soiling Index: Predicted vs Actual", fontsize=14, fontweight="bold")
    
    # Add metrics text
    text = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}\nMAE = {metrics['mae']:.4f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("soiling_prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Saved: soiling_prediction_scatter.png")


def main():
    all_labels = []
    all_predictions = []
    
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    for idx, chip_path in enumerate(selected_chips, 1):
        try:
            label, prediction, s2_rgb = load_chip_data(chip_path)
            label_val, pred_val = visualize_chip(chip_path, label, prediction, s2_rgb, idx)
            
            all_labels.append(label_val)
            all_predictions.append(pred_val)
            
            error = pred_val - label_val
            print(f"[{idx:02d}] {chip_path.name}")
            print(f"     Label: {label_val:.4f} | Pred: {pred_val:.4f} | Error: {error:+.4f}")
            
        except Exception as e:
            print(f"[{idx:02d}] {chip_path.name} - FAILED: {e}")
    
    # Compute aggregate metrics
    if len(all_labels) > 1:
        metrics = compute_metrics(all_labels, all_predictions)
        
        print("\n" + "=" * 60)
        print("AGGREGATE METRICS")
        print("=" * 60)
        print(f"  Samples:  {len(all_labels)}")
        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  R²:       {metrics['r2']:.6f}")
        print("=" * 60)
        
        # Create scatter plot
        plot_scatter(all_labels, all_predictions, metrics)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
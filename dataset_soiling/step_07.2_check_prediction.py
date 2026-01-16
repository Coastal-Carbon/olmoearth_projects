"""
Evaluate regression predictions on all chips in predict set.
Computes aggregate metrics and generates scatter plot.
"""

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path("dataset_soiling/windows/predict")

# Find all chip directories
chip_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
print(f"Total chips: {len(chip_dirs)}")


def load_chip_values(chip_path):
    """Load label and prediction mean values for a chip."""
    base = chip_path / "layers"
    
    label_path = base / "label/B1/geotiff.tif"
    prediction_path = base / "output/output/geotiff.tif"
    
    with rasterio.open(label_path) as src:
        label = src.read(1)
    
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    
    # Handle nodata
    valid_label = label[label >= 0]
    valid_pred = prediction[prediction >= 0]
    
    label_val = np.mean(valid_label) if len(valid_label) > 0 else np.nan
    pred_val = np.mean(valid_pred) if len(valid_pred) > 0 else np.nan
    
    return label_val, pred_val


def compute_metrics(labels, predictions):
    """Compute regression metrics."""
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    errors = predictions - labels
    
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def plot_scatter(labels, predictions, metrics):
    """Create scatter plot of predictions vs labels."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(labels, predictions, alpha=0.5, s=20, edgecolors="none")
    
    # Perfect prediction line
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect prediction")
    
    ax.set_xlabel("Actual (Label)", fontsize=14)
    ax.set_ylabel("Predicted", fontsize=14)
    ax.set_title(f"Soiling Index: Predicted vs Actual (n={len(labels)})", fontsize=16, fontweight="bold")
    
    # Add metrics text
    text = (
        f"R² = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"n = {len(labels)}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("soiling_prediction_scatter_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Saved: soiling_prediction_scatter_all.png")


def plot_error_distribution(labels, predictions, metrics):
    """Create error distribution histogram."""
    errors = np.array(predictions) - np.array(labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero error")
    ax.axvline(x=np.mean(errors), color="g", linestyle="-", linewidth=2, label=f"Mean: {np.mean(errors):.4f}")
    
    ax.set_xlabel("Prediction Error (Pred - Label)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"Error Distribution (n={len(errors)})", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("soiling_prediction_error_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Saved: soiling_prediction_error_dist.png")


def main():
    all_labels = []
    all_predictions = []
    failed = []
    
    print("\nLoading predictions...")
    
    for chip_path in tqdm(chip_dirs, desc="Processing chips", unit="chip"):
        try:
            label_val, pred_val = load_chip_values(chip_path)
            all_labels.append(label_val)
            all_predictions.append(pred_val)
        except Exception as e:
            failed.append((chip_path.name, str(e)))
    
    print(f"\nProcessed: {len(all_labels)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed chips:")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Filter out nan values
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    valid_mask = ~(np.isnan(all_labels) | np.isnan(all_predictions))
    
    nan_count = (~valid_mask).sum()
    all_labels = all_labels[valid_mask].tolist()
    all_predictions = all_predictions[valid_mask].tolist()
    
    print(f"\nValid samples: {len(all_labels)} (excluded {nan_count} with nan/nodata)")
    
    if len(all_labels) == 0:
        print("ERROR: No valid samples to evaluate!")
        return
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions)
    
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS (ALL CHIPS)")
    print("=" * 60)
    print(f"  Samples:  {len(all_labels)}")
    print(f"  MSE:      {metrics['mse']:.6f}")
    print(f"  RMSE:     {metrics['rmse']:.6f}")
    print(f"  MAE:      {metrics['mae']:.6f}")
    print(f"  R²:       {metrics['r2']:.6f}")
    print("=" * 60)
    
    # Plots
    plot_scatter(all_labels, all_predictions, metrics)
    plot_error_distribution(all_labels, all_predictions, metrics)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
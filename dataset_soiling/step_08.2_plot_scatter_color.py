"""
Scatter plot of predictions vs labels, color-coded by device.
Shows which devices have systematic prediction errors.
Creates two plots: all devices and sampled 10 devices.
"""

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
import random

# === CONFIG ===
BASE_DIR = Path("dataset_soiling/windows/predict")
N_SAMPLE_DEVICES = 10
SEED = 42

random.seed(SEED)

# Find all chip directories
chip_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
print(f"Total chips: {len(chip_dirs)}")


def extract_device_id(chip_name: str) -> str:
    """Extract device ID from chip name (format: PLANT_DEVICE_DATE)."""
    parts = chip_name.split("_")
    # Find last date-like part (YYYY-MM-DD format)
    for i in range(len(parts) - 1, -1, -1):
        if len(parts[i]) == 10 and parts[i].count("-") == 2:
            device_id = "_".join(parts[:i])
            return device_id
    return chip_name  # fallback


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
    
    device_id = extract_device_id(chip_path.name)
    
    return label_val, pred_val, device_id


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


def plot_scatter_by_device(labels, predictions, devices, metrics, output_file, title_suffix=""):
    """Create scatter plot color-coded by device."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique devices and assign colors
    unique_devices = sorted(set(devices))
    n_devices = len(unique_devices)
    colors = cm.tab20(np.linspace(0, 1, min(n_devices, 20)))
    
    if n_devices > 20:
        # Use tab20b and tab20c for more devices
        colors = np.vstack([
            cm.tab20(np.linspace(0, 1, 20)),
            cm.tab20b(np.linspace(0, 1, 20)),
            cm.tab20c(np.linspace(0, 1, 20))
        ])
    
    device_to_color = {dev: colors[i % len(colors)] for i, dev in enumerate(unique_devices)}
    
    # Plot each device
    for device in unique_devices:
        mask = np.array([d == device for d in devices])
        device_labels = np.array(labels)[mask]
        device_preds = np.array(predictions)[mask]
        
        ax.scatter(
            device_labels, 
            device_preds, 
            alpha=0.6, 
            s=50, 
            color=device_to_color[device],
            label=device if n_devices <= 10 else None,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Perfect prediction line
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect prediction")
    
    ax.set_xlabel("Actual (Label)", fontsize=14)
    ax.set_ylabel("Predicted", fontsize=14)
    ax.set_title(f"Soiling Index: Predicted vs Actual{title_suffix}\n(n={len(labels)} chips, {n_devices} devices)", 
                 fontsize=16, fontweight="bold")
    
    # Add metrics text
    text = (
        f"R² = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"n = {len(labels)}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Legend
    if n_devices <= 10:
        ax.legend(loc="lower right", fontsize=9, ncol=1)
    else:
        # Just show the perfect prediction line
        handles, labels_legend = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels_legend[-1]], loc="lower right", fontsize=12)
        
        # Add note about number of devices
        ax.text(0.95, 0.05, f"{n_devices} devices\n(legend omitted)", 
                transform=ax.transAxes, fontsize=10,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_file}")


def main():
    all_labels = []
    all_predictions = []
    all_devices = []
    failed = []
    
    print("\nLoading predictions...")
    
    for chip_path in chip_dirs:
        try:
            label_val, pred_val, device_id = load_chip_values(chip_path)
            all_labels.append(label_val)
            all_predictions.append(pred_val)
            all_devices.append(device_id)
        except Exception as e:
            failed.append((chip_path.name, str(e)))
    
    print(f"Processed: {len(all_labels)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed chips (first 10):")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")
    
    # Filter out nan values
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_devices = np.array(all_devices)
    
    valid_mask = ~(np.isnan(all_labels) | np.isnan(all_predictions))
    
    nan_count = (~valid_mask).sum()
    all_labels = all_labels[valid_mask].tolist()
    all_predictions = all_predictions[valid_mask].tolist()
    all_devices = all_devices[valid_mask].tolist()
    
    print(f"\nValid samples: {len(all_labels)} (excluded {nan_count} with nan/nodata)")
    print(f"Unique devices: {len(set(all_devices))}")
    
    if len(all_labels) == 0:
        print("ERROR: No valid samples to evaluate!")
        return
    
    # Compute metrics for all data
    metrics_all = compute_metrics(all_labels, all_predictions)
    
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS (ALL CHIPS)")
    print("=" * 60)
    print(f"  Samples:  {len(all_labels)}")
    print(f"  MSE:      {metrics_all['mse']:.6f}")
    print(f"  RMSE:     {metrics_all['rmse']:.6f}")
    print(f"  MAE:      {metrics_all['mae']:.6f}")
    print(f"  R²:       {metrics_all['r2']:.6f}")
    print("=" * 60)
    
    # Create scatter plot for ALL devices
    plot_scatter_by_device(
        all_labels, 
        all_predictions, 
        all_devices, 
        metrics_all,
        "soiling_prediction_scatter_all_devices.png",
        ""
    )
    
    # Sample 10 devices
    unique_devices = list(set(all_devices))
    if len(unique_devices) > N_SAMPLE_DEVICES:
        sampled_devices = random.sample(unique_devices, N_SAMPLE_DEVICES)
    else:
        sampled_devices = unique_devices
    
    print(f"\nSampled {len(sampled_devices)} devices:")
    for dev in sorted(sampled_devices):
        count = all_devices.count(dev)
        print(f"  - {dev}: {count} chips")
    
    # Filter data to sampled devices
    sampled_mask = [d in sampled_devices for d in all_devices]
    sampled_labels = [l for l, m in zip(all_labels, sampled_mask) if m]
    sampled_predictions = [p for p, m in zip(all_predictions, sampled_mask) if m]
    sampled_devices_list = [d for d, m in zip(all_devices, sampled_mask) if m]
    
    # Compute metrics for sampled data
    metrics_sampled = compute_metrics(sampled_labels, sampled_predictions)
    
    print("\n" + "=" * 60)
    print(f"AGGREGATE METRICS (SAMPLED {len(sampled_devices)} DEVICES)")
    print("=" * 60)
    print(f"  Samples:  {len(sampled_labels)}")
    print(f"  MSE:      {metrics_sampled['mse']:.6f}")
    print(f"  RMSE:     {metrics_sampled['rmse']:.6f}")
    print(f"  MAE:      {metrics_sampled['mae']:.6f}")
    print(f"  R²:       {metrics_sampled['r2']:.6f}")
    print("=" * 60)
    
    # Create scatter plot for SAMPLED devices
    plot_scatter_by_device(
        sampled_labels,
        sampled_predictions,
        sampled_devices_list,
        metrics_sampled,
        "soiling_prediction_scatter_10_devices.png",
        f" (Sampled {len(sampled_devices)} Devices)"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
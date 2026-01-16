"""
Visualize Temporal Progression: Same Device Over Time (GeoJSON version)

For N random devices, plots time series showing:
- Sentinel-2 RGB composite
- Label value (from GeoJSON)
- Prediction value (from GeoJSON)
- Error over time

Shows how predictions change over time for the same location.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# === CONFIG ===
WINDOWS_DIR = Path("dataset_soiling_reg/windows/predict")
N_DEVICES = 10
SEED = 42
OUTPUT_DIR = Path("temporal_visualizations_geojson")

random.seed(SEED)


def load_geojson_value(filepath: Path, key="soiling_normalized") -> float:
    """Load scalar value from GeoJSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    features = data.get("features", [])
    if not features:
        return np.nan
    
    return features[0]["properties"].get(key, np.nan)


def load_s2_rgb(chip_dir: Path) -> np.ndarray | None:
    """Load Sentinel-2 RGB composite."""
    s2_path = chip_dir / "layers/sentinel2_l2a/B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif"
    
    if not s2_path.exists():
        return None
    
    with rasterio.open(s2_path) as src:
        bands = []
        for i in [4, 3, 2]:  # B04, B03, B02 for RGB
            band = src.read(i).astype(float)
            p2, p98 = np.percentile(band, [2, 98])
            band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
            bands.append(band)
        return np.dstack(bands)


def group_chips_by_device(windows_dir: Path) -> dict:
    """
    Group chips by device ID.
    Returns dict: {device_id: [list of chip info dicts]}
    """
    device_chips = defaultdict(list)
    
    for chip_dir in windows_dir.iterdir():
        if not chip_dir.is_dir():
            continue
        
        chip_name = chip_dir.name
        
        # Extract device ID and date from chip name
        # Format: PLANT_DEVICE_DATE or similar
        parts = chip_name.split("_")
        
        # Find last date-like part (YYYY-MM-DD format)
        date = None
        for i in range(len(parts) - 1, -1, -1):
            if len(parts[i]) == 10 and parts[i].count("-") == 2:
                device_id = "_".join(parts[:i])
                date = parts[i]
                break
        
        if date is None:
            continue
        
        # Check if label and prediction exist
        label_path = chip_dir / "layers/label/data.geojson"
        pred_path = chip_dir / "layers/output/data.geojson"
        
        if label_path.exists() and pred_path.exists():
            device_chips[device_id].append({
                "chip_dir": chip_dir,
                "chip_name": chip_name,
                "date": date,
                "label_path": label_path,
                "pred_path": pred_path
            })
    
    return device_chips


def visualize_device_timeline(device_id: str, chips: list, output_dir: Path, device_idx: int):
    """Create timeline visualization for a single device."""
    # Sort by date
    chips_sorted = sorted(chips, key=lambda x: x["date"])
    n_dates = len(chips_sorted)
    
    # Load all values
    dates = []
    labels = []
    predictions = []
    s2_rgbs = []
    
    for chip_info in chips_sorted:
        dates.append(chip_info["date"])
        labels.append(load_geojson_value(chip_info["label_path"]))
        predictions.append(load_geojson_value(chip_info["pred_path"]))
        s2_rgbs.append(load_s2_rgb(chip_info["chip_dir"]))
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    errors = predictions - labels
    
    # Create figure with two rows:
    # Top row: S2 RGB images for each date
    # Bottom row: Time series plot
    fig = plt.figure(figsize=(max(16, n_dates * 2.5), 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Top: RGB images
    gs_top = gs[0].subgridspec(1, n_dates, wspace=0.05)
    for i, (date, s2_rgb) in enumerate(zip(dates, s2_rgbs)):
        ax = fig.add_subplot(gs_top[i])
        if s2_rgb is not None:
            ax.imshow(s2_rgb)
        else:
            ax.text(0.5, 0.5, "No S2", ha="center", va="center", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(date, fontsize=9)
        ax.axis("off")
    
    # Bottom: Time series
    ax_ts = fig.add_subplot(gs[1])
    
    x = np.arange(n_dates)
    
    # Plot lines
    ax_ts.plot(x, labels, 'o-', label='Label', linewidth=2, markersize=8, color='blue')
    ax_ts.plot(x, predictions, 's-', label='Prediction', linewidth=2, markersize=8, color='red')
    ax_ts.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    # Add error bars as vertical lines
    for i, (l, p) in enumerate(zip(labels, predictions)):
        ax_ts.plot([i, i], [l, p], 'k-', alpha=0.3, linewidth=1)
    
    ax_ts.set_xlabel('Date', fontsize=11)
    ax_ts.set_ylabel('Soiling (normalized)', fontsize=11)
    ax_ts.set_title(f'Device: {device_id}', fontsize=12, fontweight='bold')
    ax_ts.set_xticks(x)
    ax_ts.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax_ts.set_ylim(-0.05, 1.05)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(fontsize=10)
    
    # Add metrics text
    mae = np.nanmean(np.abs(errors))
    rmse = np.sqrt(np.nanmean(errors**2))
    metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nn={n_dates}'
    ax_ts.text(0.02, 0.98, metrics_text, transform=ax_ts.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    out_path = output_dir / f"device_{device_idx:02d}_{device_id.replace('/', '-')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  [{device_idx:02d}] Saved: {out_path.name}")
    
    return {
        "device_id": device_id,
        "n_dates": n_dates,
        "dates": dates,
        "mae": mae,
        "rmse": rmse
    }


def main():
    print("Temporal Device Visualization (GeoJSON)")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Group chips by device
    print(f"\nScanning: {WINDOWS_DIR}")
    device_chips = group_chips_by_device(WINDOWS_DIR)
    print(f"Found {len(device_chips)} unique devices")
    
    # Filter devices with multiple dates
    multi_date_devices = {k: v for k, v in device_chips.items() if len(v) > 1}
    print(f"Devices with multiple dates: {len(multi_date_devices)}")
    
    if not multi_date_devices:
        print("No devices with multiple dates found!")
        return
    
    # Sample N devices
    n_to_sample = min(N_DEVICES, len(multi_date_devices))
    selected_device_ids = random.sample(list(multi_date_devices.keys()), n_to_sample)
    selected_devices = [(dev_id, multi_date_devices[dev_id]) for dev_id in selected_device_ids]
    
    print(f"\nVisualizing {len(selected_devices)} devices...")
    print("-" * 70)
    
    # Visualize each device
    results = []
    for idx, (device_id, chips) in enumerate(selected_devices, 1):
        result = visualize_device_timeline(device_id, chips, OUTPUT_DIR, idx)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Devices visualized: {len(results)}")
    print(f"  Total dates: {sum(r['n_dates'] for r in results)}")
    print()
    print("  Per-device metrics:")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['device_id']:<35} {r['n_dates']:>2} dates | MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f}")
    print("=" * 70)
    print(f"\nSaved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
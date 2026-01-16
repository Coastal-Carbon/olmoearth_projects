"""
Visualize Temporal Progression: Same Device Over Time

For N random devices, plots all available dates showing:
S2 RGB | Label | Prediction for each date

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
RAW_LABELS_DIR = Path("dataset_soiling/data/labels")
WINDOWS_DIR = Path("dataset_soiling/windows/predict")
METADATA_FILE = Path("dataset_soiling/data/metadata/train_metadata.json")

N_DEVICES = 10
SEED = 42
OUTPUT_DIR = Path("temporal_visualizations")

random.seed(SEED)


def load_metadata(filepath: Path) -> dict:
    """Load train metadata JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def group_chips_by_device(windows_dir: Path) -> dict:
    """
    Group chips by device ID.
    Returns dict: {device_id: [list of chip_dirs]}
    """
    device_chips = defaultdict(list)
    
    for chip_dir in windows_dir.iterdir():
        if not chip_dir.is_dir():
            continue
        
        chip_name = chip_dir.name
        
        # Extract device ID from chip name (format: PLANT_DEVICE_DATE)
        # Device ID = everything except the last date part
        parts = chip_name.split("_")
        # Find last date-like part (YYYY-MM-DD format)
        for i in range(len(parts) - 1, -1, -1):
            if len(parts[i]) == 10 and parts[i].count("-") == 2:
                device_id = "_".join(parts[:i])
                date = parts[i]
                break
        else:
            continue
        
        # Check if prediction exists
        prediction_path = chip_dir / "layers/output/output/geotiff.tif"
        if prediction_path.exists():
            device_chips[device_id].append({
                "chip_dir": chip_dir,
                "chip_name": chip_name,
                "date": date
            })
    
    return device_chips


def load_tif(filepath: Path) -> np.ndarray:
    """Load a single-band TIF."""
    with rasterio.open(filepath) as src:
        return src.read(1)


def load_s2_rgb(chip_dir: Path) -> np.ndarray | None:
    """Load Sentinel-2 RGB from chip."""
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


def get_mean_value(arr: np.ndarray, nodata: float = -1) -> float:
    """Get mean value excluding nodata."""
    valid = arr[arr != nodata]
    return np.mean(valid) if len(valid) > 0 else np.nan


def visualize_device_timeline(device_id: str, chips: list, output_dir: Path, device_idx: int):
    """Create timeline visualization for a single device."""
    # Sort by date
    chips_sorted = sorted(chips, key=lambda x: x["date"])
    n_dates = len(chips_sorted)
    
    # Create figure: n_dates rows × 3 columns (RGB, Label, Prediction)
    fig, axes = plt.subplots(n_dates, 3, figsize=(12, 4 * n_dates))
    fig.suptitle(f"Device: {device_id} (n={n_dates} dates)", fontsize=14, fontweight="bold")
    
    # Handle single date case
    if n_dates == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, chip_info in enumerate(chips_sorted):
        chip_dir = chip_info["chip_dir"]
        date = chip_info["date"]
        
        # Load data
        s2_rgb = load_s2_rgb(chip_dir)
        label = load_tif(chip_dir / "layers/label/B1/geotiff.tif")
        prediction = load_tif(chip_dir / "layers/output/output/geotiff.tif")
        
        # Compute mean values
        label_val = get_mean_value(label)
        pred_val = get_mean_value(prediction)
        error = pred_val - label_val
        
        # Plot S2 RGB
        if s2_rgb is not None:
            axes[row_idx, 0].imshow(s2_rgb)
        else:
            axes[row_idx, 0].text(0.5, 0.5, "No S2 RGB", ha="center", va="center")
        axes[row_idx, 0].set_title(f"S2 RGB\n{date}", fontsize=10)
        axes[row_idx, 0].axis("off")
        
        # Plot Label
        im1 = axes[row_idx, 1].imshow(label, cmap="RdYlGn", vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"Label\nmean: {label_val:.4f}", fontsize=10)
        axes[row_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        
        # Plot Prediction
        im2 = axes[row_idx, 2].imshow(prediction, cmap="RdYlGn", vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f"Prediction\nmean: {pred_val:.4f}\nerror: {error:+.4f}", fontsize=10)
        axes[row_idx, 2].axis("off")
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save
    out_path = output_dir / f"device_{device_idx:02d}_{device_id.replace('/', '-')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {out_path.name}")
    
    return {
        "device_id": device_id,
        "n_dates": n_dates,
        "dates": [c["date"] for c in chips_sorted]
    }


def main():
    print("Temporal Device Visualization")
    print("=" * 70)
    
    # Create output dir
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Group chips by device
    print(f"\nScanning windows directory: {WINDOWS_DIR}")
    device_chips = group_chips_by_device(WINDOWS_DIR)
    print(f"Found {len(device_chips)} unique devices")
    
    # Filter devices with multiple dates
    multi_date_devices = {k: v for k, v in device_chips.items() if len(v) > 1}
    print(f"Devices with multiple dates: {len(multi_date_devices)}")
    
    # Sample N devices
    if len(multi_date_devices) < N_DEVICES:
        print(f"Warning: Only {len(multi_date_devices)} devices available, using all")
        selected_devices = list(multi_date_devices.items())
    else:
        selected_device_ids = random.sample(list(multi_date_devices.keys()), N_DEVICES)
        selected_devices = [(dev_id, multi_date_devices[dev_id]) for dev_id in selected_device_ids]
    
    print(f"\nSelected {len(selected_devices)} devices for visualization")
    print("-" * 70)
    
    # Visualize each device
    results = []
    for idx, (device_id, chips) in enumerate(selected_devices, 1):
        print(f"[{idx:02d}] Processing {device_id} ({len(chips)} dates)...")
        result = visualize_device_timeline(device_id, chips, OUTPUT_DIR, idx)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Devices visualized:              {len(results)}")
    print(f"  Total dates across devices:      {sum(r['n_dates'] for r in results)}")
    print()
    print("  PER-DEVICE BREAKDOWN:")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['device_id']:<40} {r['n_dates']:>3} dates")
    print("=" * 70)
    
    print(f"\nVisualizations saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
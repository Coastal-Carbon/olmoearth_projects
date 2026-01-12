"""
Box-whisker plots comparing actual vs predicted soiling per device (GeoTIFF version).
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

# === CONFIG ===
BASE_DIR = Path("dataset_soiling/windows/predict")
N_SAMPLE_DEVICES = 10
SEED = 42

random.seed(SEED)


def extract_device_id(chip_name: str) -> str:
    parts = chip_name.split("_")
    for i in range(len(parts) - 1, -1, -1):
        if len(parts[i]) == 10 and parts[i].count("-") == 2:
            return "_".join(parts[:i])
    return chip_name


def load_chip_values(chip_path):
    base = chip_path / "layers"
    label_path = base / "label/B1/geotiff.tif"
    pred_path = base / "output/output/geotiff.tif"

    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(pred_path) as src:
        prediction = src.read(1)

    valid_label = label[label >= 0]
    valid_pred = prediction[prediction >= 0]

    label_val = np.mean(valid_label) if len(valid_label) > 0 else np.nan
    pred_val = np.mean(valid_pred) if len(valid_pred) > 0 else np.nan
    device_id = extract_device_id(chip_path.name)

    return label_val, pred_val, device_id


def plot_boxplot_by_device(labels, predictions, devices, output_file):
    """Side-by-side box plots for actual vs predicted per device."""
    unique_devices = sorted(set(devices))
    n_devices = len(unique_devices)

    fig, ax = plt.subplots(figsize=(max(12, n_devices * 1.5), 8))

    positions_actual = []
    positions_pred = []
    data_actual = []
    data_pred = []

    for i, device in enumerate(unique_devices):
        mask = [d == device for d in devices]
        data_actual.append([l for l, m in zip(labels, mask) if m])
        data_pred.append([p for p, m in zip(predictions, mask) if m])
        positions_actual.append(i * 3)
        positions_pred.append(i * 3 + 1)

    bp_actual = ax.boxplot(
        data_actual,
        positions=positions_actual,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.7),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )

    bp_pred = ax.boxplot(
        data_pred,
        positions=positions_pred,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor='darkorange', alpha=0.7),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )

    tick_positions = [i * 3 + 0.5 for i in range(n_devices)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(unique_devices, rotation=45, ha='right', fontsize=10)

    ax.set_ylabel("Soiling Index", fontsize=14)
    ax.set_xlabel("Device ID", fontsize=14)
    ax.set_title(f"Actual vs Predicted Soiling by Device ID (n={n_devices} devices)",
                 fontsize=16, fontweight="bold")

    ax.legend([bp_actual["boxes"][0], bp_pred["boxes"][0]], ["Actual", "Predicted"],
              loc="upper right", fontsize=12)

    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def main():
    chip_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    print(f"Total chips: {len(chip_dirs)}")

    labels, predictions, devices, failed = [], [], [], []

    for chip in tqdm(chip_dirs, desc="Processing"):
        try:
            label, pred, device_id = load_chip_values(chip)
            labels.append(label)
            predictions.append(pred)
            devices.append(device_id)
        except Exception as exc:
            failed.append((chip.name, str(exc)))

    labels = np.array(labels)
    predictions = np.array(predictions)
    devices = np.array(devices)

    valid = ~(np.isnan(labels) | np.isnan(predictions))
    labels = labels[valid].tolist()
    predictions = predictions[valid].tolist()
    devices = devices[valid].tolist()

    print(f"Valid: {len(labels)}, Devices: {len(set(devices))}")

    # Sample devices
    unique_devices = list(set(devices))
    if len(unique_devices) > N_SAMPLE_DEVICES:
        sampled_devices = random.sample(unique_devices, N_SAMPLE_DEVICES)
    else:
        sampled_devices = unique_devices

    sampled_mask = [d in sampled_devices for d in devices]
    sampled_labels = [l for l, m in zip(labels, sampled_mask) if m]
    sampled_preds = [p for p, m in zip(predictions, sampled_mask) if m]
    sampled_devs = [d for d, m in zip(devices, sampled_mask) if m]

    # Plot 1: Sampled devices
    plot_boxplot_by_device(sampled_labels, sampled_preds, sampled_devs,
                           "soiling_tif_boxplot_sampled_devices.png")

    # Plot 2: All devices
    plot_boxplot_by_device(labels, predictions, devices,
                           "soiling_tif_boxplot_all_devices.png")

    print("Done!")


if __name__ == "__main__":
    main()
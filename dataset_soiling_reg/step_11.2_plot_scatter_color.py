"""
Evaluate regression predictions on all windows in predict set (GeoJSON-based).
Computes aggregate metrics and generates scatter + error plots.
Color-coded by device.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
import random

# === CONFIG ===
BASE_DIR = Path("dataset_soiling_reg/windows/predict")
N_SAMPLE_DEVICES = 10
SEED = 42

random.seed(SEED)


def extract_device_id(chip_name: str) -> str:
    """Extract device ID from chip name (format: PLANT_DEVICE_DATE)."""
    parts = chip_name.split("_")
    for i in range(len(parts) - 1, -1, -1):
        if len(parts[i]) == 10 and parts[i].count("-") == 2:
            return "_".join(parts[:i])
    return chip_name


def load_geojson_value(path, key="soiling_normalized"):
    """Load a single scalar value from a GeoJSON FeatureCollection."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if len(features) == 0:
        return np.nan

    return features[0]["properties"].get(key, np.nan)


def load_chip_values(chip_path):
    """Load label and prediction values for one chip."""
    base = chip_path / "layers"

    label_path = base / "label" / "data.geojson"
    pred_path = base / "output" / "data.geojson"

    if not label_path.exists() or not pred_path.exists():
        raise FileNotFoundError("Missing label or output GeoJSON")

    label_val = load_geojson_value(label_path)
    pred_val = load_geojson_value(pred_path)
    device_id = extract_device_id(chip_path.name)

    return label_val, pred_val, device_id


def compute_metrics(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)

    errors = predictions - labels

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def plot_scatter_by_device(labels, predictions, devices, metrics, output_file, title_suffix=""):
    """Create scatter plot color-coded by device."""
    fig, ax = plt.subplots(figsize=(12, 10))

    unique_devices = sorted(set(devices))
    n_devices = len(unique_devices)

    if n_devices <= 20:
        colors = cm.tab20(np.linspace(0, 1, 20))
    else:
        colors = np.vstack([
            cm.tab20(np.linspace(0, 1, 20)),
            cm.tab20b(np.linspace(0, 1, 20)),
            cm.tab20c(np.linspace(0, 1, 20))
        ])

    device_to_color = {dev: colors[i % len(colors)] for i, dev in enumerate(unique_devices)}

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

    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual (Label)", fontsize=14)
    ax.set_ylabel("Predicted", fontsize=14)
    ax.set_title(f"Soiling Index: Predicted vs Actual{title_suffix}\n(n={len(labels)} chips, {n_devices} devices)",
                 fontsize=16, fontweight="bold")

    text = (
        f"R² = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"n = {len(labels)}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if n_devices <= 10:
        ax.legend(loc="lower right", fontsize=9, ncol=1)
    else:
        handles, labels_legend = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels_legend[-1]], loc="lower right", fontsize=12)
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


def plot_error_distribution(labels, predictions):
    errors = np.array(predictions) - np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=40, edgecolor="black", alpha=0.7)

    ax.axvline(0, color="r", linestyle="--", linewidth=2)
    ax.axvline(errors.mean(), color="g", linewidth=2)

    ax.set_xlabel("Prediction Error (Pred − Label)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (n={len(errors)})")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("soiling_reg_prediction_error_dist.png", dpi=150)
    plt.close()


def main():
    chip_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    print(f"Total chips: {len(chip_dirs)}")

    labels = []
    predictions = []
    devices = []
    failed = []

    for chip in tqdm(chip_dirs, desc="Processing chips"):
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

    print(f"Valid samples: {len(labels)}")
    print(f"Unique devices: {len(set(devices))}")
    if failed:
        print(f"Failed chips: {len(failed)}")

    if len(labels) == 0:
        raise RuntimeError("No valid samples found")

    metrics = compute_metrics(labels, predictions)

    print("\nAGGREGATE METRICS (ALL CHIPS)")
    print(f"  Samples: {len(labels)}")
    print(f"  MSE:     {metrics['mse']:.6f}")
    print(f"  RMSE:    {metrics['rmse']:.6f}")
    print(f"  MAE:     {metrics['mae']:.6f}")
    print(f"  R²:      {metrics['r2']:.6f}")

    # All devices plot
    plot_scatter_by_device(labels, predictions, devices, metrics,
                           "soiling_reg_prediction_scatter_all_devices.png", "")

    # Sampled devices plot
    unique_devices = list(set(devices))
    if len(unique_devices) > N_SAMPLE_DEVICES:
        sampled_devices = random.sample(unique_devices, N_SAMPLE_DEVICES)
    else:
        sampled_devices = unique_devices

    print(f"\nSampled {len(sampled_devices)} devices:")
    for dev in sorted(sampled_devices):
        print(f"  - {dev}: {devices.count(dev)} chips")

    sampled_mask = [d in sampled_devices for d in devices]
    sampled_labels = [l for l, m in zip(labels, sampled_mask) if m]
    sampled_preds = [p for p, m in zip(predictions, sampled_mask) if m]
    sampled_devs = [d for d, m in zip(devices, sampled_mask) if m]

    metrics_sampled = compute_metrics(sampled_labels, sampled_preds)

    print(f"\nAGGREGATE METRICS (SAMPLED {len(sampled_devices)} DEVICES)")
    print(f"  Samples: {len(sampled_labels)}")
    print(f"  MSE:     {metrics_sampled['mse']:.6f}")
    print(f"  RMSE:    {metrics_sampled['rmse']:.6f}")
    print(f"  MAE:     {metrics_sampled['mae']:.6f}")
    print(f"  R²:      {metrics_sampled['r2']:.6f}")

    plot_scatter_by_device(sampled_labels, sampled_preds, sampled_devs, metrics_sampled,
                           "soiling_reg_prediction_scatter_10_devices.png",
                           f" (Sampled {len(sampled_devices)} Devices)")

    plot_error_distribution(labels, predictions)

    print("\nDone!")


if __name__ == "__main__":
    main()
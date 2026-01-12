"""
Evaluate regression predictions on all windows in predict set (GeoJSON-based).
Computes aggregate metrics and generates scatter + error plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# === CONFIG ===
BASE_DIR = Path("dataset_soiling_reg/windows/predict")


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

    return label_val, pred_val


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


def plot_scatter(labels, predictions, metrics):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(labels, predictions, s=20, alpha=0.5, edgecolors="none")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2)

    ax.set_xlabel("Actual (Label)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual (n={len(labels)})")

    text = (
        f"R² = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("soiling_reg_prediction_scatter_all.png", dpi=150)
    plt.close()



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
    failed = []

    for chip in tqdm(chip_dirs, desc="Processing chips"):
        try:
            label, pred = load_chip_values(chip)
            labels.append(label)
            predictions.append(pred)
        except Exception as exc:
            failed.append((chip.name, str(exc)))

    labels = np.array(labels)
    predictions = np.array(predictions)

    valid = ~(np.isnan(labels) | np.isnan(predictions))
    labels = labels[valid]
    predictions = predictions[valid]

    print(f"Valid samples: {len(labels)}")
    if failed:
        print(f"Failed chips: {len(failed)}")

    if len(labels) == 0:
        raise RuntimeError("No valid samples found")

    metrics = compute_metrics(labels, predictions)

    print("\nAGGREGATE METRICS")
    print(f"  Samples: {len(labels)}")
    print(f"  MSE:     {metrics['mse']:.6f}")
    print(f"  RMSE:    {metrics['rmse']:.6f}")
    print(f"  MAE:     {metrics['mae']:.6f}")
    print(f"  R²:      {metrics['r2']:.6f}")

    plot_scatter(labels, predictions, metrics)
    plot_error_distribution(labels, predictions)

    print("\nSaved plots:")
    print("  soiling_reg_prediction_scatter_all.png")
    print("  soiling_reg_prediction_error_dist.png")


if __name__ == "__main__":
    main()

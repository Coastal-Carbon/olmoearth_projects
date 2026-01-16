"""
Compare variance/spread of actual vs predicted soiling for both models.
"""

import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

# === CONFIG ===
GEOJSON_DIR = Path("dataset_soiling_reg/windows/predict")
TIF_DIR = Path("dataset_soiling/windows/predict")


def load_geojson_values(chip_path):
    base = chip_path / "layers"
    label_path = base / "label" / "data.geojson"
    pred_path = base / "output" / "data.geojson"

    if not label_path.exists() or not pred_path.exists():
        raise FileNotFoundError()

    with open(label_path) as f:
        label_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    label = label_data["features"][0]["properties"].get("soiling_normalized", np.nan)
    pred = pred_data["features"][0]["properties"].get("soiling_normalized", np.nan)
    return label, pred


def load_tif_values(chip_path):
    base = chip_path / "layers"
    label_path = base / "label/B1/geotiff.tif"
    pred_path = base / "output/output/geotiff.tif"

    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    valid_label = label[label >= 0]
    valid_pred = pred[pred >= 0]

    label_val = np.mean(valid_label) if len(valid_label) > 0 else np.nan
    pred_val = np.mean(valid_pred) if len(valid_pred) > 0 else np.nan
    return label_val, pred_val


def load_all_data(base_dir, loader_func, name):
    chip_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    labels, preds = [], []

    for chip in tqdm(chip_dirs, desc=f"Loading {name}"):
        try:
            label, pred = loader_func(chip)
            labels.append(label)
            preds.append(pred)
        except:
            pass

    labels = np.array(labels)
    preds = np.array(preds)
    valid = ~(np.isnan(labels) | np.isnan(preds))
    return labels[valid], preds[valid]


def compute_spread_stats(actual, predicted, name):
    """Compute and print variance statistics."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Samples:         {len(actual)}")
    print(f"\n  Actual:")
    print(f"    Mean:          {np.mean(actual):.4f}")
    print(f"    Std:           {np.std(actual):.4f}")
    print(f"    Variance:      {np.var(actual):.6f}")
    print(f"    Range:         [{np.min(actual):.4f}, {np.max(actual):.4f}]")
    print(f"    IQR:           {np.percentile(actual, 75) - np.percentile(actual, 25):.4f}")

    print(f"\n  Predicted:")
    print(f"    Mean:          {np.mean(predicted):.4f}")
    print(f"    Std:           {np.std(predicted):.4f}")
    print(f"    Variance:      {np.var(predicted):.6f}")
    print(f"    Range:         [{np.min(predicted):.4f}, {np.max(predicted):.4f}]")
    print(f"    IQR:           {np.percentile(predicted, 75) - np.percentile(predicted, 25):.4f}")

    var_ratio = np.var(predicted) / np.var(actual)
    print(f"\n  Variance Ratio (Pred/Actual): {var_ratio:.4f}")
    if var_ratio > 1:
        print(f"    -> Predictions have MORE variance ({(var_ratio-1)*100:.1f}% more)")
    else:
        print(f"    -> Predictions have LESS variance ({(1-var_ratio)*100:.1f}% less)")

    stat, pval = stats.levene(actual, predicted)
    print(f"\n  Levene's Test: stat={stat:.4f}, p={pval:.4f}")
    if pval < 0.05:
        print(f"    -> Variances are significantly DIFFERENT")
    else:
        print(f"    -> Variances are NOT significantly different")

    return {
        "actual_mean": np.mean(actual),
        "pred_mean": np.mean(predicted),
        "actual_std": np.std(actual),
        "pred_std": np.std(predicted),
        "actual_var": np.var(actual),
        "pred_var": np.var(predicted),
        "var_ratio": var_ratio,
        "levene_stat": stat,
        "levene_p": pval
    }


def plot_spread_comparison(data_dict, stats_dict, output_file):
    """Box + violin plot comparing spread across both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (name, (actual, predicted)) in zip(axes, data_dict.items()):
        s = stats_dict[name]

        parts = ax.violinplot([actual, predicted], positions=[0, 1], showmedians=True)

        parts['bodies'][0].set_facecolor('steelblue')
        parts['bodies'][1].set_facecolor('darkorange')
        for part in parts['bodies']:
            part.set_alpha(0.7)

        bp = ax.boxplot([actual, predicted], positions=[0, 1], widths=0.15,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('darkorange')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Actual', 'Predicted'])
        ax.set_ylabel('Soiling Index')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis='y')

        stats_text = (
            f"Actual:  μ={s['actual_mean']:.3f}, σ={s['actual_std']:.3f}\n"
            f"Pred:    μ={s['pred_mean']:.3f}, σ={s['pred_std']:.3f}\n"
            f"Var Ratio: {s['var_ratio']:.3f}\n"
            f"Levene p: {s['levene_p']:.2e}\n"
            f"{'*** Sig. Different' if s['levene_p'] < 0.05 else 'Not Sig.'}"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                family='monospace')

    plt.suptitle('Spread Comparison: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_file}")


def main():
    # Load both datasets
    geojson_actual, geojson_pred = load_all_data(GEOJSON_DIR, load_geojson_values, "RegressionHead")
    tif_actual, tif_pred = load_all_data(TIF_DIR, load_tif_values, "PerPixelRegressionHead")

    # Compute stats
    geojson_stats = compute_spread_stats(geojson_actual, geojson_pred, "RegressionHead")
    tif_stats = compute_spread_stats(tif_actual, tif_pred, "PerPixelRegressionHead")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  RegressionHead variance ratio:         {geojson_stats['var_ratio']:.4f}")
    print(f"  PerPixelRegressionHead variance ratio: {tif_stats['var_ratio']:.4f}")

    # Plot
    plot_spread_comparison(
        {
            "RegressionHead": (geojson_actual, geojson_pred),
            "PerPixelRegressionHead": (tif_actual, tif_pred)
        },
        {
            "RegressionHead": geojson_stats,
            "PerPixelRegressionHead": tif_stats
        },
        "soiling_spread_comparison.png"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
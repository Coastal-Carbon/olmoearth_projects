"""
Soiling Index Distribution Analysis

Analyzes soiling index distribution with cloud cover filtering,
percentile clipping, and normalization (MinMax, Z-Score).
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FILE = "dataset_soiling/soiling_with_stac.parquet"
OUTPUT_FILE = "soiling_index_distribution.png"
CLOUD_COVER_THRESHOLD = 1  # Keep only rows with cloud_cover < this value (%)
CLIP_LOWER_PERCENTILE = 0.05
CLIP_UPPER_PERCENTILE = 0.95


def load_and_filter(filepath: str, cloud_threshold: float) -> pd.DataFrame:
    """Load data and apply cloud cover filter."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)
    
    original_count = len(df)
    df_filtered = df[df["cloud_cover"] < cloud_threshold]
    filtered_count = len(df_filtered)
    
    logger.info(
        f"Cloud filter (< {cloud_threshold}%): "
        f"{original_count:,} -> {filtered_count:,} rows "
        f"({filtered_count / original_count * 100:.2f}% retained)"
    )
    
    return df_filtered


def compute_stats(series: pd.Series, name: str) -> dict:
    """Compute and log descriptive statistics."""
    stats = {
        "count": len(series),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
    }
    
    logger.info(
        f"{name}: n={stats['count']:,}, "
        f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
    )
    
    return stats


def clip_series(series: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
    """Clip series at specified percentiles."""
    p_lower = series.quantile(lower_pct)
    p_upper = series.quantile(upper_pct)
    
    logger.info(
        f"Clipping at {lower_pct * 100:.0f}th ({p_lower:.4f}) "
        f"and {upper_pct * 100:.0f}th ({p_upper:.4f}) percentiles"
    )
    
    return series.clip(lower=p_lower, upper=p_upper)


def normalize_series(series: pd.Series) -> tuple:
    """Apply MinMax and Z-Score normalization."""
    values = series.values.reshape(-1, 1)
    minmax = MinMaxScaler().fit_transform(values).flatten()
    zscore = StandardScaler().fit_transform(values).flatten()
    return minmax, zscore


def print_percentiles(series: pd.Series):
    """Print percentile distribution."""
    print("\n" + "=" * 60)
    print("PERCENTILES")
    print("=" * 60)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:>2}th: {series.quantile(p / 100):.6f}")
    print("=" * 60)


def print_summary(original: pd.Series, clipped: pd.Series, minmax, zscore):
    """Print comprehensive statistics summary."""
    print("\n" + "=" * 60)
    print("SOILING INDEX STATISTICS")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Original':>12} {'Clipped':>12} {'MinMax':>12} {'Z-Score':>12}")
    print("-" * 60)
    print(f"  {'Count':<20} {len(original):>12,} {len(clipped):>12,} {len(minmax):>12,} {len(zscore):>12,}")
    print(f"  {'Min':<20} {original.min():>12.4f} {clipped.min():>12.4f} {minmax.min():>12.4f} {zscore.min():>12.4f}")
    print(f"  {'Max':<20} {original.max():>12.4f} {clipped.max():>12.4f} {minmax.max():>12.4f} {zscore.max():>12.4f}")
    print(f"  {'Mean':<20} {original.mean():>12.4f} {clipped.mean():>12.4f} {minmax.mean():>12.4f} {zscore.mean():>12.4f}")
    print(f"  {'Std':<20} {original.std():>12.4f} {clipped.std():>12.4f} {minmax.std():>12.4f} {zscore.std():>12.4f}")
    print("=" * 60)


def create_distribution_plots(
    original: pd.Series,
    clipped: pd.Series,
    minmax,
    zscore,
    cloud_threshold: float,
    output_path: str
):
    """Create 2x2 distribution plot."""
    logger.info("Generating distribution plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Soiling Index Distribution (cloud_cover < {cloud_threshold}%)",
        fontsize=14,
        fontweight="bold"
    )
    
    plot_configs = [
        (axes[0, 0], original, "steelblue", "Original"),
        (axes[0, 1], clipped, "forestgreen", "Clipped 5th-95th"),
        (axes[1, 0], minmax, "purple", "Min-Max Normalized"),
        (axes[1, 1], zscore, "darkorange", "Z-Score Normalized"),
    ]
    
    for ax, data, color, title in plot_configs:
        ax.hist(data, bins=100, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(
            data.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {data.mean():.4f}"
        )
        ax.set_xlabel("Soiling Index")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of Soiling Index ({title})")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved plot to {output_path}")


def main():
    logger.info("Starting soiling index distribution analysis")
    
    df = load_and_filter(INPUT_FILE, CLOUD_COVER_THRESHOLD)
    soiling = df["SOILING_INDEX_DAILY"]
    
    compute_stats(soiling, "Original")
    print_percentiles(soiling)
    
    soiling_clipped = clip_series(soiling, CLIP_LOWER_PERCENTILE, CLIP_UPPER_PERCENTILE)
    compute_stats(soiling_clipped, "Clipped")
    
    soiling_minmax, soiling_zscore = normalize_series(soiling_clipped)
    compute_stats(pd.Series(soiling_minmax), "MinMax")
    compute_stats(pd.Series(soiling_zscore), "Z-Score")
    
    print_summary(soiling, soiling_clipped, soiling_minmax, soiling_zscore)
    
    create_distribution_plots(
        soiling,
        soiling_clipped,
        soiling_minmax,
        soiling_zscore,
        CLOUD_COVER_THRESHOLD,
        OUTPUT_FILE
    )
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
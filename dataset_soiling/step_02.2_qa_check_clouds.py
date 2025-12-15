"""
Cloud Cover Distribution Analysis

Analyzes cloud cover distribution across STAC-matched soiling data,
with overall and zoomed (0-1%) breakdowns by device and plant.
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FILE = "dataset_soiling/soiling_with_stac.parquet"
OUTPUT_OVERALL = "cloud_distribution.png"
OUTPUT_ZOOMED = "cloud_distribution_zoomed.png"

CLOUD_BINS = [0, 1, 5, 10, 20, 50, 100]
CLOUD_LABELS = ["0-1%", "1-5%", "5-10%", "10-20%", "20-50%", "50-100%"]

ZOOM_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ZOOM_LABELS = [
    "0-0.1%", "0.1-0.2%", "0.2-0.3%", "0.3-0.4%", "0.4-0.5%",
    "0.5-0.6%", "0.6-0.7%", "0.7-0.8%", "0.8-0.9%", "0.9-1%"
]

TOP_DEVICES_COUNT = 20


def load_stac_data(filepath: str) -> pd.DataFrame:
    """Load data and filter to rows with STAC matches."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)
    
    total = len(df)
    with_stac = df[df["item_id"].notna()].copy()
    
    logger.info(f"STAC coverage: {len(with_stac):,} / {total:,} rows ({len(with_stac) / total * 100:.1f}%)")
    return with_stac


def bin_cloud_cover(df: pd.DataFrame, bins: list, labels: list, col_name: str) -> pd.DataFrame:
    """Add binned cloud cover column."""
    df[col_name] = pd.cut(df["cloud_cover"], bins=bins, labels=labels, include_lowest=True)
    return df


def print_distribution(dist: pd.Series, total: int, title: str):
    """Print distribution summary."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for label, count in dist.items():
        pct = count / total * 100
        print(f"  {label:<12} {count:>8,} days ({pct:>5.1f}%)")
    print("-" * 60)
    print(f"  {'Total':<12} {total:>8,} days")
    print("=" * 60)


def get_device_breakdown(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Get per-device cloud cover breakdown."""
    device_cloud = df.groupby(["ID_PLANT", "ID_DEVICE", bin_col]).size().unstack(fill_value=0)
    device_cloud["total"] = device_cloud.sum(axis=1)
    device_cloud = device_cloud.reset_index()
    return device_cloud


def print_top_devices(device_df: pd.DataFrame, sort_col: str, n: int):
    """Print top devices by specified column."""
    sorted_df = device_df.sort_values(sort_col, ascending=False).head(n)
    print(f"\n=== TOP {n} DEVICES BY LOW CLOUD DAYS ({sort_col}) ===")
    print(sorted_df.to_string(index=False))


def plot_distribution(
    dist: pd.Series,
    title: str,
    color: str,
    output_path: str,
    figsize: tuple = (10, 6)
):
    """Create and save bar chart of distribution."""
    logger.info(f"Generating plot: {title}")
    
    fig, ax = plt.subplots(figsize=figsize)
    dist.plot(kind="bar", ax=ax, color=color, edgecolor="black")
    
    ax.set_xlabel("Cloud Cover")
    ax.set_ylabel("Number of Days")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    
    for i, v in enumerate(dist):
        ax.text(i, v + (dist.max() * 0.02), f"{v:,}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved: {output_path}")


def print_summary(with_stac: pd.DataFrame, low_cloud: pd.DataFrame):
    """Print overall summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total days with STAC match:      {len(with_stac):>10,}")
    print(f"  Days with cloud <= 1%:           {len(low_cloud):>10,}")
    print(f"  Low cloud ratio:                 {len(low_cloud) / len(with_stac) * 100:>10.2f}%")
    print()
    print(f"  Unique plants:                   {with_stac['ID_PLANT'].nunique():>10}")
    print(f"  Unique devices:                  {with_stac['ID_DEVICE'].nunique():>10}")
    print("=" * 60)


def main():
    logger.info("Starting cloud cover distribution analysis")
    
    with_stac = load_stac_data(INPUT_FILE)
    
    # Overall distribution
    with_stac = bin_cloud_cover(with_stac, CLOUD_BINS, CLOUD_LABELS, "cloud_bin")
    cloud_dist = with_stac["cloud_bin"].value_counts().sort_index()
    print_distribution(cloud_dist, len(with_stac), "CLOUD COVER DISTRIBUTION (OVERALL)")
    
    # Device breakdown
    device_cloud = get_device_breakdown(with_stac, "cloud_bin")
    print_top_devices(device_cloud, "0-1%", TOP_DEVICES_COUNT)
    
    # Zoomed distribution (0-1%)
    low_cloud = with_stac[with_stac["cloud_cover"] <= 1].copy()
    low_cloud = bin_cloud_cover(low_cloud, ZOOM_BINS, ZOOM_LABELS, "cloud_bin_zoom")
    cloud_dist_zoom = low_cloud["cloud_bin_zoom"].value_counts().sort_index()
    print_distribution(cloud_dist_zoom, len(low_cloud), "CLOUD COVER DISTRIBUTION (0-1% ZOOMED)")
    
    print_summary(with_stac, low_cloud)
    
    # Plots
    plot_distribution(
        cloud_dist,
        "Distribution of Cloud Cover Across All STAC Items",
        "steelblue",
        OUTPUT_OVERALL
    )
    
    plot_distribution(
        cloud_dist_zoom,
        "Distribution of Cloud Cover (0-1% Zoomed)",
        "forestgreen",
        OUTPUT_ZOOMED,
        figsize=(12, 6)
    )
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
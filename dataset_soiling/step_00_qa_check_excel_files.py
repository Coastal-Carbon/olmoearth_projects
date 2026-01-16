"""
Soiling-Portion Data Matcher

Compares ID_PLANT and ID_DEVICE between soiling and portion datasets
to identify matching and orphaned records for data quality assessment.
"""

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
SOILING_FILE = "dataset_soiling/soiling_data (1).xlsx"
PORTION_FILE = "dataset_soiling/portion_json (1).xlsx"


def load_and_normalize(filepath: str, name: str) -> pd.DataFrame:
    """Load Excel file and normalize ID columns to uppercase."""
    logger.info(f"Loading {name} data from {filepath}")
    df = pd.read_excel(filepath)
    df["ID_PLANT_UPPER"] = df["ID_PLANT"].str.upper().str.strip()
    df["ID_DEVICE_UPPER"] = df["ID_DEVICE"].str.upper().str.strip()
    logger.info(f"Loaded {len(df):,} rows from {name}")
    return df


def get_unique_sets(df: pd.DataFrame) -> tuple[set, set]:
    """Extract unique plants and devices from dataframe."""
    plants = set(df["ID_PLANT_UPPER"].unique())
    devices = set(df["ID_DEVICE_UPPER"].unique())
    return plants, devices


def analyze_overlap(set_a: set, set_b: set, name: str) -> dict:
    """Analyze overlap between two sets."""
    result = {
        "in_both": set_a & set_b,
        "only_in_soiling": set_a - set_b,
        "only_in_portion": set_b - set_a,
    }
    logger.info(
        f"{name} overlap: {len(result['in_both'])} shared, "
        f"{len(result['only_in_soiling'])} soiling-only, "
        f"{len(result['only_in_portion'])} portion-only"
    )
    return result


def print_orphaned_plants(soiling: pd.DataFrame, portion: pd.DataFrame, plant_analysis: dict):
    """Print details of plants missing from one dataset."""
    print("\n" + "=" * 60)
    print("PLANTS ONLY IN SOILING (missing geometry)")
    print("=" * 60)
    for p in sorted(plant_analysis["only_in_soiling"]):
        mask = soiling["ID_PLANT_UPPER"] == p
        device_count = soiling.loc[mask, "ID_DEVICE_UPPER"].nunique()
        row_count = mask.sum()
        print(f"  {p:<30} | {device_count:>3} devices | {row_count:>6} rows")

    print("\n" + "=" * 60)
    print("PLANTS ONLY IN PORTION (missing soiling data)")
    print("=" * 60)
    for p in sorted(plant_analysis["only_in_portion"]):
        mask = portion["ID_PLANT_UPPER"] == p
        device_count = portion.loc[mask, "ID_DEVICE_UPPER"].nunique()
        print(f"  {p:<30} | {device_count:>3} devices")


def print_orphaned_devices(soiling: pd.DataFrame, portion: pd.DataFrame, device_analysis: dict):
    """Print details of devices missing from one dataset."""
    print("\n" + "=" * 60)
    print("DEVICES ONLY IN SOILING (no geometry)")
    print("=" * 60)
    for d in sorted(device_analysis["only_in_soiling"]):
        mask = soiling["ID_DEVICE_UPPER"] == d
        plant = soiling.loc[mask, "ID_PLANT"].iloc[0]
        row_count = mask.sum()
        print(f"  {d:<40} | {plant:<20} | {row_count:>6} rows")

    print("\n" + "=" * 60)
    print("DEVICES ONLY IN PORTION (no soiling data)")
    print("=" * 60)
    for d in sorted(device_analysis["only_in_portion"]):
        mask = portion["ID_DEVICE_UPPER"] == d
        plant = portion.loc[mask, "ID_PLANT"].iloc[0]
        print(f"  {d:<40} | {plant:<20}")


def print_summary(soiling: pd.DataFrame, portion: pd.DataFrame, plant_analysis: dict, device_analysis: dict):
    """Print final summary statistics."""
    total_soiling_rows = len(soiling)
    orphaned_device_rows = soiling[soiling["ID_DEVICE_UPPER"].isin(device_analysis["only_in_soiling"])].shape[0]
    matchable_rows = total_soiling_rows - orphaned_device_rows
    match_rate = (matchable_rows / total_soiling_rows) * 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total soiling rows:              {total_soiling_rows:>10,}")
    print(f"  Rows with matching geometry:     {matchable_rows:>10,}")
    print(f"  Rows missing geometry:           {orphaned_device_rows:>10,}")
    print(f"  Match rate:                      {match_rate:>10.2f}%")
    print()
    print(f"  Plants matched:                  {len(plant_analysis['in_both']):>10}")
    print(f"  Plants soiling-only:             {len(plant_analysis['only_in_soiling']):>10}")
    print(f"  Plants portion-only:             {len(plant_analysis['only_in_portion']):>10}")
    print()
    print(f"  Devices matched:                 {len(device_analysis['in_both']):>10}")
    print(f"  Devices soiling-only:            {len(device_analysis['only_in_soiling']):>10}")
    print(f"  Devices portion-only:            {len(device_analysis['only_in_portion']):>10}")
    print("=" * 60)


def main():
    logger.info("Starting soiling-portion data comparison")

    soiling = load_and_normalize(SOILING_FILE, "soiling")
    portion = load_and_normalize(PORTION_FILE, "portion")

    soiling_plants, soiling_devices = get_unique_sets(soiling)
    portion_plants, portion_devices = get_unique_sets(portion)

    logger.info(f"Soiling: {len(soiling_plants)} plants, {len(soiling_devices)} devices")
    logger.info(f"Portion: {len(portion_plants)} plants, {len(portion_devices)} devices")

    plant_analysis = analyze_overlap(soiling_plants, portion_plants, "Plants")
    device_analysis = analyze_overlap(soiling_devices, portion_devices, "Devices")

    print_orphaned_plants(soiling, portion, plant_analysis)
    print_orphaned_devices(soiling, portion, device_analysis)
    print_summary(soiling, portion, plant_analysis, device_analysis)

    logger.info("Comparison complete")


if __name__ == "__main__":
    main()
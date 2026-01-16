"""
Soiling-Portion Data Merger

Joins soiling data with portion geometry data on ID_DEVICE,
with plant name mismatch detection for data quality tracking.
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
OUTPUT_FILE = "dataset_soiling/merged_soiling.parquet"


def load_and_normalize(filepath: str, name: str) -> pd.DataFrame:
    """Load Excel file and normalize ID columns to uppercase."""
    logger.info(f"Loading {name} data from {filepath}")
    df = pd.read_excel(filepath)
    df["ID_PLANT_UPPER"] = df["ID_PLANT"].str.upper().str.strip()
    df["ID_DEVICE_UPPER"] = df["ID_DEVICE"].str.upper().str.strip()
    logger.info(f"Loaded {len(df):,} rows, {df['ID_DEVICE_UPPER'].nunique()} unique devices")
    return df


def deduplicate_portion(portion: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate portion data on ID_DEVICE (keep first occurrence)."""
    before = len(portion)
    portion_deduped = portion.drop_duplicates(subset=["ID_DEVICE_UPPER"], keep="first")
    after = len(portion_deduped)
    
    if before != after:
        logger.warning(f"Removed {before - after} duplicate devices from portion")
    else:
        logger.info("No duplicate devices in portion")
    
    return portion_deduped


def check_plant_mismatches(soiling: pd.DataFrame, portion: pd.DataFrame) -> int:
    """Log devices where plant names don't match between datasets."""
    soiling_devices = set(soiling["ID_DEVICE_UPPER"].unique())
    portion_devices = set(portion["ID_DEVICE_UPPER"].unique())
    common_devices = soiling_devices & portion_devices
    
    mismatch_count = 0
    mismatch_rows = 0
    
    for device in common_devices:
        soiling_plants = set(soiling[soiling["ID_DEVICE_UPPER"] == device]["ID_PLANT_UPPER"].unique())
        portion_plants = set(portion[portion["ID_DEVICE_UPPER"] == device]["ID_PLANT_UPPER"].unique())
        
        if soiling_plants != portion_plants:
            mismatch_count += 1
            mismatch_rows += len(soiling[soiling["ID_DEVICE_UPPER"] == device])
    
    if mismatch_count > 0:
        logger.warning(
            f"Plant name mismatch: {mismatch_count} devices ({mismatch_rows:,} rows) "
            f"have different plant names between soiling and portion"
        )
    
    return mismatch_count


def merge_datasets(soiling: pd.DataFrame, portion: pd.DataFrame) -> pd.DataFrame:
    """Merge soiling with portion data on ID_DEVICE only."""
    logger.info("Merging datasets on ID_DEVICE")
    
    merged = soiling.merge(
        portion[["ID_DEVICE_UPPER", "portion"]],
        on="ID_DEVICE_UPPER",
        how="inner"
    )
    merged = merged.drop(columns=["ID_PLANT_UPPER", "ID_DEVICE_UPPER"])
    
    dropped_rows = len(soiling) - len(merged)
    logger.info(f"Merge complete: {len(merged):,} rows retained, {dropped_rows:,} rows dropped")
    
    return merged


def print_summary(soiling: pd.DataFrame, portion: pd.DataFrame, merged: pd.DataFrame):
    """Print merge summary statistics."""
    retention_rate = (len(merged) / len(soiling)) * 100
    
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"  Input soiling rows:              {len(soiling):>10,}")
    print(f"  Input portion rows:              {len(portion):>10,}")
    print(f"  Output merged rows:              {len(merged):>10,}")
    print(f"  Retention rate:                  {retention_rate:>10.2f}%")
    print()
    print(f"  Unique plants:                   {merged['ID_PLANT'].nunique():>10}")
    print(f"  Unique devices:                  {merged['ID_DEVICE'].nunique():>10}")
    print(f"  Date range:                      {merged['date'].min()} to {merged['date'].max()}")
    print()
    print(f"  Columns: {merged.columns.tolist()}")
    print("=" * 60)
    print(f"\nSample:\n{merged.head()}")


def main():
    logger.info("Starting soiling-portion merge")
    
    soiling = load_and_normalize(SOILING_FILE, "soiling")
    portion = load_and_normalize(PORTION_FILE, "portion")
    
    portion = deduplicate_portion(portion)
    check_plant_mismatches(soiling, portion)
    
    merged = merge_datasets(soiling, portion)
    
    print_summary(soiling, portion, merged)
    
    logger.info(f"Saving merged data to {OUTPUT_FILE}")
    merged.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(merged):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
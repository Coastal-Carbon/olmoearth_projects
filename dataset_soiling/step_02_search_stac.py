"""
STAC Catalog Query for Soiling Data

Queries Microsoft Planetary Computer STAC API for Sentinel-2 L2A imagery
matching device geometries and date ranges, then joins results to soiling data.
Extracts UTM EPSG code from each STAC item for later reprojection.
"""

import pandas as pd
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pystac_client import Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FILE = "dataset_soiling/merged_soiling.parquet"
OUTPUT_FILE = "dataset_soiling/soiling_with_stac.parquet"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
MAX_WORKERS = 8
MAX_RETRIES = 5
RETRY_DELAY = 2
N_PLANTS = None  # None for all, or number to test


def parse_utm_epsg(proj_code: str) -> int | None:
    """Parse UTM EPSG integer from proj:code string (e.g., 'EPSG:32630' -> 32630)."""
    if not proj_code:
        return None
    try:
        return int(proj_code.split(":")[1])
    except (IndexError, ValueError):
        return None


def load_and_prepare_data(filepath: str, n_plants: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged data and prepare device query info."""
    logger.info(f"Loading data from {filepath}")
    merged = pd.read_parquet(filepath)
    
    plants = merged["ID_PLANT"].unique()
    if n_plants:
        plants = plants[:n_plants]
        merged = merged[merged["ID_PLANT"].isin(plants)]
        logger.info(f"Limited to {n_plants} plants for testing")
    
    device_info = merged.groupby("ID_DEVICE").agg(
        plant=("ID_PLANT", "first"),
        min_date=("date", "min"),
        max_date=("date", "max"),
        portion=("portion", "first")
    ).reset_index()
    
    logger.info(f"Plants: {len(plants)}, Devices to query: {len(device_info)}")
    return merged, device_info


def query_stac_for_device(row: pd.Series) -> list[dict]:
    """Query STAC catalog for a single device with retry logic."""
    device_id = row["ID_DEVICE"]
    geometry = json.loads(row["portion"])
    start_date = row["min_date"].strftime("%Y-%m-%d")
    end_date = row["max_date"].strftime("%Y-%m-%d")
    
    for attempt in range(MAX_RETRIES):
        try:
            client = Client.open(STAC_URL)
            search = client.search(
                collections=[COLLECTION],
                intersects=geometry,
                datetime=f"{start_date}/{end_date}",
                max_items=None
            )
            items = list(search.items())
            
            results = []
            for item in items:
                # Extract UTM EPSG from proj:code (e.g., "EPSG:32630" -> 32630)
                proj_code = item.properties.get("proj:code")
                utm_epsg = parse_utm_epsg(proj_code)
                
                results.append({
                    "device_id": device_id,
                    "item_id": item.id,
                    "item_date": item.datetime.date(),
                    "cloud_cover": item.properties.get("eo:cloud_cover"),
                    "utm_epsg": utm_epsg,
                })
            
            logger.info(f"[OK] {device_id}: {len(items)} items ({start_date} to {end_date})")
            return results
            
        except Exception as e:
            logger.warning(f"[RETRY {attempt + 1}/{MAX_RETRIES}] {device_id}: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    logger.error(f"[FAIL] {device_id}: max retries exceeded")
    return []


def query_all_devices(device_info: pd.DataFrame) -> list[dict]:
    """Query STAC for all devices using thread pool."""
    logger.info(f"Starting parallel STAC queries with {MAX_WORKERS} workers")
    all_items = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(query_stac_for_device, row): row["ID_DEVICE"]
            for _, row in device_info.iterrows()
        }
        
        for future in as_completed(futures):
            result = future.result()
            all_items.extend(result)
    
    logger.info(f"Total STAC items retrieved: {len(all_items):,}")
    return all_items


def process_stac_results(all_items: list[dict]) -> pd.DataFrame:
    """Process STAC results and pick best tile per device per date."""
    stac_df = pd.DataFrame(all_items)
    stac_df["item_date"] = pd.to_datetime(stac_df["item_date"])
    
    before = len(stac_df)
    stac_df = stac_df.sort_values("cloud_cover").drop_duplicates(
        subset=["device_id", "item_date"],
        keep="first"
    )
    logger.info(f"Deduplicated STAC items: {before:,} -> {len(stac_df):,} (best cloud cover per device/date)")
    
    # Log UTM EPSG distribution
    utm_counts = stac_df["utm_epsg"].value_counts()
    logger.info(f"UTM zones found: {len(utm_counts)}")
    for epsg, count in utm_counts.head(10).items():
        logger.info(f"  EPSG:{epsg}: {count:,} items")
    
    return stac_df


def merge_stac_with_soiling(merged: pd.DataFrame, stac_df: pd.DataFrame) -> pd.DataFrame:
    """Join STAC results back to soiling data."""
    logger.info("Merging STAC results with soiling data")
    
    merged["date"] = pd.to_datetime(merged["date"])
    final = merged.merge(
        stac_df,
        left_on=["ID_DEVICE", "date"],
        right_on=["device_id", "item_date"],
        how="left"
    )
    final = final.drop(columns=["device_id", "item_date"])
    
    return final


def print_summary(final: pd.DataFrame):
    """Print final merge summary."""
    rows_with_stac = final["item_id"].notna().sum()
    rows_without_stac = final["item_id"].isna().sum()
    coverage_pct = (rows_with_stac / len(final)) * 100
    
    # UTM coverage
    rows_with_utm = final["utm_epsg"].notna().sum()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Total rows:                      {len(final):>10,}")
    print(f"  Rows with STAC item:             {rows_with_stac:>10,}")
    print(f"  Rows without STAC item:          {rows_without_stac:>10,}")
    print(f"  Rows with UTM EPSG:              {rows_with_utm:>10,}")
    print(f"  STAC coverage:                   {coverage_pct:>10.2f}%")
    print()
    print(f"  Unique UTM zones:                {final['utm_epsg'].nunique():>10}")
    print()
    print(f"  Columns: {final.columns.tolist()}")
    print("=" * 60)
    print(f"\nSample:\n{final.head(10)}")


def main():
    logger.info("Starting STAC catalog query pipeline")
    
    merged, device_info = load_and_prepare_data(INPUT_FILE, N_PLANTS)
    
    all_items = query_all_devices(device_info)
    
    if not all_items:
        logger.error("No STAC items retrieved, exiting")
        return
    
    stac_df = process_stac_results(all_items)
    final = merge_stac_with_soiling(merged, stac_df)
    
    print_summary(final)
    
    logger.info(f"Saving to {OUTPUT_FILE}")
    final.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(final):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
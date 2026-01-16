"""
Portion Geometry Size Analysis

Analyzes device portion polygon areas and compares against chip size options.
"""

import json
import logging

import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FILE = "dataset_soiling/soiling_with_stac.parquet"
RESOLUTION = 10
CHIP_SIZES = [32, 64, 128, 256]


def calculate_area_m2(portion_json: str, utm_epsg: int) -> float:
    """Calculate portion area in m² using UTM projection."""
    geom = shape(json.loads(portion_json))
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    return transform(transformer.transform, geom).area


def main():
    logger.info(f"Loading {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    df = df[df["utm_epsg"].notna()].copy()
    df["utm_epsg"] = df["utm_epsg"].astype(int)
    
    devices = df.drop_duplicates(subset=["ID_DEVICE"])[["ID_PLANT", "ID_DEVICE", "portion", "utm_epsg"]]
    devices["area_m2"] = devices.apply(lambda r: calculate_area_m2(r["portion"], r["utm_epsg"]), axis=1)
    
    logger.info(f"Devices: {len(devices)}")
    
    # Stats
    print("\n" + "=" * 60)
    print("PORTION AREA (m²)")
    print("=" * 60)
    print(f"  Min: {devices['area_m2'].min():,.0f}")
    print(f"  Max: {devices['area_m2'].max():,.0f}")
    print(f"  Mean: {devices['area_m2'].mean():,.0f}")
    print(f"  Median: {devices['area_m2'].median():,.0f}")
    
    # Chip analysis
    print("\n" + "=" * 60)
    print("CHIP SIZE COVERAGE")
    print("=" * 60)
    median = devices["area_m2"].median()
    
    for size in CHIP_SIZES:
        chip_area = (size * RESOLUTION) ** 2
        fit = (devices["area_m2"] <= chip_area).sum()
        pct = fit / len(devices) * 100
        ratio = chip_area / median
        print(f"  {size}x{size} @ {RESOLUTION}m = {chip_area:,} m² | {fit}/{len(devices)} ({pct:.1f}%) | {ratio:.1f}x median")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
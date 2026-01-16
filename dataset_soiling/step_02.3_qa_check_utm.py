"""
Quick QA check for UTM EPSG coverage in soiling_with_stac.parquet
"""

import pandas as pd

df = pd.read_parquet("dataset_soiling/soiling_with_stac.parquet")

has_item = df["item_id"].notna()
has_utm = df["utm_epsg"].notna()

print("=== UTM EPSG QA CHECK ===")
print(f"Total rows:                  {len(df):,}")
print(f"Rows with item_id:           {has_item.sum():,}")
print(f"Rows with utm_epsg:          {has_utm.sum():,}")
print()

# Check: if item_id exists, utm_epsg should also exist
missing_utm = has_item & ~has_utm
print(f"Rows with item_id but NO utm_epsg: {missing_utm.sum()}")

if missing_utm.sum() > 0:
    print("\n[WARNING] Some STAC items missing UTM EPSG:")
    print(df[missing_utm][["ID_DEVICE", "date", "item_id"]].head(10))
else:
    print("[OK] All STAC items have UTM EPSG")

print(f"\n=== UNIQUE UTM ZONES ===")
print(f"Count: {df['utm_epsg'].nunique()}")
print(df["utm_epsg"].value_counts().sort_index())
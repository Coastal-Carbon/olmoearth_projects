"""
Patch RSLearn Label TIFs with Correct Values

Reads window names, looks up correct normalized soiling value from metadata,
and overwrites the RSLearn label TIF with the correct value.
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
WINDOWS_DIR = Path("dataset_soiling/windows")
METADATA_FILE = Path("dataset_soiling/data/metadata/train_metadata.json")
GROUPS = ["default", "predict"]  # Window groups to patch


def load_metadata(filepath: Path) -> dict:
    """Load metadata and create lookup by window name."""
    logger.info(f"Loading metadata from {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Create lookup: window_name -> soiling_normalized
    # Window name = filename without "_label.tif"
    lookup = {}
    for entry in data["entries"]:
        window_name = entry["filename"].replace("_label.tif", "")
        lookup[window_name] = entry["soiling_normalized"]
    
    logger.info(f"Loaded {len(lookup)} entries")
    return lookup


def patch_label_tif(tif_path: Path, value: float) -> bool:
    """Overwrite label TIF with correct value, preserving geospatial metadata."""
    try:
        with rasterio.open(tif_path, "r") as src:
            profile = src.profile
            data = src.read(1)
        
        # Create new data with correct value (preserve nodata pixels)
        nodata = profile.get("nodata", -1)
        new_data = np.where(data == nodata, nodata, value).astype(np.float32)
        
        # Write back
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(new_data, 1)
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch {tif_path}: {e}")
        return False


def patch_group(group_dir: Path, lookup: dict) -> tuple[int, int, int]:
    """Patch all windows in a group."""
    patched = 0
    skipped = 0
    failed = 0
    
    if not group_dir.exists():
        logger.warning(f"Group directory not found: {group_dir}")
        return 0, 0, 0
    
    for window_dir in group_dir.iterdir():
        if not window_dir.is_dir():
            continue
        
        window_name = window_dir.name
        label_path = window_dir / "layers/label/B1/geotiff.tif"
        
        if not label_path.exists():
            skipped += 1
            continue
        
        if window_name not in lookup:
            logger.warning(f"No metadata for window: {window_name}")
            skipped += 1
            continue
        
        correct_value = lookup[window_name]
        
        if patch_label_tif(label_path, correct_value):
            patched += 1
        else:
            failed += 1
    
    return patched, skipped, failed


def main():
    logger.info("Starting label patching")
    
    lookup = load_metadata(METADATA_FILE)
    
    total_patched = 0
    total_skipped = 0
    total_failed = 0
    
    for group in GROUPS:
        group_dir = WINDOWS_DIR / group
        logger.info(f"Processing group: {group}")
        
        patched, skipped, failed = patch_group(group_dir, lookup)
        
        logger.info(f"  {group}: patched={patched}, skipped={skipped}, failed={failed}")
        
        total_patched += patched
        total_skipped += skipped
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("PATCH SUMMARY")
    print("=" * 60)
    print(f"  Total patched:                   {total_patched:>10}")
    print(f"  Total skipped:                   {total_skipped:>10}")
    print(f"  Total failed:                    {total_failed:>10}")
    print("=" * 60)
    
    logger.info("Patching complete")


if __name__ == "__main__":
    main()
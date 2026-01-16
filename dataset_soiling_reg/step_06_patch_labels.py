"""
Patch RSLearn Label GeoJSONs - Update existing + Create missing

Reads window names, looks up correct normalized soiling value from metadata,
and either updates existing label GeoJSON or creates new one if missing.
Also creates the 'completed' marker file that rslearn requires.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
WINDOWS_DIR = Path("dataset_soiling_reg/windows")
METADATA_FILE = Path("dataset_soiling_reg/data/metadata/train_metadata.json")
GROUPS = ["default", "predict"]


def load_metadata(filepath: Path) -> dict:
    """Load metadata and create lookup by window name."""
    logger.info(f"Loading metadata from {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    lookup = {}
    for entry in data["entries"]:
        window_name = entry["filename"].replace(".geojson", "")
        lookup[window_name] = {
            "soiling_normalized": entry["soiling_normalized"],
            "soiling_raw": entry["soiling_raw"],
            "soiling_clipped": entry["soiling_clipped"],
        }
    
    logger.info(f"Loaded {len(lookup)} entries")
    return lookup


def load_window_metadata(window_dir: Path) -> dict | None:
    """Load window metadata.json to get bounds and CRS."""
    meta_file = window_dir / "metadata.json"
    if not meta_file.exists():
        return None
    
    with open(meta_file) as f:
        return json.load(f)


def create_label_geojson(window_meta: dict, values: dict) -> dict:
    """Create a new label GeoJSON from window metadata and values."""
    bounds = window_meta.get("bounds", [0, 0, 0, 0])
    crs = window_meta.get("projection", {}).get("crs", "EPSG:32630")
    
    # Compute centroid
    centroid = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
    
    return {
        "type": "FeatureCollection",
        "properties": {
            "crs": crs,
            "x_resolution": 10.0,
            "y_resolution": -10.0,
        },
        "features": [{
            "type": "Feature",
            "properties": {
                "soiling_normalized": values["soiling_normalized"],
                "soiling_raw": values["soiling_raw"],
                "soiling_clipped": values["soiling_clipped"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": centroid
            }
        }]
    }


def patch_or_create_label(window_dir: Path, values: dict) -> str:
    """Patch existing or create new label. Returns: 'patched', 'created', or 'failed'."""
    label_dir = window_dir / "layers" / "label"
    label_path = label_dir / "data.geojson"
    completed_path = label_dir / "completed"
    
    try:
        if label_path.exists():
            # Patch existing
            with open(label_path, "r") as f:
                data = json.load(f)
            for feature in data.get("features", []):
                feature["properties"]["soiling_normalized"] = values["soiling_normalized"]
                feature["properties"]["soiling_raw"] = values["soiling_raw"]
                feature["properties"]["soiling_clipped"] = values["soiling_clipped"]
            with open(label_path, "w") as f:
                json.dump(data, f)
            # Ensure completed marker exists
            completed_path.touch()
            return "patched"
        else:
            # Create new
            window_meta = load_window_metadata(window_dir)
            if not window_meta:
                return "failed"
            
            label_dir.mkdir(parents=True, exist_ok=True)
            data = create_label_geojson(window_meta, values)
            with open(label_path, "w") as f:
                json.dump(data, f)
            # Create completed marker
            completed_path.touch()
            return "created"
    except Exception as e:
        logger.error(f"Failed for {window_dir.name}: {e}")
        return "failed"


def process_group(group_dir: Path, lookup: dict) -> dict:
    """Process all windows in a group."""
    counts = {"patched": 0, "created": 0, "skipped": 0, "failed": 0}
    
    if not group_dir.exists():
        logger.warning(f"Group directory not found: {group_dir}")
        return counts
    
    for window_dir in group_dir.iterdir():
        if not window_dir.is_dir():
            continue
        
        window_name = window_dir.name
        if window_name.endswith(".geojson"):
            window_name = window_name[:-8]
        
        if window_name not in lookup:
            counts["skipped"] += 1
            continue
        
        result = patch_or_create_label(window_dir, lookup[window_name])
        counts[result] += 1
    
    return counts


def main():
    logger.info("Starting label patching/creation")
    
    lookup = load_metadata(METADATA_FILE)
    totals = {"patched": 0, "created": 0, "skipped": 0, "failed": 0}
    
    for group in GROUPS:
        group_dir = WINDOWS_DIR / group
        logger.info(f"Processing group: {group}")
        
        counts = process_group(group_dir, lookup)
        for k, v in counts.items():
            totals[k] += v
        
        logger.info(f"  {group}: patched={counts['patched']}, created={counts['created']}, skipped={counts['skipped']}, failed={counts['failed']}")
    
    print("\n" + "=" * 60)
    print("PATCH SUMMARY")
    print("=" * 60)
    print(f"  Patched (existing):              {totals['patched']:>10}")
    print(f"  Created (new):                   {totals['created']:>10}")
    print(f"  Skipped (no metadata):           {totals['skipped']:>10}")
    print(f"  Failed:                          {totals['failed']:>10}")
    print(f"  Total processed:                 {totals['patched'] + totals['created']:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
Generate train_metadata.json for PASTIS-HD.

This script:
- Downloads metadata.geojson from HuggingFace Hub (IGNF/PASTIS-HD)
- Saves it inside output_dir
- Extracts per-patch bbox + CRS
- Computes global min/max date across S1A, S1D, and S2
- Writes train_metadata.json into the SAME directory
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from shapely.geometry import shape, MultiPolygon
from huggingface_hub import hf_hub_download



# Helpers
def parse_dates(date_dict):
    """Convert {'0': 20181004, ...} → list[datetime]."""
    if not date_dict:
        return []
    return [
        datetime.strptime(str(v), "%Y%m%d").replace(tzinfo=timezone.utc)
        for _, v in sorted(date_dict.items(), key=lambda x: int(x[0]))
    ]


def get_bbox_from_geometry(geometry):
    """Return bbox (minx, miny, maxx, maxy)."""
    geom = shape(geometry)
    if not isinstance(geom, MultiPolygon):
        geom = MultiPolygon([geom])
    return list(geom.bounds)



# MAIN LOGIC
def generate_train_metadata(output_dir):
    """
    output_dir = directory where patch labels + metadata live
    e.g. "/data/pastis_hd/" → train_metadata.json will be saved here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading metadata.geojson into: {output_dir}")

    metadata_path = hf_hub_download(
        repo_id="IGNF/PASTIS-HD",
        repo_type="dataset",
        filename="metadata.geojson",
        local_dir=str(output_dir),
    )

    # Load metadata.geojson
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    # Process each patch
    for feat in data["features"]:
        patch_id = feat["id"]
        props = feat["properties"]
        geometry = feat["geometry"]

        bbox = get_bbox_from_geometry(geometry)
        crs = "EPSG:2154"

        # Collect all dates from S1A, S1D, S2
        s1a = parse_dates(props.get("dates-S1A", {}))
        s1d = parse_dates(props.get("dates-S1D", {}))
        s2 = parse_dates(props.get("dates-S2", {}))

        all_dates = sorted(s1a + s1d + s2)

        if all_dates:
            date_min = all_dates[0].isoformat()
            date_max = all_dates[-1].isoformat()
        else:
            date_min = None
            date_max = None

        # Final JSON entry
        results.append({
            "patch_id": patch_id,
            "bbox": bbox,
            "crs": crs,
            "date_min": date_min,
            "date_max": date_max
        })

    # Write train_metadata.json
    out_path = output_dir / "train_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} entries → {out_path}")



# RUN
if __name__ == "__main__":
    generate_train_metadata(output_dir="pastis_hd_target/metadata")

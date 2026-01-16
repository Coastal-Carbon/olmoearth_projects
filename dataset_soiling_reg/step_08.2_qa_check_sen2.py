#!/usr/bin/env python3
"""
Check which windows have materialized Sentinel-2 imagery.
"""

from pathlib import Path
from collections import defaultdict

WINDOWS_DIR = Path("dataset_soiling_reg/windows")
GROUPS = ["default", "predict"]


def check_group(group_dir: Path) -> dict:
    """Check windows in a group for Sentinel-2 data."""
    counts = {
        "total": 0,
        "has_sentinel2": 0,
        "has_label": 0,
        "has_both": 0,
        "missing_sentinel2": 0,
    }
    
    if not group_dir.exists():
        return counts
    
    for window_dir in group_dir.iterdir():
        if not window_dir.is_dir():
            continue
        
        counts["total"] += 1
        
        # Check for Sentinel-2 TIF files
        sentinel2_dir = window_dir / "layers" / "sentinel2_l2a"
        has_sentinel2 = False
        if sentinel2_dir.exists():
            tifs = list(sentinel2_dir.glob("**/*.tif"))
            has_sentinel2 = len(tifs) > 0
        
        # Check for label
        label_file = window_dir / "layers" / "label" / "data.geojson"
        has_label = label_file.exists()
        
        if has_sentinel2:
            counts["has_sentinel2"] += 1
        else:
            counts["missing_sentinel2"] += 1
        
        if has_label:
            counts["has_label"] += 1
        
        if has_sentinel2 and has_label:
            counts["has_both"] += 1
    
    return counts


def main():
    print("=" * 60)
    print("SENTINEL-2 IMAGERY CHECK")
    print("=" * 60)
    
    totals = defaultdict(int)
    
    for group in GROUPS:
        group_dir = WINDOWS_DIR / group
        counts = check_group(group_dir)
        
        print(f"\n{group}:")
        print(f"  Total windows:      {counts['total']}")
        print(f"  Has Sentinel-2:     {counts['has_sentinel2']}")
        print(f"  Has label:          {counts['has_label']}")
        print(f"  Has both:           {counts['has_both']}")
        print(f"  Missing Sentinel-2: {counts['missing_sentinel2']}")
        
        for k, v in counts.items():
            totals[k] += v
    
    print("\n" + "=" * 60)
    print("TOTALS")
    print("=" * 60)
    print(f"  Total windows:      {totals['total']}")
    print(f"  Has Sentinel-2:     {totals['has_sentinel2']}")
    print(f"  Has label:          {totals['has_label']}")
    print(f"  Has both:           {totals['has_both']}")
    print(f"  Missing Sentinel-2: {totals['missing_sentinel2']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Compare label GeoJSON files between source and materialized windows.
Verifies that soiling_normalized values match correctly.
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
SOURCE_DIR = Path("dataset_soiling_reg/data/labels")
WINDOWS_DIR = Path("dataset_soiling_reg/windows")
GROUPS = ["default", "predict"]

def load_source_labels(source_dir: Path) -> dict[str, dict]:
    """Load all source label files."""
    labels = {}
    for f in source_dir.glob("*.geojson"):
        name = f.stem  # filename without extension
        with open(f) as fp:
            data = json.load(fp)
        
        # Extract the soiling value from first feature
        if data.get("features"):
            props = data["features"][0].get("properties", {})
            labels[name] = {
                "soiling_normalized": props.get("soiling_normalized"),
                "soiling_raw": props.get("soiling_raw"),
                "path": str(f)
            }
    return labels

def load_window_labels(windows_dir: Path, groups: list[str]) -> dict[str, dict]:
    """Load all materialized label files from windows across all groups."""
    labels = {}
    for group in groups:
        group_dir = windows_dir / group
        if not group_dir.exists():
            continue
            
        for window_dir in group_dir.iterdir():
            if not window_dir.is_dir():
                continue
            
            label_file = window_dir / "layers" / "label" / "data.geojson"
            if not label_file.exists():
                continue
            
            # Remove .geojson suffix from window name
            window_name = window_dir.name
            if window_name.endswith(".geojson"):
                window_name = window_name[:-8]  # Remove ".geojson"
            
            with open(label_file) as fp:
                data = json.load(fp)
            
            if data.get("features"):
                props = data["features"][0].get("properties", {})
                labels[window_name] = {
                    "soiling_normalized": props.get("soiling_normalized"),
                    "soiling_raw": props.get("soiling_raw"),
                    "path": str(label_file),
                    "group": group
                }
    return labels

def compare_labels(source: dict, windows: dict):
    """Compare source and window labels, print summary."""
    
    print("=" * 60)
    print("LABEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Source labels:     {len(source)}")
    print(f"Window labels:     {len(windows)}")
    print()
    
    # Count by group
    group_counts = defaultdict(int)
    for name, data in windows.items():
        group_counts[data["group"]] += 1
    for group, count in group_counts.items():
        print(f"  {group}: {count}")
    print()
    
    # Check matches
    matched = 0
    mismatched = []
    missing_in_windows = []
    extra_in_windows = []
    
    for name, src_data in source.items():
        if name in windows:
            win_data = windows[name]
            src_val = src_data["soiling_normalized"]
            win_val = win_data["soiling_normalized"]
            
            if src_val is not None and win_val is not None:
                if abs(src_val - win_val) < 1e-6:
                    matched += 1
                else:
                    mismatched.append({
                        "name": name,
                        "source": src_val,
                        "window": win_val,
                        "diff": abs(src_val - win_val),
                        "group": win_data["group"]
                    })
            else:
                mismatched.append({
                    "name": name,
                    "source": src_val,
                    "window": win_val,
                    "diff": None,
                    "group": win_data["group"]
                })
        else:
            missing_in_windows.append(name)
    
    for name in windows:
        if name not in source:
            extra_in_windows.append(name)
    
    print(f"Matched:           {matched}")
    print(f"Mismatched:        {len(mismatched)}")
    print(f"Missing in windows:{len(missing_in_windows)}")
    print(f"Extra in windows:  {len(extra_in_windows)}")
    print()
    
    if mismatched:
        print("-" * 60)
        print("MISMATCHED VALUES:")
        print("-" * 60)
        for m in mismatched[:20]:  # Show first 20
            print(f"  {m['name']} ({m['group']})")
            print(f"    Source: {m['source']}")
            print(f"    Window: {m['window']}")
            print(f"    Diff:   {m['diff']}")
        if len(mismatched) > 20:
            print(f"  ... and {len(mismatched) - 20} more")
        print()
    
    if missing_in_windows:
        print("-" * 60)
        print("MISSING IN WINDOWS (source exists, no window label):")
        print("-" * 60)
        for name in missing_in_windows[:20]:
            print(f"  {name}")
        if len(missing_in_windows) > 20:
            print(f"  ... and {len(missing_in_windows) - 20} more")
        print()
    
    if extra_in_windows:
        print("-" * 60)
        print("EXTRA IN WINDOWS (window exists, no source):")
        print("-" * 60)
        for name in extra_in_windows[:20]:
            print(f"  {name}")
        if len(extra_in_windows) > 20:
            print(f"  ... and {len(extra_in_windows) - 20} more")
        print()
    
    # Final verdict
    print("=" * 60)
    if matched == len(source) == len(windows) and not mismatched:
        print("✓ ALL LABELS MATCH CORRECTLY")
    else:
        print("✗ ISSUES FOUND - SEE ABOVE")
    print("=" * 60)

def main():
    print("Loading source labels...")
    source = load_source_labels(SOURCE_DIR)
    
    print("Loading window labels...")
    windows = load_window_labels(WINDOWS_DIR, GROUPS)
    
    compare_labels(source, windows)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Validate label GeoJSON creation logic by comparing against rslearn-generated files.

This script:
1. Reads existing materialized labels (created by rslearn ingest)
2. Reads window metadata to understand bounds/CRS
3. Simulates what we would create
4. Compares structure and format (not values, since those are wrong)

No files are written - this is read-only validation.
"""

import json
from pathlib import Path
from dataclasses import dataclass

# === CONFIG ===
WINDOWS_DIR = Path("dataset_soiling_reg/windows")
METADATA_FILE = Path("dataset_soiling_reg/data/metadata/train_metadata.json")
GROUPS = ["default", "predict"]
MAX_SAMPLES = 10  # Number of samples to inspect in detail


@dataclass
class WindowInfo:
    """Info extracted from a window's metadata.json"""
    name: str
    group: str
    bounds: list[float]  # [min_x, min_y, max_x, max_y]
    crs: str
    resolution: float
    

@dataclass 
class LabelInfo:
    """Info extracted from a materialized label file"""
    crs: str
    x_resolution: float
    y_resolution: float
    geometry_type: str
    coordinates: list[float]
    properties: dict
    raw_data: dict  # Full JSON for inspection


def load_metadata(filepath: Path) -> dict[str, dict]:
    """Load metadata and create lookup by window name."""
    print(f"Loading metadata from {filepath}")
    
    with open(filepath) as f:
        data = json.load(f)
    
    lookup = {}
    for entry in data["entries"]:
        window_name = entry["filename"].replace(".geojson", "")
        lookup[window_name] = entry
    
    print(f"Loaded {len(lookup)} entries")
    return lookup


def load_window_metadata(window_dir: Path) -> WindowInfo | None:
    """Load window metadata.json to get bounds and CRS."""
    meta_file = window_dir / "metadata.json"
    if not meta_file.exists():
        return None
    
    with open(meta_file) as f:
        meta = json.load(f)
    
    return WindowInfo(
        name=window_dir.name,
        group=window_dir.parent.name,
        bounds=meta.get("bounds", []),
        crs=meta.get("projection", {}).get("crs", ""),
        resolution=meta.get("projection", {}).get("x_resolution", 10.0)
    )


def load_label_file(label_path: Path) -> LabelInfo | None:
    """Load and parse a materialized label GeoJSON."""
    if not label_path.exists():
        return None
    
    with open(label_path) as f:
        data = json.load(f)
    
    props = data.get("properties", {})
    features = data.get("features", [])
    
    if not features:
        return None
    
    feat = features[0]
    geom = feat.get("geometry", {})
    
    return LabelInfo(
        crs=props.get("crs", ""),
        x_resolution=props.get("x_resolution", 0),
        y_resolution=props.get("y_resolution", 0),
        geometry_type=geom.get("type", ""),
        coordinates=geom.get("coordinates", []),
        properties=feat.get("properties", {}),
        raw_data=data
    )


def compute_window_centroid(bounds: list[float]) -> list[float]:
    """Compute centroid from window bounds [min_x, min_y, max_x, max_y]."""
    if len(bounds) != 4:
        return [0, 0]
    min_x, min_y, max_x, max_y = bounds
    return [(min_x + max_x) / 2, (min_y + max_y) / 2]


def analyze_existing_labels():
    """Analyze existing materialized labels to understand the format."""
    
    print("\n" + "=" * 70)
    print("ANALYZING EXISTING MATERIALIZED LABELS")
    print("=" * 70)
    
    samples_analyzed = 0
    format_summary = {
        "crs_formats": set(),
        "geometry_types": set(),
        "resolution_pairs": set(),
        "property_keys": set(),
    }
    
    coordinate_analysis = []
    
    for group in GROUPS:
        group_dir = WINDOWS_DIR / group
        if not group_dir.exists():
            continue
        
        for window_dir in group_dir.iterdir():
            if not window_dir.is_dir():
                continue
            
            label_path = window_dir / "layers" / "label" / "data.geojson"
            if not label_path.exists():
                continue
            
            # Load both window metadata and label
            window_info = load_window_metadata(window_dir)
            label_info = load_label_file(label_path)
            
            if not window_info or not label_info:
                continue
            
            # Collect format info
            format_summary["crs_formats"].add(label_info.crs)
            format_summary["geometry_types"].add(label_info.geometry_type)
            format_summary["resolution_pairs"].add(
                (label_info.x_resolution, label_info.y_resolution)
            )
            for key in label_info.properties.keys():
                format_summary["property_keys"].add(key)
            
            # Analyze coordinate relationship to window bounds
            if window_info.bounds and label_info.coordinates:
                centroid = compute_window_centroid(window_info.bounds)
                coord_analysis = {
                    "window_name": window_info.name,
                    "window_bounds": window_info.bounds,
                    "window_crs": window_info.crs,
                    "computed_centroid": centroid,
                    "label_coordinates": label_info.coordinates,
                    "label_crs": label_info.crs,
                    "coord_diff": [
                        label_info.coordinates[0] - centroid[0] if len(label_info.coordinates) > 0 else None,
                        label_info.coordinates[1] - centroid[1] if len(label_info.coordinates) > 1 else None,
                    ]
                }
                coordinate_analysis.append(coord_analysis)
            
            samples_analyzed += 1
            
            # Print detailed info for first few samples
            if samples_analyzed <= MAX_SAMPLES:
                print(f"\n--- Sample {samples_analyzed}: {window_info.name} ---")
                print(f"Window bounds: {window_info.bounds}")
                print(f"Window CRS: {window_info.crs}")
                print(f"Computed centroid: {compute_window_centroid(window_info.bounds)}")
                print(f"Label CRS: {label_info.crs}")
                print(f"Label resolution: ({label_info.x_resolution}, {label_info.y_resolution})")
                print(f"Label geometry type: {label_info.geometry_type}")
                print(f"Label coordinates: {label_info.coordinates}")
                print(f"Label properties: {list(label_info.properties.keys())}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FORMAT SUMMARY")
    print("=" * 70)
    print(f"Total samples analyzed: {samples_analyzed}")
    print(f"\nCRS formats found: {format_summary['crs_formats']}")
    print(f"Geometry types: {format_summary['geometry_types']}")
    print(f"Resolution pairs (x, y): {format_summary['resolution_pairs']}")
    print(f"Property keys: {format_summary['property_keys']}")
    
    # Analyze coordinate patterns
    if coordinate_analysis:
        print("\n" + "=" * 70)
        print("COORDINATE ANALYSIS")
        print("=" * 70)
        print("Checking if label coordinates match window centroids...")
        
        matches = 0
        mismatches = 0
        
        for analysis in coordinate_analysis[:20]:  # Check first 20
            diff_x = analysis["coord_diff"][0]
            diff_y = analysis["coord_diff"][1]
            
            if diff_x is not None and diff_y is not None:
                # Check if coordinates are close to centroid
                if abs(diff_x) < 1 and abs(diff_y) < 1:
                    matches += 1
                else:
                    mismatches += 1
                    if mismatches <= 5:  # Show first 5 mismatches
                        print(f"\nMismatch: {analysis['window_name']}")
                        print(f"  Window bounds: {analysis['window_bounds']}")
                        print(f"  Computed centroid: {analysis['computed_centroid']}")
                        print(f"  Label coordinates: {analysis['label_coordinates']}")
                        print(f"  Difference: ({diff_x:.2f}, {diff_y:.2f})")
        
        print(f"\nCoordinate check (first 20): {matches} near centroid, {mismatches} not")
        
        if mismatches > 0:
            print("\n*** Labels do NOT use window centroid - they use original source geometry ***")
            print("This means we need to use the source label coordinates, not compute new ones.")


def simulate_label_creation(metadata_lookup: dict):
    """Simulate what we would create and compare to existing."""
    
    print("\n" + "=" * 70)
    print("SIMULATING LABEL CREATION")
    print("=" * 70)
    
    simulated = 0
    format_matches = 0
    format_mismatches = []
    
    for group in GROUPS:
        group_dir = WINDOWS_DIR / group
        if not group_dir.exists():
            continue
        
        for window_dir in group_dir.iterdir():
            if not window_dir.is_dir():
                continue
            
            window_name = window_dir.name
            if window_name.endswith(".geojson"):
                window_name = window_name[:-8]
            
            label_path = window_dir / "layers" / "label" / "data.geojson"
            if not label_path.exists():
                continue  # Skip missing - we'll create those later
            
            if window_name not in metadata_lookup:
                continue
            
            # Load existing label
            existing = load_label_file(label_path)
            window_info = load_window_metadata(window_dir)
            
            if not existing or not window_info:
                continue
            
            # Simulate what we would create
            metadata_entry = metadata_lookup[window_name]
            
            simulated_data = {
                "type": "FeatureCollection",
                "properties": {
                    "crs": window_info.crs,
                    "x_resolution": 10.0,
                    "y_resolution": -10.0,
                },
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "soiling_normalized": metadata_entry["soiling_normalized"],
                        "soiling_raw": metadata_entry["soiling_raw"],
                        "soiling_clipped": metadata_entry["soiling_clipped"],
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": compute_window_centroid(window_info.bounds)
                    }
                }]
            }
            
            # Compare structure (not values)
            structure_ok = True
            issues = []
            
            # Check CRS format
            if existing.crs != window_info.crs:
                # CRS might be in different format
                issues.append(f"CRS: existing={existing.crs}, window={window_info.crs}")
            
            # Check resolution
            if existing.x_resolution != 10.0:
                issues.append(f"x_resolution: {existing.x_resolution}")
            if existing.y_resolution != -10.0:
                issues.append(f"y_resolution: {existing.y_resolution}")
            
            # Check geometry type
            if existing.geometry_type != "Point":
                issues.append(f"geometry_type: {existing.geometry_type}")
            
            # Check property keys
            expected_keys = {"soiling_normalized", "soiling_raw", "soiling_clipped"}
            actual_keys = set(existing.properties.keys())
            if expected_keys != actual_keys:
                issues.append(f"property_keys: expected={expected_keys}, actual={actual_keys}")
            
            if issues:
                structure_ok = False
                format_mismatches.append({
                    "window": window_name,
                    "issues": issues
                })
            else:
                format_matches += 1
            
            simulated += 1
    
    print(f"\nSimulated {simulated} labels")
    print(f"Format matches: {format_matches}")
    print(f"Format mismatches: {len(format_mismatches)}")
    
    if format_mismatches:
        print("\nFirst 10 mismatches:")
        for m in format_mismatches[:10]:
            print(f"  {m['window']}: {m['issues']}")


def main():
    metadata_lookup = load_metadata(METADATA_FILE)
    analyze_existing_labels()
    simulate_label_creation(metadata_lookup)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nNext step: If format looks correct, run the actual patch creation script.")


if __name__ == "__main__":
    main()
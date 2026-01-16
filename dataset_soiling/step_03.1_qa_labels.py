"""
QA Check for Generated Label TIFs

Validates label TIFs against train_metadata.json:
- Checks TIF dimensions, CRS, bbox, resolution
- Verifies pixel values match metadata
- Reports any mismatches or missing files
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
LABELS_DIR = "dataset_soiling/data/labels"
METADATA_FILE = "dataset_soiling/data/metadata/train_metadata.json"
N_SAMPLES = None  # None for all, or int to spot-check subset


def load_metadata(filepath: str) -> dict:
    """Load train metadata JSON."""
    logger.info(f"Loading metadata from {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def check_tif(filepath: Path, entry: dict, config: dict) -> dict:
    """Validate a single TIF against its metadata entry."""
    errors = []
    
    if not filepath.exists():
        return {"filename": entry["filename"], "status": "MISSING", "errors": ["File not found"]}
    
    try:
        with rasterio.open(filepath) as src:
            # Check dimensions
            expected_size = config["chip_size"]
            if src.height != expected_size or src.width != expected_size:
                errors.append(f"Dimensions: expected {expected_size}x{expected_size}, got {src.height}x{src.width}")
            
            # Check CRS
            expected_epsg = entry["utm_epsg"]
            actual_epsg = src.crs.to_epsg() if src.crs else None
            if actual_epsg != expected_epsg:
                errors.append(f"CRS: expected EPSG:{expected_epsg}, got EPSG:{actual_epsg}")
            
            # Check resolution
            expected_res = config["resolution"]
            actual_res_x = abs(src.transform[0])
            actual_res_y = abs(src.transform[4])
            if not (np.isclose(actual_res_x, expected_res, atol=0.01) and 
                    np.isclose(actual_res_y, expected_res, atol=0.01)):
                errors.append(f"Resolution: expected {expected_res}m, got {actual_res_x:.2f}x{actual_res_y:.2f}m")
            
            # Check bbox
            expected_bbox = entry["bbox"]
            actual_bbox = list(src.bounds)
            bbox_match = all(np.isclose(a, e, atol=0.1) for a, e in zip(actual_bbox, expected_bbox))
            if not bbox_match:
                errors.append(f"Bbox mismatch: expected {expected_bbox}, got {actual_bbox}")
            
            # Check pixel value
            data = src.read(1)
            expected_value = entry["soiling_normalized"]
            unique_values = np.unique(data[data != -1])  # Exclude nodata
            
            if len(unique_values) != 1:
                errors.append(f"Multiple pixel values found: {unique_values}")
            elif not np.isclose(unique_values[0], expected_value, atol=1e-6):
                errors.append(f"Pixel value: expected {expected_value:.6f}, got {unique_values[0]:.6f}")
            
            # Check nodata
            if src.nodata != -1:
                errors.append(f"Nodata: expected -1, got {src.nodata}")
    
    except Exception as e:
        return {"filename": entry["filename"], "status": "ERROR", "errors": [str(e)]}
    
    status = "OK" if not errors else "FAIL"
    return {"filename": entry["filename"], "status": status, "errors": errors}


def print_summary(results: list, config: dict, norm_params: dict):
    """Print QA summary."""
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    missing_count = sum(1 for r in results if r["status"] == "MISSING")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    print("\n" + "=" * 70)
    print("LABEL TIF QA SUMMARY")
    print("=" * 70)
    print(f"  Total checked:                   {len(results):>10,}")
    print(f"  OK:                              {ok_count:>10,}")
    print(f"  FAIL:                            {fail_count:>10,}")
    print(f"  MISSING:                         {missing_count:>10,}")
    print(f"  ERROR:                           {error_count:>10,}")
    print()
    print("  EXPECTED CONFIG:")
    print(f"    Chip size:                     {config['chip_size']} x {config['chip_size']} pixels")
    print(f"    Resolution:                    {config['resolution']}m")
    print(f"    Cloud cover threshold:         <{config['cloud_cover_threshold']}%")
    print()
    print("  NORMALIZATION PARAMS:")
    print(f"    Clip range:                    [{norm_params['clip_lower']:.6f}, {norm_params['clip_upper']:.6f}]")
    print(f"    Norm range:                    [{norm_params['norm_min']:.6f}, {norm_params['norm_max']:.6f}]")
    print("=" * 70)
    
    # Print failures if any
    failures = [r for r in results if r["status"] in ("FAIL", "MISSING", "ERROR")]
    if failures:
        print("\nFAILURES:")
        print("-" * 70)
        for r in failures[:20]:  # Limit to first 20
            print(f"\n  {r['filename']} [{r['status']}]")
            for err in r["errors"]:
                print(f"    - {err}")
        if len(failures) > 20:
            print(f"\n  ... and {len(failures) - 20} more failures")
    else:
        print("\n[OK] All TIFs passed validation")


def print_sample_tif_info(labels_dir: Path, entries: list):
    """Print detailed info for a sample TIF."""
    if not entries:
        return
    
    sample = entries[0]
    filepath = labels_dir / sample["filename"]
    
    print("\n" + "=" * 70)
    print("SAMPLE TIF INSPECTION")
    print("=" * 70)
    print(f"  Filename: {sample['filename']}")
    
    if filepath.exists():
        with rasterio.open(filepath) as src:
            print(f"\n  FILE PROPERTIES:")
            print(f"    Dimensions:                    {src.height} x {src.width} pixels")
            print(f"    CRS:                           {src.crs}")
            print(f"    Resolution:                    {abs(src.transform[0]):.2f} x {abs(src.transform[4]):.2f}m")
            print(f"    Bounds:                        {list(src.bounds)}")
            print(f"    Nodata:                        {src.nodata}")
            print(f"    Dtype:                         {src.dtypes[0]}")
            
            data = src.read(1)
            print(f"\n  PIXEL VALUES:")
            print(f"    Min:                           {data.min():.6f}")
            print(f"    Max:                           {data.max():.6f}")
            print(f"    Mean:                          {data.mean():.6f}")
            print(f"    Unique values:                 {len(np.unique(data))}")
        
        print(f"\n  METADATA ENTRY:")
        print(f"    Plant:                         {sample['id_plant']}")
        print(f"    Device:                        {sample['id_device']}")
        print(f"    Date:                          {sample['date']}")
        print(f"    Cloud cover:                   {sample['cloud_cover']:.4f}%")
        print(f"    UTM EPSG:                      {sample['utm_epsg']}")
        print(f"    Soiling raw:                   {sample['soiling_raw']:.6f}")
        print(f"    Soiling clipped:               {sample['soiling_clipped']:.6f}")
        print(f"    Soiling normalized:            {sample['soiling_normalized']:.6f}")
    
    print("=" * 70)


def main():
    logger.info("Starting label TIF QA check")
    
    labels_dir = Path(LABELS_DIR)
    metadata = load_metadata(METADATA_FILE)
    
    config = metadata["config"]
    norm_params = metadata["normalization"]
    entries = metadata["entries"]
    
    logger.info(f"Config: {config['chip_size']}x{config['chip_size']} @ {config['resolution']}m")
    logger.info(f"Entries to check: {len(entries)}")
    
    # Limit samples if set
    if N_SAMPLES:
        entries = entries[:N_SAMPLES]
        logger.info(f"Limited to {N_SAMPLES} samples")
    
    # Check each TIF
    results = []
    for idx, entry in enumerate(entries):
        if idx % 500 == 0 and idx > 0:
            logger.info(f"Checked {idx}/{len(entries)} TIFs")
        
        filepath = labels_dir / entry["filename"]
        result = check_tif(filepath, entry, config)
        results.append(result)
    
    logger.info(f"Checked {len(results)} TIFs")
    
    # Print results
    print_sample_tif_info(labels_dir, entries)
    print_summary(results, config, norm_params)
    
    logger.info("QA check complete")


if __name__ == "__main__":
    main()
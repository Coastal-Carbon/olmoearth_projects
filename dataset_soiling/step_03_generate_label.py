"""
Soiling Label TIF Generator

Generates georeferenced label TIFs for soiling prediction model training.
Each TIF is a single-band raster filled with the normalized soiling index value,
centered on the device geometry in the native UTM projection from Sentinel-2.

Output:
- labels/*.tif — 128x128 (configurable) label chips
- metadata/normalization.json — global normalization params for denormalization
- metadata/train_metadata.json — per-file metadata for dataloader
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import shape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FILE = "dataset_soiling/soiling_with_stac.parquet"
OUTPUT_DIR = "dataset_soiling/data"
LABELS_SUBDIR = "labels"
METADATA_SUBDIR = "metadata"

CHIP_SIZE = 32                 # pixels
RESOLUTION = 10                 # meters per pixel
CLOUD_COVER_THRESHOLD = 1       # filter cloud_cover < this %
CLIP_LOWER_PERCENTILE = 0.05    # 5th percentile
CLIP_UPPER_PERCENTILE = 0.95    # 95th percentile
N_ROWS = None                   # None for all, or int for testing


def setup_directories(base_dir: str) -> tuple[Path, Path]:
    """Create output directories."""
    base = Path(base_dir)
    labels_dir = base / LABELS_SUBDIR
    metadata_dir = base / METADATA_SUBDIR
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directories: {labels_dir}, {metadata_dir}")
    return labels_dir, metadata_dir


def load_and_filter_data(filepath: str, cloud_threshold: float, n_rows: int = None) -> pd.DataFrame:
    """Load data and filter to rows with STAC match and low cloud cover."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)
    
    initial_count = len(df)
    
    # Filter to rows with STAC match
    df = df[df["item_id"].notna()]
    logger.info(f"After STAC filter: {len(df):,} rows (from {initial_count:,})")
    
    # Filter by cloud cover
    df = df[df["cloud_cover"] < cloud_threshold]
    logger.info(f"After cloud filter (<{cloud_threshold}%): {len(df):,} rows")
    
    # Apply row limit if set
    if n_rows:
        df = df.head(n_rows)
        logger.info(f"Limited to {n_rows} rows for testing")
    
    return df.reset_index(drop=True)


def compute_normalization_params(soiling_values: pd.Series) -> dict:
    """Compute clipping bounds and normalization parameters."""
    clip_lower = soiling_values.quantile(CLIP_LOWER_PERCENTILE)
    clip_upper = soiling_values.quantile(CLIP_UPPER_PERCENTILE)
    
    clipped = soiling_values.clip(lower=clip_lower, upper=clip_upper)
    norm_min = clipped.min()
    norm_max = clipped.max()
    
    params = {
        "clip_lower_percentile": CLIP_LOWER_PERCENTILE,
        "clip_upper_percentile": CLIP_UPPER_PERCENTILE,
        "clip_lower": float(clip_lower),
        "clip_upper": float(clip_upper),
        "norm_min": float(norm_min),
        "norm_max": float(norm_max),
        "created_at": datetime.now().isoformat(),
    }
    
    logger.info(
        f"Normalization params: clip=[{clip_lower:.6f}, {clip_upper:.6f}], "
        f"norm_range=[{norm_min:.6f}, {norm_max:.6f}]"
    )
    
    return params


def normalize_value(value: float, params: dict) -> tuple[float, float]:
    """Apply clipping and min-max normalization to a single value."""
    clipped = np.clip(value, params["clip_lower"], params["clip_upper"])
    
    norm_range = params["norm_max"] - params["norm_min"]
    if norm_range == 0:
        normalized = 0.0
    else:
        normalized = (clipped - params["norm_min"]) / norm_range
    
    return float(clipped), float(normalized)


def get_geometry_centroid(portion_json: str) -> tuple[float, float]:
    """Parse portion geometry and return centroid in EPSG:4326."""
    geom = shape(json.loads(portion_json))
    centroid = geom.centroid
    return centroid.x, centroid.y  # lon, lat


def reproject_point(lon: float, lat: float, target_epsg: int) -> tuple[float, float]:
    """Reproject point from EPSG:4326 to target UTM CRS."""
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def compute_utm_bbox(center_x: float, center_y: float, chip_size: int, resolution: float) -> tuple:
    """Compute bbox in UTM meters centered on point, snapped to pixel grid."""
    half_size = (chip_size * resolution) / 2
    
    # Snap center to resolution grid
    center_x_snapped = round(center_x / resolution) * resolution
    center_y_snapped = round(center_y / resolution) * resolution
    
    minx = center_x_snapped - half_size
    maxx = center_x_snapped + half_size
    miny = center_y_snapped - half_size
    maxy = center_y_snapped + half_size
    
    return minx, miny, maxx, maxy


def generate_filename(plant: str, device: str, date: pd.Timestamp) -> str:
    """Generate clean filename for label TIF."""
    # Clean plant and device names (remove special chars, spaces)
    plant_clean = plant.replace(" ", "_").replace("/", "-")
    device_clean = device.replace(" ", "_").replace("/", "-")
    date_str = date.strftime("%Y-%m-%d")
    
    return f"{plant_clean}_{device_clean}_{date_str}_label.tif"


def save_label_tif(
    filepath: Path,
    value: float,
    bbox: tuple,
    epsg: int,
    chip_size: int
) -> None:
    """Save single-band label TIF filled with normalized soiling value."""
    minx, miny, maxx, maxy = bbox
    transform = from_bounds(minx, miny, maxx, maxy, chip_size, chip_size)
    
    # Create chip filled with the normalized value
    data = np.full((chip_size, chip_size), value, dtype=np.float32)
    
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=chip_size,
        width=chip_size,
        count=1,
        dtype=np.float32,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=-1,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "soiling_index_normalized")


def process_row(
    row: pd.Series,
    norm_params: dict,
    labels_dir: Path,
    chip_size: int,
    resolution: float
) -> dict | None:
    """Process a single row and generate label TIF."""
    try:
        # Extract values
        plant = row["ID_PLANT"]
        device = row["ID_DEVICE"]
        date = row["date"]
        soiling_raw = row["SOILING_INDEX_DAILY"]
        item_id = row["item_id"]
        cloud_cover = row["cloud_cover"]
        utm_epsg = int(row["utm_epsg"])
        portion = row["portion"]
        
        # Normalize soiling value
        soiling_clipped, soiling_normalized = normalize_value(soiling_raw, norm_params)
        
        # Get centroid and reproject to UTM
        lon, lat = get_geometry_centroid(portion)
        utm_x, utm_y = reproject_point(lon, lat, utm_epsg)
        
        # Compute UTM bbox
        bbox = compute_utm_bbox(utm_x, utm_y, chip_size, resolution)
        
        # Generate filename and save TIF
        filename = generate_filename(plant, device, date)
        filepath = labels_dir / filename
        
        save_label_tif(filepath, soiling_normalized, bbox, utm_epsg, chip_size)
        
        # Return metadata entry
        return {
            "filename": filename,
            "id_plant": plant,
            "id_device": device,
            "date": date.strftime("%Y-%m-%d"),
            "item_id": item_id,
            "cloud_cover": float(cloud_cover),
            "utm_epsg": utm_epsg,
            "bbox": list(bbox),
            "centroid_lon": lon,
            "centroid_lat": lat,
            "centroid_utm_x": utm_x,
            "centroid_utm_y": utm_y,
            "soiling_raw": float(soiling_raw),
            "soiling_clipped": soiling_clipped,
            "soiling_normalized": soiling_normalized,
        }
        
    except Exception as e:
        logger.error(f"Failed to process row {row.name}: {e}")
        return None


def save_metadata(
    metadata_dir: Path,
    norm_params: dict,
    entries: list[dict],
    config: dict
) -> None:
    """Save normalization params and per-file metadata."""
    # Save normalization params
    norm_path = metadata_dir / "normalization.json"
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    logger.info(f"Saved normalization params to {norm_path}")
    
    # Save train metadata
    train_meta = {
        "config": config,
        "normalization": norm_params,
        "entries": entries,
    }
    
    meta_path = metadata_dir / "train_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(train_meta, f, indent=2)
    logger.info(f"Saved train metadata ({len(entries)} entries) to {meta_path}")


def print_summary(entries: list[dict], labels_dir: Path, failed_count: int):
    """Print generation summary."""
    # Calculate directory size
    total_size = sum(f.stat().st_size for f in labels_dir.glob("*.tif"))
    size_mb = total_size / (1024 * 1024)
    
    # Stats on normalized values
    norm_values = [e["soiling_normalized"] for e in entries]
    
    print("\n" + "=" * 60)
    print("LABEL GENERATION SUMMARY")
    print("=" * 60)
    print(f"  TIFs generated:                  {len(entries):>10,}")
    print(f"  Failed rows:                     {failed_count:>10,}")
    print(f"  Output directory:                {labels_dir}")
    print(f"  Total size:                      {size_mb:>10.2f} MB")
    print()
    print(f"  Normalized value range:          [{min(norm_values):.4f}, {max(norm_values):.4f}]")
    print(f"  Normalized value mean:           {np.mean(norm_values):>10.4f}")
    print()
    print(f"  Unique plants:                   {len(set(e['id_plant'] for e in entries)):>10}")
    print(f"  Unique devices:                  {len(set(e['id_device'] for e in entries)):>10}")
    print(f"  Unique UTM zones:                {len(set(e['utm_epsg'] for e in entries)):>10}")
    print("=" * 60)


def main():
    logger.info("Starting soiling label TIF generation")
    
    # Setup
    labels_dir, metadata_dir = setup_directories(OUTPUT_DIR)
    
    # Load and filter data
    df = load_and_filter_data(INPUT_FILE, CLOUD_COVER_THRESHOLD, N_ROWS)
    
    if len(df) == 0:
        logger.error("No rows to process after filtering")
        return
    
    # Compute normalization params
    norm_params = compute_normalization_params(df["SOILING_INDEX_DAILY"])
    
    # Process each row
    entries = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"Processing row {idx + 1}/{len(df)}")
        
        result = process_row(row, norm_params, labels_dir, CHIP_SIZE, RESOLUTION)
        
        if result:
            entries.append(result)
        else:
            failed_count += 1
    
    logger.info(f"Processing complete: {len(entries)} succeeded, {failed_count} failed")
    
    # Save metadata
    config = {
        "chip_size": CHIP_SIZE,
        "resolution": RESOLUTION,
        "cloud_cover_threshold": CLOUD_COVER_THRESHOLD,
        "clip_lower_percentile": CLIP_LOWER_PERCENTILE,
        "clip_upper_percentile": CLIP_UPPER_PERCENTILE,
        "input_file": INPUT_FILE,
        "created_at": datetime.now().isoformat(),
    }
    
    save_metadata(metadata_dir, norm_params, entries, config)
    
    # Print summary
    print_summary(entries, labels_dir, failed_count)
    
    logger.info("Label generation complete")


if __name__ == "__main__":
    main()
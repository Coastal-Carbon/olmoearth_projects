"""
Export ONLY the PASTIS-HD TARGET crop_type layer.
For each location:
  - Load PASTISHD_CROP_TYPE chip
  - Save 1-band GeoTIFF
  - Save bbox + CRS metadata
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely import wkb

from hum_ai.data_engine.database.registry import get_registry
from hum_ai.data_engine.database.session import get_session
from hum_ai.data_engine.datasets import dataset, dataset_exists
from hum_ai.data_engine.ingredients import ObservationType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def decode_bbox_wkb(bbox_wkb):
    """Convert WKB hex bbox to (minx, miny, maxx, maxy)."""
    if bbox_wkb is None:
        return None
    geom = wkb.loads(bytes.fromhex(bbox_wkb))
    return geom.bounds


def save_tiff(data, path, bbox, epsg, band_name="crop_type"):
    """Save a single-band GeoTIFF with bbox + CRS."""
    if data.ndim != 2:
        raise ValueError("Expected a 2D chip for target crop type")

    h, w = data.shape
    minx, miny, maxx, maxy = bbox
    transform = from_bounds(minx, miny, maxx, maxy, w, h)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs=f"EPSG:{epsg}",
        nodata=-1,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, band_name)


class PastisHDTargetConverter:
    """Convert ONLY the PASTISHD_CROP_TYPE layer to GeoTIFF."""

    def __init__(self, dataset_name: str = "pastis-hd"):
        self.dataset_name = dataset_name

    def convert(self, out_dir: str, max_locations: Optional[int] = None):
        out_root = Path(out_dir)
        target_dir = out_root / "TARGET"
        metadata_dir = out_root / "metadata"

        target_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        with get_session(read_only=True) as session:
            registry = get_registry(session)

            if not dataset_exists(self.dataset_name, registry):
                logger.error(f"Dataset not found: {self.dataset_name}")
                return {}

            logger.info("Loading crop_type TARGET chips...")
            rows = dataset(
                self.dataset_name,
                registry,
                chip_type=ObservationType.PASTISHD_CROP_TYPE
            ).take_all()

            logger.info(f"Found {len(rows)} target chips")

            # group by location
            target_data = {row["location"]: row for row in rows}

            locations = sorted(target_data.keys())
            if max_locations:
                locations = locations[:max_locations]

            chip_metadata: Dict[str, Dict] = {}
            processed = []

            for idx, location in enumerate(locations, 1):
                row = target_data[location]
                logger.info(f"Processing {location} ({idx}/{len(locations)})")

                bbox_wkb = row.get("bbox")
                bbox_hex = bbox_wkb.hex() if isinstance(bbox_wkb, bytes) else bbox_wkb
                bbox = decode_bbox_wkb(bbox_hex)

                if bbox is None:
                    logger.warning(f"No bbox for {location}, skipping")
                    continue

                epsg = int(row.get("epsg", 2154))
                chip = row["chip"]

                # output path
                out_path = target_dir / f"{location}_TARGET.tif"

                # save tiff
                save_tiff(
                    chip,
                    out_path,
                    bbox=bbox,
                    epsg=epsg,
                    band_name="crop_type"
                )

                # save metadata
                chip_metadata[location] = {
                    "centroid_lon": float(row.get("centroid_lon", 0.0)),
                    "centroid_lat": float(row.get("centroid_lat", 0.0)),
                    "epsg": epsg,
                    "bbox_wkb": bbox_hex,
                }

                processed.append(location)

            # write metadata file
            metadata_path = metadata_dir / "target_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(chip_metadata, f, indent=2)

            logger.info(f"Saved metadata for {len(processed)} locations")

            return {"processed": processed}


def main():
    out_dir = "pastis_hd_target"
    max_locations = None   # or set something like 10 for testing

    logger.info("Exporting PASTIS-HD crop_type TARGET only...")
    converter = PastisHDTargetConverter()
    results = converter.convert(out_dir, max_locations)

    print("\nDONE")
    print(f"Processed {len(results['processed'])} locations")


if __name__ == "__main__":
    main()

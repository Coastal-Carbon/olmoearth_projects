"""Tests for olmoearth_projects.projects.forest_loss_driver.extract_alerts."""

from datetime import timedelta
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.projects.forest_loss_driver.extract_alerts import (
    BASE_DATETIME,
    ExtractAlertsArgs,
    extract_alerts,
)

# A 10 m/pixel similar to ones used by GLAD rasters.
TEST_PROJECTION = Projection(CRS.from_epsg(32610), 10, -10)


def test_days_limit(tmp_path: Path) -> None:
    """Make sure that extract_alerts respects limit on days in the past."""
    # Create date raster that has one pixel with good days (+10) and one with bad (+1).
    # Days are measured from BASE_DATETIME.
    date_raster = np.zeros((3, 3), dtype=np.uint32)
    date_raster[0, 0] = 10
    date_raster[2, 2] = 1
    conf_raster = np.ones((3, 3), dtype=np.uint32)
    conf_prefix = tmp_path / "alert"
    date_prefix = tmp_path / "alertDate"
    conf_prefix.mkdir()
    date_prefix.mkdir()
    GeotiffRasterFormat().encode_raster(
        path=UPath(conf_prefix),
        projection=TEST_PROJECTION,
        bounds=(0, 0, 3, 3),
        array=conf_raster[None, :, :],
        fname="raster.tif",
    )
    GeotiffRasterFormat().encode_raster(
        path=UPath(date_prefix),
        projection=TEST_PROJECTION,
        bounds=(0, 0, 3, 3),
        array=date_raster[None, :, :],
        fname="raster.tif",
    )

    # Now extract alerts and validate result.
    out_fname = tmp_path / "result.geojson"
    extract_alerts(
        ExtractAlertsArgs(
            gcs_tiff_filenames=["raster.tif"],
            out_fname=str(out_fname),
            conf_prefix=str(conf_prefix),
            date_prefix=str(date_prefix),
            prediction_utc_time=BASE_DATETIME + timedelta(days=12),
            min_confidence=1,
            days=5,
            min_area=1,
        )
    )
    features = GeojsonVectorFormat().decode_from_file(UPath(out_fname))
    assert len(features) == 1
    feat = features[0]
    assert feat.properties["tif_fname"] == "raster.tif"
    assert tuple(feat.properties["center_pixel"]) == (0, 0)


def test_slice_days(tmp_path: Path) -> None:
    """Make sure slice_days separates components from different time slices.

    Two horizontally adjacent alert pixels with dates in different slices would merge
    into a single connected component without slicing, but should yield two separate
    events when slice_days is set.
    """
    # Pixels (col=0, row=0) and (col=1, row=0) are adjacent but dated 30 and 80 days
    # from BASE_DATETIME, which fall into different 50-day slices.
    date_raster = np.zeros((3, 3), dtype=np.uint32)
    date_raster[0, 0] = 30
    date_raster[0, 1] = 80
    conf_raster = np.ones((3, 3), dtype=np.uint32)
    conf_prefix = tmp_path / "alert"
    date_prefix = tmp_path / "alertDate"
    conf_prefix.mkdir()
    date_prefix.mkdir()
    GeotiffRasterFormat().encode_raster(
        path=UPath(conf_prefix),
        projection=TEST_PROJECTION,
        bounds=(0, 0, 3, 3),
        array=conf_raster[None, :, :],
        fname="raster.tif",
    )
    GeotiffRasterFormat().encode_raster(
        path=UPath(date_prefix),
        projection=TEST_PROJECTION,
        bounds=(0, 0, 3, 3),
        array=date_raster[None, :, :],
        fname="raster.tif",
    )

    out_fname = tmp_path / "result.geojson"
    extract_alerts(
        ExtractAlertsArgs(
            gcs_tiff_filenames=["raster.tif"],
            out_fname=str(out_fname),
            conf_prefix=str(conf_prefix),
            date_prefix=str(date_prefix),
            prediction_utc_time=BASE_DATETIME + timedelta(days=110),
            min_confidence=1,
            days=200,
            min_area=1,
            slice_days=50,
        )
    )
    features = GeojsonVectorFormat().decode_from_file(UPath(out_fname))
    # The two differently-dated adjacent pixels should be separate events.
    assert len(features) == 2
    center_pixels = {tuple(feat.properties["center_pixel"]) for feat in features}
    assert center_pixels == {(0, 0), (1, 0)}

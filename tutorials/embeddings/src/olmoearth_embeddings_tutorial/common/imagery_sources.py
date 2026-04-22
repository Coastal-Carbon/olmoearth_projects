"""Download Sentinel-2 RGB and ESA WorldCover aligned to an embedding COG.

These helpers use Microsoft Planetary Computer for STAC discovery and access.
The output rasters are reprojected and clipped to match the embedding grid
so that pixel indices align 1:1 between embeddings and imagery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.fill
import rioxarray  # noqa: F401 (registers .rio accessor)
import stackstac
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds


def _wgs84_bounds(embed_path: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS 84 for the embedding extent."""
    with rasterio.open(embed_path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return west, south, east, north


def _projected_bounds(embed_path: Path) -> tuple[float, float, float, float]:
    """Return (left, bottom, right, top) in the embed CRS."""
    with rasterio.open(embed_path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def download_s2_rgb(
    embed_path: Path,
    out_path: Path,
    datetime_range: str = "2024-01-01/2024-12-31",
    cloud_max: float = 60.0,
    fill_holes_pixels: int = 48,
) -> Path:
    """Download a cloud-median Sentinel-2 L2A RGB composite aligned to *embed_path*.

    Uses bands B04 (R), B03 (G), B02 (B) and computes the time median across
    all scenes within *datetime_range* that pass the cloud cover filter.  The
    result is reprojected to match the embedding grid exactly.

    Args:
        embed_path: Path to the embedding COG (defines extent and CRS).
        out_path: Destination for the 3-band float32 COG (reflectance 0-1).
        datetime_range: ISO 8601 interval for scene search.
        cloud_max: Maximum ``eo:cloud_cover`` percentage.
        fill_holes_pixels: Max search distance for nodata hole filling.
            Set to 0 to disable.

    Returns:
        The *out_path* that was written.
    """
    west, south, east, north = _wgs84_bounds(embed_path)
    print(f"S2 RGB: WGS 84 bbox ({west:.4f}, {south:.4f}, {east:.4f}, {north:.4f})")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=(west, south, east, north),
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_max}},
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(
            "No Sentinel-2 items found. Widen the datetime range or raise cloud_max."
        )
    print(f"S2 RGB: {len(items)} scenes found")

    with rasterio.open(embed_path) as ref:
        epsg = ref.crs.to_epsg()
        if epsg is None:
            raise RuntimeError("Embedding CRS must have an EPSG code.")
        resolution = float(abs(ref.transform.a))

    proj_bounds = _projected_bounds(embed_path)
    print("S2 RGB: building median stack...")
    stack = stackstac.stack(
        items,
        assets=["B04", "B03", "B02"],
        bounds=proj_bounds,
        resolution=resolution,
        epsg=epsg,
        dtype=np.float64,
        rescale=False,
        fill_value=np.nan,
    )
    median = stack.median(dim="time", skipna=True).compute()
    median = (median / 10_000.0).clip(0.0, 1.0).astype(np.float32)
    median = median.assign_coords(band=["B04", "B03", "B02"])
    median = median.rio.write_crs(f"EPSG:{epsg}")

    ref_da = rioxarray.open_rasterio(embed_path).isel(band=0, drop=True)
    aligned = median.rio.reproject_match(ref_da).astype("float32")

    if fill_holes_pixels > 0:
        print(f"S2 RGB: filling nodata holes (max {fill_holes_pixels} px)...")
        aligned = aligned.load()
        arr = np.asarray(aligned.values, dtype=np.float32).copy()
        for b in range(arr.shape[0]):
            plane = arr[b]
            valid = np.isfinite(plane)
            if not np.any(valid) or np.all(valid):
                continue
            mask = valid.astype(np.uint8)
            tmp = np.where(valid, plane, 0.0).astype(np.float32)
            arr[b] = rasterio.fill.fillnodata(
                tmp,
                mask=mask,
                max_search_distance=float(fill_holes_pixels),
                smoothing_iterations=0,
            )
        aligned.values[:] = arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.rio.to_raster(str(out_path), driver="COG", compress="deflate")
    print(f"S2 RGB: wrote {out_path}")
    return out_path


def download_worldcover(
    embed_path: Path,
    out_path: Path,
) -> Path:
    """Download ESA WorldCover 2021 v200 aligned to *embed_path*.

    The result is a single-band uint8 GeoTIFF with the same grid as the
    embedding COG. Class codes follow the ESA WorldCover legend (e.g. 95 for
    mangroves, 80 for permanent water).

    Args:
        embed_path: Path to the embedding COG (defines extent and CRS).
        out_path: Destination for the single-band GeoTIFF.

    Returns:
        The *out_path* that was written.
    """
    with rasterio.open(embed_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w = ref.height, ref.width
        west, south, east, north = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)

    print(f"WorldCover: WGS 84 bbox ({west:.4f}, {south:.4f}, {east:.4f}, {north:.4f})")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=(west, south, east, north),
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError("No WorldCover items found for this extent.")
    print(f"WorldCover: {len(items)} tiles")

    datasets = [rasterio.open(item.assets["map"].href) for item in items]
    try:
        if len(datasets) == 1:
            mosaic = datasets[0].read(1)
            mosaic_transform = datasets[0].transform
            mosaic_crs = datasets[0].crs
        else:
            mosaic_arr, mosaic_transform = merge(datasets)
            mosaic = mosaic_arr[0]
            mosaic_crs = datasets[0].crs
    finally:
        for ds in datasets:
            ds.close()

    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=mosaic,
        destination=dst,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=dst_h,
        width=dst_w,
        count=1,
        dtype="uint8",
        crs=dst_crs,
        transform=dst_transform,
        compress="deflate",
    ) as out:
        out.write(dst, 1)

    print(f"WorldCover: wrote {out_path}")
    return out_path

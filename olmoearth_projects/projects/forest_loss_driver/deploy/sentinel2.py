"""Utilities for identifying Sentinel-2 assets to show for a forest loss event."""

import functools
import json
import multiprocessing
import os
from collections.abc import Callable
from datetime import datetime, timedelta
from multiprocessing.pool import IMapIterator
from typing import Any

import requests
import shapely
import tqdm
from geojson_pydantic.geometries import parse_geometry_obj
from olmoearth_shared.api.common.search_filters import (
    DatetimeFilter,
    KeywordFilter,
    SortDirection,
)
from olmoearth_shared.api.datasets.items import ItemSortField, SearchItemsRequest
from olmoearth_shared.models.datasets.bands.sentinel2 import Sentinel2L2ABand
from olmoearth_shared.models.datasets.collection import Collection
from olmoearth_shared.models.datasets.data_provider_name import DataProviderName
from olmoearth_shared.models.datasets.item import Item as ApiItem
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.manage import retry
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry

# Duration and offset modifications to make to the forest loss event timestamp when
# looking for assets for visualization.
DURATION = timedelta(days=180)
PRE_OFFSET = timedelta(days=-300)
POST_OFFSET = timedelta(days=7)

# Collection and bands of the visual (TCI) asset that we show in the web app.
COLLECTION = Collection.SENTINEL_2_L2A
VISUAL_BANDS = [Sentinel2L2ABand.R, Sentinel2L2ABand.G, Sentinel2L2ABand.B]

# Environment variables specifying how to reach the olmoearth_datasets API.
API_URL_ENV_VAR = "OEDATASETS_API_URL"
API_TOKEN_ENV_VAR = "DATASETS_API_TOKEN"  # nosec

# Maximum number of candidate scenes to request from the API per search. The API sorts
# by ascending cloud cover, so the clearest scenes come first and we only need the first
# few that fully contain the event geometry.
CANDIDATE_LIMIT = 100

# Timeout (seconds) for olmoearth_datasets API requests.
REQUEST_TIMEOUT = 30


@functools.cache
def _get_session() -> requests.Session:
    """Get a cached requests Session for talking to the olmoearth_datasets API."""
    return requests.Session()


def _search_items(geometry: STGeometry, limit: int) -> list[ApiItem]:
    """Search olmoearth_datasets for Sentinel-2 items intersecting the geometry.

    We query the olmoearth_datasets API rather than the Planetary Computer STAC API
    directly since the STAC API is heavily rate limited. Items are returned sorted by
    ascending cloud cover.

    Args:
        geometry: the spatiotemporal geometry to search for.
        limit: the maximum number of items to return.

    Returns:
        list of matching items, sorted by ascending cloud cover.
    """
    wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
    geojson_dict = json.loads(shapely.to_geojson(wgs84_geometry.shp))
    request = SearchItemsRequest(
        collection=KeywordFilter[Collection](eq=COLLECTION),
        intersects_geometry=parse_geometry_obj(geojson_dict),
        limit=limit,
        offset=0,
        sort_by=ItemSortField.CLOUD_COVER,
        sort_direction=SortDirection.ASC,
        # We store unsigned URLs; the web app signs them itself when reading images.
        sign_urls=False,
    )
    if wgs84_geometry.time_range is not None:
        request.collected_at = DatetimeFilter(
            gte=wgs84_geometry.time_range[0],
            lt=wgs84_geometry.time_range[1],
        )

    api_url = os.environ[API_URL_ENV_VAR].rstrip("/")
    url = f"{api_url}/api/v1/items/search"
    headers = {"Content-Type": "application/json"}
    token = os.environ.get(API_TOKEN_ENV_VAR)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = _get_session().post(
        url,
        json=request.model_dump(mode="json", exclude_none=True),
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    records = response.json()["records"] or []
    return [ApiItem.model_validate(record) for record in records]


def _get_planetary_computer_visual_url(api_item: ApiItem) -> str | None:
    """Get the unsigned Planetary Computer visual (TCI) asset URL for an item.

    Args:
        api_item: the item from the olmoearth_datasets API.

    Returns:
        the unsigned Planetary Computer URL for the visual asset, or None if the item
        has no Planetary Computer visual asset.
    """
    provider = api_item.data_providers.get(DataProviderName.PLANETARY_COMPUTER)
    if provider is None:
        return None
    for asset in provider.assets:
        if set(asset.bands) == set(VISUAL_BANDS):
            return asset.url
    return None


class StarImapWrapper:
    """Wrapper for a function to implement star_imap.

    A kwargs dict is passed to this wrapper, which then calls the underlying function
    with the unwrapped kwargs.
    """

    def __init__(self, fn: Callable[..., Any]):
        """Create a new StarImap.

        Args:
            fn: the underlying function to call.
        """
        self.fn = fn

    def __call__(self, kwargs: dict[str, Any]) -> Any:
        """Wrapped call to the underlying function.

        Args:
            kwargs: dict of keyword arguments to pass to the function.
        """
        return self.fn(**kwargs)


def star_imap(
    p: multiprocessing.pool.Pool,
    fn: Callable[..., Any],
    kwargs_list: list[dict[str, Any]],
) -> IMapIterator:
    """Wrapper for Pool.imap that exposes kwargs to the function.

    Args:
        p: the multiprocessing.pool.Pool to use.
        fn: the function to call, which accepts keyword arguments.
        kwargs_list: list of kwargs dicts to pass to the function.

    Returns:
        generator for outputs from the function in arbitrary order.
    """
    return p.imap(StarImapWrapper(fn), kwargs_list)


def _get_assets_for_feat(
    feat: Feature, num_assets: int
) -> tuple[list[dict], list[dict]]:
    """Get the Sentinel-2 assets for a feature.

    Args:
        feat: the feature to identify assets for.
        num_assets: the maximum number of pre and post assets to get.

    Returns: a tuple (pre_assets, post_assets) containing lists of Planetary Computer
        asset URLs for pre-event Sentinel-2 TCI images and post-event images.
    """

    def get_assets_for_time_offset(
        offset: timedelta, duration: timedelta
    ) -> list[dict[str, Any]]:
        """Get assets after applying a time offset on the feature geometry."""
        ts = datetime.fromisoformat(feat.properties["oe_start_time"])
        geometry = STGeometry(
            feat.geometry.projection,
            shapely.box(*feat.geometry.shp.bounds),
            (ts + offset, ts + offset + duration),
        )
        # The geometry that the scene must fully contain, in WGS84 (matching the
        # projection of the geometries returned by the API).
        box_shp = geometry.to_projection(WGS84_PROJECTION).shp

        # Run request with retries since the API may have transient errors.
        api_items: list[ApiItem] = retry(
            fn=lambda: _search_items(geometry, limit=CANDIDATE_LIMIT),
            retry_max_attempts=10,
            retry_backoff=timedelta(seconds=5),
        )

        assets: list[dict] = []
        for api_item in api_items:
            if len(assets) >= num_assets:
                break
            # Only use scenes that fully contain the event geometry. We use covers()
            # rather than contains() so degenerate (e.g. point) geometries, whose
            # interior is empty, are still matched.
            item_shp = shapely.geometry.shape(api_item.properties.geometry)
            if not item_shp.covers(box_shp):
                continue
            # Get the unsigned Planetary Computer URL for the visual (TCI) asset. The
            # web app signs the URL itself when it reads the image.
            url = _get_planetary_computer_visual_url(api_item)
            if url is None:
                continue
            assets.append(
                dict(
                    url=url,
                    ts=api_item.properties.collected_at.isoformat(),
                )
            )

        return assets

    pre_assets = get_assets_for_time_offset(PRE_OFFSET, DURATION)
    post_assets = get_assets_for_time_offset(POST_OFFSET, DURATION)
    return (pre_assets, post_assets)


def get_sentinel2_assets(
    features: list[Feature], workers: int, num_assets: int = 3
) -> None:
    """Identify suitable pre and post Sentinel-2 assets for each feature.

    The Planetary Computer asset URLs are added as properties to the feature. The
    assets are used for visualization in the website.

    Args:
        features: the forest loss event features. Their properties will be updated.
        workers: number of worker processes to use.
        num_assets: the maximum number of pre and post assets to get
    """
    p = multiprocessing.Pool(workers)
    get_assets_for_feat_args = [
        dict(
            feat=feat,
            num_assets=num_assets,
        )
        for feat in features
    ]
    outputs = tqdm.tqdm(
        star_imap(p, _get_assets_for_feat, get_assets_for_feat_args),
        desc="Identify pre/post Sentinel-2 assets",
        total=len(features),
    )
    for feat, (pre_assets, post_assets) in zip(features, outputs):
        feat.properties["pre_assets"] = pre_assets
        feat.properties["post_assets"] = post_assets
    p.close()

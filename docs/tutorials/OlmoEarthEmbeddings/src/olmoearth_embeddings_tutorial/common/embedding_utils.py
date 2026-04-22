"""Embedding loading utilities for OlmoEarth tutorial scripts.

Provides a consistent interface for reading OlmoEarth embedding COGs and
identifying valid (non-nodata) pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

NODATA_EMB: float = -128.0
"""Sentinel value used in int8 embedding COGs for missing data."""


def load_embeddings(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, rasterio.DatasetReader]:
    """Load an OlmoEarth embedding COG.

    Args:
        path: Path to a multi-band int8 COG produced by Studio.

    Returns:
        A 3-tuple of:
        - ``embeddings``: float32 array of shape ``(C, H, W)``.
        - ``valid``: boolean mask of shape ``(H, W)`` where ``True`` means all
          bands have real data (not the nodata sentinel).
        - ``dataset``: the open :class:`rasterio.DatasetReader`. The caller is
          responsible for closing it when done.
    """
    dataset = rasterio.open(path)
    embeddings = dataset.read().astype(np.float32)
    valid = np.all(embeddings != NODATA_EMB, axis=0)
    return embeddings, valid, dataset

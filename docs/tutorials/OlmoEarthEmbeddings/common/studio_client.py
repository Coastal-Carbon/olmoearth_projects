"""OlmoEarth Studio API client for embeddings tutorials.

Wraps the Studio REST API with helpers for project creation, model setup,
prediction submission, polling, and result download.  Also provides simple
JSON-based config persistence so scripts can share project/model IDs.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://olmoearth.allenai.org"


class StudioClient:
    """Thin wrapper around the OlmoEarth Studio REST API.

    Args:
        api_key: Bearer token for authentication.
        base_url: Studio API root (no trailing slash).
    """

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
        """Initialize the client with an API key and optional base URL."""
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._base = base_url.rstrip("/")

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Issue a GET request and return the JSON response body."""
        resp = self._session.get(f"{self._base}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def post(self, path: str, json_body: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Issue a POST request and return the JSON response body."""
        resp = self._session.post(f"{self._base}{path}", json=json_body, **kwargs)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def create_project(self, name: str) -> dict[str, Any]:
        """Create a new project and return its metadata."""
        return self.post("/api/v1/projects", {"name": name})

    def create_embeddings_model(
        self,
        project_id: str,
        model_name: str,
        encoder_variant: str = "tiny",
        resolution: str = "forty_meter",
        num_periods: int = 12,
        imagery_sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an embeddings model via the wizard endpoint."""
        if imagery_sources is None:
            imagery_sources = ["sentinel2_l2a"]
        body: dict[str, Any] = {
            "model_type": "embeddings",
            "wizard_answers": {
                "wizard_id": "embeddings_v1",
                "project_id": project_id,
                "model_name": model_name,
                "encoder_variant": encoder_variant,
                "resolution": resolution,
                "num_periods": num_periods,
                "imagery_sources": imagery_sources,
            },
        }
        return self.post("/api/v1/models/from_config", body)

    def create_prediction(
        self,
        project_id: str,
        model_id: str,
        name: str,
        geojson: dict[str, Any],
        datetime_range: str = "2024-01-01/2025-01-01",
    ) -> dict[str, Any]:
        """Submit a new prediction job."""
        body: dict[str, Any] = {
            "project_id": project_id,
            "model_id": model_id,
            "name": name,
            "geojson": geojson,
            "datetime": datetime_range,
        }
        return self.post("/api/v1/predictions", body)

    def get_prediction(
        self,
        prediction_id: str,
        include_result_metadata: bool = True,
    ) -> dict[str, Any]:
        """Fetch current prediction status."""
        params: dict[str, Any] = {}
        if include_result_metadata:
            params["include_result_metadata"] = "true"
        return self.get(f"/api/v1/predictions/{prediction_id}", params=params)

    def poll_prediction(
        self,
        prediction_id: str,
        poll_interval: float = 10.0,
        max_wait: float = 3600.0,
    ) -> dict[str, Any]:
        """Block until the prediction reaches a terminal status.

        Uses exponential backoff starting from *poll_interval* up to 60 s.

        Raises:
            RuntimeError: If the prediction fails or times out.
        """
        elapsed = 0.0
        interval = poll_interval
        while elapsed < max_wait:
            pred = self.get_prediction(prediction_id)
            status = pred.get("status", "unknown")
            print(f"  prediction {prediction_id}: {status} ({elapsed:.0f}s)")
            if status == "completed":
                return pred
            if status in ("failed", "cancelled"):
                raise RuntimeError(
                    f"Prediction {prediction_id} ended with status '{status}'"
                )
            time.sleep(interval)
            elapsed += interval
            interval = min(interval * 1.5, 60.0)
        raise RuntimeError(
            f"Prediction {prediction_id} still running after {max_wait:.0f}s"
        )

    def download_prediction_result(
        self,
        download_token: str,
        out_dir: Path,
    ) -> Path:
        """Download a prediction result ZIP and extract the embedding COG.

        Returns:
            Path to the extracted .tif file.
        """
        url = (
            f"{self._base}/api/v1/prediction-results/files"
            f"?download_token={download_token}"
        )
        resp = self._session.get(url, stream=True)
        resp.raise_for_status()

        buf = io.BytesIO(resp.content)
        out_dir.mkdir(parents=True, exist_ok=True)
        tif_path: Path | None = None
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                if name.endswith(".tif"):
                    zf.extract(name, out_dir)
                    tif_path = out_dir / name
        if tif_path is None:
            raise RuntimeError("No .tif found in prediction result ZIP")
        print(f"  extracted {tif_path}")
        return tif_path


def load_config(config_path: Path) -> dict[str, Any]:
    """Read a JSON config file."""
    with open(config_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def save_config(config_path: Path, data: dict[str, Any]) -> None:
    """Write a JSON config file (pretty-printed)."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Saved config to {config_path}")


def client_from_args(
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> StudioClient:
    """Build a StudioClient from an explicit key or the environment.

    Falls back to the ``OLMOEARTH_API_KEY`` environment variable when
    *api_key* is ``None``.

    Raises:
        SystemExit: If no API key is available.
    """
    key = api_key or os.environ.get("OLMOEARTH_API_KEY")
    if not key:
        raise SystemExit(
            "Provide --api-key or set the OLMOEARTH_API_KEY environment variable."
        )
    return StudioClient(key, base_url)

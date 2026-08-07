"""Client for interacting with the OlmoEarth Studio predictions API.

This wraps the Studio REST API with helpers for submitting prediction jobs and
looking up their status and results. It centralizes the HTTP details (base URL,
authentication headers, timeouts, and the ``records`` response envelope) that would
otherwise be duplicated across projects.
"""

import os
from typing import Any

import requests

DEFAULT_BASE_URL = "https://olmoearth.allenai.org/api/v1"

# Environment variable that must be set to authenticate with Studio.
API_KEY_ENV_VAR = "STUDIO_API_KEY"  # nosec

# Default timeouts (seconds) for connecting to and reading from the API. Uploading a
# prediction request (which includes the input GeoJSON) can take a while, so the read
# timeout for those requests is larger.
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_UPLOAD_TIMEOUT = 300


class StudioClient:
    """Thin client for the OlmoEarth Studio predictions API.

    Args:
        api_key: the Bearer token used to authenticate with Studio.
        base_url: the Studio API root (no trailing slash).
        request_timeout: timeout (seconds) for regular requests.
        upload_timeout: read timeout (seconds) for requests that upload data, such as
            submitting a prediction with its input GeoJSON.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        upload_timeout: int = DEFAULT_UPLOAD_TIMEOUT,
    ) -> None:
        """Create a new StudioClient."""
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._upload_timeout = upload_timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "StudioClient":
        """Create a StudioClient using the API key from the environment.

        The API key is read from the STUDIO_API_KEY environment variable.

        Args:
            kwargs: additional keyword arguments passed to the constructor.

        Returns:
            the StudioClient.
        """
        return cls(api_key=os.environ[API_KEY_ENV_VAR], **kwargs)

    def _get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Issue a GET request and return the parsed JSON response body."""
        kwargs.setdefault("timeout", self._request_timeout)
        response = self._session.get(f"{self._base_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def _post(self, path: str, json_body: Any, **kwargs: Any) -> dict[str, Any]:
        """Issue a POST request and return the parsed JSON response body."""
        kwargs.setdefault("timeout", (self._request_timeout, self._upload_timeout))
        response = self._session.post(
            f"{self._base_url}{path}", json=json_body, **kwargs
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _single_record(json_data: dict[str, Any]) -> dict[str, Any]:
        """Extract the single record from a Studio API response.

        Many Studio endpoints wrap their result in a "records" list. For the requests
        we make here, we always expect exactly one record.

        Args:
            json_data: the parsed JSON response body.

        Returns:
            the single record dict.

        Raises:
            ValueError: if the response does not contain exactly one record.
        """
        if "records" not in json_data or len(json_data["records"]) != 1:
            raise ValueError(
                f"expected response to have one record, but got {json_data}"
            )
        return json_data["records"][0]  # type: ignore[no-any-return]

    def create_prediction(
        self,
        project_id: str,
        model_id: str,
        name: str,
        geojson: dict[str, Any],
        min_window_success_rate: float | None = None,
    ) -> str:
        """Submit a prediction job to Studio.

        Args:
            project_id: the project to run the prediction in.
            model_id: the model to run.
            name: the name to give the prediction job.
            geojson: the input GeoJSON FeatureCollection.
            min_window_success_rate: if set, the minimum fraction of windows that must
                succeed for the job to be considered successful.

        Returns:
            the ID of the created prediction job.
        """
        json_request_data: dict[str, Any] = {
            "model_id": model_id,
            "name": name,
            "project_id": project_id,
            "geojson": geojson,
        }
        if min_window_success_rate is not None:
            json_request_data["min_window_success_rate"] = min_window_success_rate

        json_data = self._post("/predictions", json_request_data)
        return self._single_record(json_data)["id"]  # type: ignore[no-any-return]

    def get_prediction(self, job_id: str) -> dict[str, Any]:
        """Get the record for a prediction job.

        Args:
            job_id: the prediction job ID.

        Returns:
            the prediction job record.
        """
        json_data = self._get(f"/predictions/{job_id}")
        return self._single_record(json_data)

    def get_prediction_result_url(self, job_id: str) -> str:
        """Get the download URL for a prediction job's result.

        The returned URL points to a zip archive containing the prediction result. The
        URL is authorized by an embedded download token, so it does not require the
        client's authentication headers to download.

        Args:
            job_id: the prediction job ID.

        Returns:
            the URL to download the prediction result zip archive.
        """
        record = self.get_prediction(job_id)
        download_token = record["result"]["download_token"]
        return (
            f"{self._base_url}/prediction-results/files?download_token={download_token}"
        )

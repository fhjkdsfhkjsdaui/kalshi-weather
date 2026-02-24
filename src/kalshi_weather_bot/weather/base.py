"""Provider-agnostic weather interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from .models import WeatherFetchResult


class WeatherProvider(ABC):
    """Base contract for weather providers used by the ingestion pipeline."""

    @abstractmethod
    def fetch_forecast(
        self,
        *,
        lat: float | None = None,
        lon: float | None = None,
        forecast_url: str | None = None,
        forecast_type: Literal["daily", "hourly"] = "daily",
    ) -> WeatherFetchResult:
        """Fetch and normalize a forecast snapshot."""

    @abstractmethod
    def close(self) -> None:
        """Release provider resources."""

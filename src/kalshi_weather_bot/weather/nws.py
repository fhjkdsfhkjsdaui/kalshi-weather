"""NWS (api.weather.gov) weather provider implementation."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from ..config import Settings
from ..exceptions import WeatherProviderError
from ..redaction import sanitize_text
from .base import WeatherProvider
from .models import WeatherFetchResult, WeatherForecastPeriod, WeatherLocation, WeatherSnapshot


class NWSWeatherProvider(WeatherProvider):
    """Fetches and normalizes forecast snapshots from api.weather.gov."""

    provider_name = "nws"
    provider_version = "day2-v1"

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        max_retries: int = 1,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._client = httpx.Client(
            timeout=settings.weather_timeout_seconds,
            headers={
                "Accept": "application/geo+json",
                "User-Agent": settings.nws_user_agent,
            },
        )

    def __enter__(self) -> NWSWeatherProvider:
        return self

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch_forecast(
        self,
        *,
        lat: float | None = None,
        lon: float | None = None,
        forecast_url: str | None = None,
        forecast_type: Literal["daily", "hourly"] = "daily",
    ) -> WeatherFetchResult:
        """Fetch forecast via either points lookup or direct forecast URL."""
        self._validate_input(lat=lat, lon=lon, forecast_url=forecast_url)

        points_payload: dict[str, Any] | None = None
        points_url: str | None = None

        if forecast_url is None:
            if lat is None or lon is None:
                raise WeatherProviderError(
                    "Internal error: coordinates required but not provided after validation."
                )
            points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
            points_payload = self._request_json(points_url, context="points lookup")
            forecast_url = self._extract_forecast_url(points_payload, forecast_type=forecast_type)

        forecast_payload = self._request_json(forecast_url, context="forecast fetch")
        snapshot = self._normalize_snapshot(
            forecast_payload=forecast_payload,
            forecast_url=forecast_url,
            forecast_type=forecast_type,
            points_payload=points_payload,
            points_url=points_url,
            fallback_lat=lat,
            fallback_lon=lon,
        )
        return WeatherFetchResult(
            snapshot=snapshot,
            raw_forecast_payload=forecast_payload,
            raw_points_payload=points_payload,
        )

    def _validate_input(
        self,
        *,
        lat: float | None,
        lon: float | None,
        forecast_url: str | None,
    ) -> None:
        if forecast_url and (lat is not None or lon is not None):
            raise WeatherProviderError("Use either --url or --lat/--lon, not both.")
        if forecast_url:
            return
        if lat is None or lon is None:
            raise WeatherProviderError("Missing coordinates: provide both latitude and longitude.")
        if not (-90 <= lat <= 90):
            raise WeatherProviderError(f"Invalid latitude {lat}; expected between -90 and 90.")
        if not (-180 <= lon <= 180):
            raise WeatherProviderError(f"Invalid longitude {lon}; expected between -180 and 180.")

    def _request_json(self, url: str, context: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Don't retry 4xx client errors except 429 rate-limit.
                if 400 <= status < 500 and status != 429:
                    raise WeatherProviderError(
                        f"NWS {context} failed with status {status} "
                        f"at {url}: {sanitize_text(exc.response.text[:300])}"
                    ) from exc
                last_error = exc
                if attempt < self._max_retries:
                    self.logger.warning(
                        "NWS %s failed (HTTP %d); retrying",
                        context, status,
                    )
                    time.sleep(self._retry_delay)
                    continue
                raise WeatherProviderError(
                    f"NWS {context} failed with status {status} "
                    f"at {url}: {sanitize_text(exc.response.text[:300])}"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    self.logger.warning(
                        "NWS %s request failed (%s); retrying",
                        context, type(exc).__name__,
                    )
                    time.sleep(self._retry_delay)
                    continue
                raise WeatherProviderError(
                    f"NWS {context} request failed at {url}: {sanitize_text(str(exc))}"
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise WeatherProviderError(
                    f"NWS {context} returned non-JSON response at {url}."
                ) from exc

            if not isinstance(payload, dict):
                raise WeatherProviderError(
                    f"NWS {context} returned unexpected payload type "
                    f"{type(payload).__name__} at {url}."
                )
            return payload

        raise WeatherProviderError(
            f"NWS {context} failed after retries: {last_error}"
        )

    def _extract_forecast_url(
        self, points_payload: dict[str, Any], forecast_type: Literal["daily", "hourly"]
    ) -> str:
        properties = points_payload.get("properties")
        if not isinstance(properties, dict):
            raise WeatherProviderError("NWS points payload missing 'properties' object.")

        # TODO(NWS_API): confirm if additional forecast URLs are needed for special products.
        candidate_key = "forecastHourly" if forecast_type == "hourly" else "forecast"
        forecast_url = properties.get(candidate_key)
        if not isinstance(forecast_url, str) or not forecast_url.strip():
            raise WeatherProviderError(
                f"NWS points payload missing expected '{candidate_key}' forecast URL."
            )
        return forecast_url.strip()

    def _normalize_snapshot(
        self,
        *,
        forecast_payload: dict[str, Any],
        forecast_url: str,
        forecast_type: Literal["daily", "hourly"],
        points_payload: dict[str, Any] | None,
        points_url: str | None,
        fallback_lat: float | None,
        fallback_lon: float | None,
    ) -> WeatherSnapshot:
        properties = forecast_payload.get("properties")
        if not isinstance(properties, dict):
            raise WeatherProviderError("NWS forecast payload missing 'properties' object.")

        raw_periods = properties.get("periods")
        if not isinstance(raw_periods, list):
            raise WeatherProviderError("NWS forecast payload missing 'properties.periods' list.")
        if not raw_periods:
            raise WeatherProviderError("NWS forecast payload contained no forecast periods.")

        periods = [self._normalize_period(item) for item in raw_periods if isinstance(item, dict)]
        if not periods:
            raise WeatherProviderError(
                "NWS forecast payload periods were present but not parseable."
            )

        location = self._normalize_location(
            points_payload=points_payload,
            forecast_payload=forecast_payload,
            fallback_lat=fallback_lat,
            fallback_lon=fallback_lon,
        )
        return WeatherSnapshot(
            provider=self.provider_name,
            provider_version=self.provider_version,
            retrieval_timestamp=datetime.now(UTC),
            forecast_type=forecast_type,
            source_url=forecast_url,
            points_url=points_url,
            generated_timestamp=self._parse_datetime(properties.get("generatedAt")),
            updated_timestamp=self._parse_datetime(properties.get("updated")),
            location=location,
            periods=periods,
            raw_payload_path=None,
        )

    def _normalize_location(
        self,
        *,
        points_payload: dict[str, Any] | None,
        forecast_payload: dict[str, Any],
        fallback_lat: float | None,
        fallback_lon: float | None,
    ) -> WeatherLocation:
        grid_id: str | None = None
        grid_x: int | None = None
        grid_y: int | None = None
        city: str | None = None
        state: str | None = None

        if points_payload is not None:
            props = points_payload.get("properties")
            if isinstance(props, dict):
                grid_id = self._as_str(props.get("gridId"))
                grid_x = self._as_int(props.get("gridX"))
                grid_y = self._as_int(props.get("gridY"))
                relative_location = props.get("relativeLocation")
                if isinstance(relative_location, dict):
                    relative_props = relative_location.get("properties")
                    if isinstance(relative_props, dict):
                        city = self._as_str(relative_props.get("city"))
                        state = self._as_str(relative_props.get("state"))

        lat = fallback_lat
        lon = fallback_lon
        forecast_geometry = forecast_payload.get("geometry")
        if isinstance(forecast_geometry, dict):
            geom_type = forecast_geometry.get("type")
            coords = forecast_geometry.get("coordinates")
            # GeoJSON Point has [lon, lat]; Polygon has [[[lon, lat], ...]].
            # Only extract from Point to avoid silent data corruption.
            if (
                geom_type == "Point"
                and isinstance(coords, list)
                and len(coords) >= 2
                and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))
            ):
                lon = float(coords[0])
                lat = float(coords[1])

        return WeatherLocation(
            latitude=lat,
            longitude=lon,
            grid_id=grid_id,
            grid_x=grid_x,
            grid_y=grid_y,
            city=city,
            state=state,
        )

    def _normalize_period(self, period: dict[str, Any]) -> WeatherForecastPeriod:
        start_time = self._parse_datetime(period.get("startTime"))
        if start_time is None:
            raise WeatherProviderError("NWS period missing or invalid 'startTime'.")

        precip_value = period.get("probabilityOfPrecipitation")
        precip: float | None = None
        if isinstance(precip_value, dict):
            raw_val = precip_value.get("value")
            if isinstance(raw_val, (int, float)):
                precip = float(raw_val)

        return WeatherForecastPeriod(
            name=self._as_str(period.get("name")),
            start_time=start_time,
            end_time=self._parse_datetime(period.get("endTime")),
            temperature=self._as_float(period.get("temperature")),
            temperature_unit=self._as_str(period.get("temperatureUnit")),
            wind_speed=self._as_str(period.get("windSpeed")),
            wind_direction=self._as_str(period.get("windDirection")),
            short_forecast=self._as_str(period.get("shortForecast")),
            detailed_forecast=self._as_str(period.get("detailedForecast")),
            probability_of_precipitation=precip,
        )

    @staticmethod
    def _as_str(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return None
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return None

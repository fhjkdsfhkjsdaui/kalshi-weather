"""Day 2 tests for NWS provider assumptions and normalization."""

from __future__ import annotations

import logging
from argparse import Namespace
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kalshi_weather_bot.config import Settings
from kalshi_weather_bot.exceptions import JournalError, WeatherProviderError
from kalshi_weather_bot.journal import JournalWriter
from kalshi_weather_bot.weather.nws import NWSWeatherProvider
from kalshi_weather_bot.weather_cli import _validate_cli_input


def _make_settings(**overrides: Any) -> Any:
    defaults = {
        "weather_timeout_seconds": 5.0,
        "nws_user_agent": "kalshi-weather-bot-tests/0.1 (contact: test@example.com)",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_provider(**settings_overrides: Any) -> NWSWeatherProvider:
    settings = _make_settings(**settings_overrides)
    logger = logging.getLogger("test_nws_provider")
    return NWSWeatherProvider(settings=settings, logger=logger)


def test_points_flow_normalizes_hourly_forecast() -> None:
    provider = _make_provider()
    points_url = "https://api.weather.gov/points/40.7128,-74.0060"
    forecast_hourly_url = "https://api.weather.gov/gridpoints/OKX/33,35/forecast/hourly"

    points_payload = {
        "properties": {
            "forecast": "https://api.weather.gov/gridpoints/OKX/33,35/forecast",
            "forecastHourly": forecast_hourly_url,
            "gridId": "OKX",
            "gridX": 33,
            "gridY": 35,
            "relativeLocation": {"properties": {"city": "New York", "state": "NY"}},
        }
    }
    forecast_payload = {
        "geometry": {"type": "Point", "coordinates": [-74.0060, 40.7128]},
        "properties": {
            "generatedAt": "2026-02-24T12:00:00Z",
            "updated": "2026-02-24T12:15:00Z",
            "periods": [
                {
                    "name": "This Afternoon",
                    "startTime": "2026-02-24T12:00:00-05:00",
                    "endTime": "2026-02-24T18:00:00-05:00",
                    "temperature": 41,
                    "temperatureUnit": "F",
                    "windSpeed": "8 mph",
                    "windDirection": "NW",
                    "shortForecast": "Partly Cloudy",
                    "detailedForecast": "Partly cloudy with a slight chance of rain.",
                    "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 20},
                }
            ],
        },
    }

    responses = {
        points_url: points_payload,
        forecast_hourly_url: forecast_payload,
    }
    provider._request_json = lambda url, context: responses[url]  # type: ignore[assignment]

    result = provider.fetch_forecast(lat=40.7128, lon=-74.0060, forecast_type="hourly")
    snapshot = result.snapshot

    assert snapshot.provider == "nws"
    assert snapshot.forecast_type == "hourly"
    assert snapshot.source_url == forecast_hourly_url
    assert snapshot.points_url == points_url
    assert snapshot.location.city == "New York"
    assert snapshot.location.state == "NY"
    assert snapshot.location.grid_id == "OKX"
    assert snapshot.location.grid_x == 33
    assert snapshot.location.grid_y == 35
    assert len(snapshot.periods) == 1
    assert snapshot.periods[0].temperature == 41
    assert snapshot.periods[0].temperature_unit == "F"
    assert snapshot.periods[0].probability_of_precipitation == 20
    assert result.raw_points_payload == points_payload
    assert result.raw_forecast_payload == forecast_payload


def test_direct_forecast_url_mode_skips_points_lookup() -> None:
    provider = _make_provider()
    direct_url = "https://api.weather.gov/gridpoints/LWX/96,70/forecast"

    forecast_payload = {
        "geometry": {"type": "Point", "coordinates": [-77.0365, 38.8951]},
        "properties": {
            "generatedAt": "2026-02-24T10:00:00+00:00",
            "updated": "2026-02-24T10:10:00+00:00",
            "periods": [
                {
                    "name": "Tonight",
                    "startTime": "2026-02-24T18:00:00-05:00",
                    "endTime": "2026-02-25T06:00:00-05:00",
                    "temperature": 32,
                    "temperatureUnit": "F",
                    "shortForecast": "Clear",
                }
            ],
        },
    }

    called_urls: list[str] = []

    def _fake_request(url: str, context: str) -> dict[str, Any]:
        called_urls.append(url)
        assert context == "forecast fetch"
        return forecast_payload

    provider._request_json = _fake_request  # type: ignore[assignment]

    result = provider.fetch_forecast(forecast_url=direct_url, forecast_type="daily")
    assert called_urls == [direct_url]
    assert result.raw_points_payload is None
    assert result.snapshot.points_url is None
    assert result.snapshot.source_url == direct_url
    assert result.snapshot.location.latitude == 38.8951
    assert result.snapshot.location.longitude == -77.0365


def test_points_payload_missing_expected_forecast_url_raises() -> None:
    provider = _make_provider()
    provider._request_json = lambda url, context: {"properties": {}}  # type: ignore[assignment]

    with pytest.raises(WeatherProviderError, match="missing expected 'forecastHourly'"):
        provider.fetch_forecast(lat=39.0, lon=-77.0, forecast_type="hourly")


def test_malformed_forecast_payload_raises() -> None:
    provider = _make_provider()
    points_url = "https://api.weather.gov/points/39.0000,-77.0000"
    forecast_url = "https://api.weather.gov/gridpoints/LWX/96,70/forecast"

    responses = {
        points_url: {"properties": {"forecast": forecast_url}},
        forecast_url: {"properties": {"periods": "not-a-list"}},
    }
    provider._request_json = lambda url, context: responses[url]  # type: ignore[assignment]

    with pytest.raises(WeatherProviderError, match="missing 'properties.periods' list"):
        provider.fetch_forecast(lat=39.0, lon=-77.0, forecast_type="daily")


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (None, -77.0),
        (39.0, None),
        (91.0, -77.0),
        (39.0, -181.0),
    ],
)
def test_invalid_lat_lon_input_raises(lat: float | None, lon: float | None) -> None:
    provider = _make_provider()
    with pytest.raises(WeatherProviderError):
        provider.fetch_forecast(lat=lat, lon=lon, forecast_type="daily")


def test_datetime_parsing_handles_z_suffix() -> None:
    parsed = NWSWeatherProvider._parse_datetime("2026-02-24T12:00:00Z")
    assert parsed is not None
    assert parsed.tzinfo == UTC


def test_normalized_snapshot_journaling_is_serializable(tmp_path: Path) -> None:
    provider = _make_provider()
    direct_url = "https://api.weather.gov/gridpoints/LWX/96,70/forecast"
    forecast_payload = {
        "geometry": {"type": "Point", "coordinates": [-77.0365, 38.8951]},
        "properties": {
            "generatedAt": "2026-02-24T10:00:00+00:00",
            "updated": "2026-02-24T10:10:00+00:00",
            "periods": [
                {
                    "name": "Tonight",
                    "startTime": "2026-02-24T18:00:00-05:00",
                    "endTime": "2026-02-25T06:00:00-05:00",
                    "temperature": 32,
                    "temperatureUnit": "F",
                    "shortForecast": "Clear",
                }
            ],
        },
    }
    provider._request_json = lambda url, context: forecast_payload  # type: ignore[assignment]
    result = provider.fetch_forecast(forecast_url=direct_url, forecast_type="daily")

    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="day2test",
    )
    snapshot_dict = result.snapshot.model_dump(mode="json")
    try:
        journal.write_event("weather_snapshot_normalized", payload=snapshot_dict)
        path = journal.write_raw_snapshot("nws_forecast", result.raw_forecast_payload)
    except JournalError as exc:  # pragma: no cover
        pytest.fail(f"Expected weather snapshot journaling to serialize cleanly, got: {exc}")
    assert path.exists()


def test_periods_with_only_non_dict_items_raises() -> None:
    """Periods list exists but contains no dicts — should raise."""
    provider = _make_provider()
    forecast_payload = {
        "properties": {
            "generatedAt": "2026-02-24T10:00:00Z",
            "periods": [None, "garbage", 42],
        },
    }
    provider._request_json = lambda url, context: forecast_payload  # type: ignore[assignment]
    with pytest.raises(WeatherProviderError, match="not parseable"):
        provider.fetch_forecast(
            forecast_url="https://api.weather.gov/gridpoints/LWX/96,70/forecast",
            forecast_type="daily",
        )


def test_empty_periods_list_raises() -> None:
    """Empty periods list should raise before attempting normalization."""
    provider = _make_provider()
    forecast_payload = {
        "properties": {
            "generatedAt": "2026-02-24T10:00:00Z",
            "periods": [],
        },
    }
    provider._request_json = lambda url, context: forecast_payload  # type: ignore[assignment]
    with pytest.raises(WeatherProviderError, match="no forecast periods"):
        provider.fetch_forecast(
            forecast_url="https://api.weather.gov/gridpoints/LWX/96,70/forecast",
            forecast_type="daily",
        )


def test_polygon_geometry_does_not_corrupt_coordinates() -> None:
    """NWS daily forecasts return Polygon geometry — should not extract lat/lon."""
    provider = _make_provider()
    forecast_payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-77.1, 38.9], [-77.0, 38.9], [-77.0, 38.8]]],
        },
        "properties": {
            "generatedAt": "2026-02-24T10:00:00Z",
            "periods": [
                {
                    "name": "Tonight",
                    "startTime": "2026-02-24T18:00:00-05:00",
                    "temperature": 32,
                    "temperatureUnit": "F",
                    "shortForecast": "Clear",
                }
            ],
        },
    }
    provider._request_json = lambda url, context: forecast_payload  # type: ignore[assignment]
    result = provider.fetch_forecast(
        forecast_url="https://api.weather.gov/gridpoints/LWX/96,70/forecast",
        forecast_type="daily",
    )
    # No lat/lon should be extracted from Polygon geometry (no fallback coords either)
    assert result.snapshot.location.latitude is None
    assert result.snapshot.location.longitude is None


def test_datetime_parsing_handles_offset() -> None:
    """NWS period times include timezone offsets like -05:00."""
    parsed = NWSWeatherProvider._parse_datetime("2026-02-24T18:00:00-05:00")
    assert parsed is not None
    assert parsed.tzinfo is not None
    # Should be converted to UTC: 23:00
    assert parsed.hour == 23


def test_precip_none_value_handled() -> None:
    """NWS sometimes returns probabilityOfPrecipitation with null value."""
    provider = _make_provider()
    forecast_payload = {
        "properties": {
            "generatedAt": "2026-02-24T10:00:00Z",
            "periods": [
                {
                    "name": "Tonight",
                    "startTime": "2026-02-24T18:00:00-05:00",
                    "temperature": 32,
                    "temperatureUnit": "F",
                    "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": None},
                    "shortForecast": "Clear",
                }
            ],
        },
    }
    provider._request_json = lambda url, context: forecast_payload  # type: ignore[assignment]
    result = provider.fetch_forecast(
        forecast_url="https://api.weather.gov/gridpoints/LWX/96,70/forecast",
        forecast_type="daily",
    )
    assert result.snapshot.periods[0].probability_of_precipitation is None


def test_settings_empty_weather_default_coords_parse_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("KALSHI_AUTH_MODE", "bearer")
    monkeypatch.setenv("KALSHI_BEARER_TOKEN", "token")
    monkeypatch.setenv("NWS_USER_AGENT", "test-agent")
    monkeypatch.setenv("WEATHER_DEFAULT_LAT", "")
    monkeypatch.setenv("WEATHER_DEFAULT_LON", "")

    settings = Settings(_env_file=None)
    assert settings.weather_default_lat is None
    assert settings.weather_default_lon is None


def test_validate_cli_input_url_mode_ignores_default_coords() -> None:
    args = Namespace(
        url="https://api.weather.gov/gridpoints/OKX/33,35/forecast",
        lat=None,
        lon=None,
        max_print=None,
    )
    settings = SimpleNamespace(weather_default_lat=40.0, weather_default_lon=-74.0)
    lat, lon = _validate_cli_input(args, settings)
    assert lat is None
    assert lon is None


def test_validate_cli_input_url_with_explicit_coords_raises() -> None:
    args = Namespace(
        url="https://api.weather.gov/gridpoints/OKX/33,35/forecast",
        lat=40.0,
        lon=None,
        max_print=None,
    )
    settings = SimpleNamespace(weather_default_lat=None, weather_default_lon=None)
    with pytest.raises(WeatherProviderError, match="Use either --url or --lat/--lon"):
        _validate_cli_input(args, settings)

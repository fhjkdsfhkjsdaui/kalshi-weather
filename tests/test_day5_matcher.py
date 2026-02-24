"""Day 5 matcher tests."""

from __future__ import annotations

from datetime import UTC, datetime

from kalshi_weather_bot.contracts.models import ParsedWeatherContract
from kalshi_weather_bot.signal.matcher import WeatherMarketMatcher, time_window_overlap
from kalshi_weather_bot.weather.models import (
    WeatherForecastPeriod,
    WeatherLocation,
    WeatherSnapshot,
)


def _contract(**overrides: object) -> ParsedWeatherContract:
    payload = {
        "provider_market_id": "KX-WEATHER-1",
        "ticker": "KX-WEATHER-1",
        "event_id": "KX-WEATHER",
        "raw_title": "Will high temperature in Seattle, WA be above 70F?",
        "weather_candidate": True,
        "weather_dimension": "temperature",
        "metric_subtype": "high_temp",
        "location_raw": "Seattle, WA",
        "location_normalized": {"city": "Seattle", "state": "WA"},
        "threshold_operator": ">",
        "threshold_value": 70,
        "threshold_unit": "F",
        "contract_start_time": datetime(2026, 2, 24, 0, 0, tzinfo=UTC),
        "contract_end_time": datetime(2026, 2, 24, 23, 59, tzinfo=UTC),
        "resolution_time": datetime(2026, 2, 25, 0, 0, tzinfo=UTC),
        "parse_confidence": 0.9,
        "parse_status": "parsed",
    }
    payload.update(overrides)
    return ParsedWeatherContract.model_validate(payload)


def _snapshot(**overrides: object) -> WeatherSnapshot:
    payload = {
        "provider": "nws",
        "provider_version": "day2-v1",
        "retrieval_timestamp": datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
        "forecast_type": "hourly",
        "source_url": "https://api.weather.gov/gridpoints/SEW/124,67/forecast/hourly",
        "location": WeatherLocation(city="Seattle", state="WA", latitude=47.6, longitude=-122.3),
        "periods": [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 9, 0, tzinfo=UTC),
                temperature=72,
                temperature_unit="F",
                wind_speed="12 mph",
                probability_of_precipitation=20,
            ),
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 12, 0, tzinfo=UTC),
                temperature=68,
                temperature_unit="F",
                wind_speed="10 mph",
                probability_of_precipitation=30,
            ),
        ],
    }
    payload.update(overrides)
    return WeatherSnapshot.model_validate(payload)


def test_time_window_overlap_helper() -> None:
    assert time_window_overlap(
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
        datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
    )
    assert not time_window_overlap(
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 3, 0, tzinfo=UTC),
    )


def test_matcher_happy_path_with_overlap() -> None:
    matcher = WeatherMarketMatcher()
    result = matcher.match_contract(_contract(), _snapshot())
    assert result.matched is True
    assert result.decision_code == "matched"
    assert result.selected_period_count == 2
    assert result.matcher_confidence > 0.5


def test_matcher_rejects_no_time_overlap() -> None:
    matcher = WeatherMarketMatcher()
    contract = _contract(
        contract_start_time=datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
        contract_end_time=datetime(2026, 3, 1, 1, 0, tzinfo=UTC),
    )
    result = matcher.match_contract(contract, _snapshot())
    assert result.matched is False
    assert result.decision_code == "no_time_window_overlap"


def test_matcher_rejects_location_mismatch() -> None:
    matcher = WeatherMarketMatcher()
    contract = _contract(location_normalized={"city": "Boston", "state": "MA"})
    result = matcher.match_contract(contract, _snapshot())
    assert result.matched is False
    assert result.decision_code == "location_mismatch"


def test_matcher_rejects_missing_forecast_periods() -> None:
    matcher = WeatherMarketMatcher()
    snapshot = _snapshot(periods=[])
    result = matcher.match_contract(_contract(), snapshot)
    assert result.matched is False
    assert result.decision_code == "no_forecast_periods"


def test_matcher_adds_timezone_assumption_when_location_present() -> None:
    matcher = WeatherMarketMatcher()
    result = matcher.match_contract(_contract(), _snapshot())
    assert any("timezone" in assumption.lower() for assumption in result.assumptions)


# --- Time window overlap edge cases ---


class TestTimeWindowOverlap:
    """Exercise time_window_overlap boundary and None handling."""

    def test_touching_endpoints_overlap(self) -> None:
        """Inclusive endpoints: [0-1] and [1-2] should overlap."""
        assert time_window_overlap(
            datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
        )

    def test_none_start_returns_false(self) -> None:
        assert not time_window_overlap(
            None,
            datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
        )

    def test_none_end_defaults_to_one_hour(self) -> None:
        """If end_a is None, defaults to start_a + 1h."""
        assert time_window_overlap(
            datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            None,
            datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
        )

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetimes should be treated as UTC via _as_utc."""
        assert time_window_overlap(
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 1, 0),
            datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
        )


# --- Location matching edge cases ---


class TestLocationMatch:
    """Exercise location matching with varied inputs."""

    def test_city_only_match_returns_lower_confidence(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(location_normalized={"city": "Seattle", "state": None})
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True
        assert result.matcher_confidence > 0

    def test_state_only_match_returns_reduced_confidence(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(location_normalized={"city": None, "state": "WA"})
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True

    def test_no_contract_location_still_matches(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(location_normalized=None)
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True

    def test_state_mismatch_rejects(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(location_normalized={"city": "Seattle", "state": "OR"})
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is False

    def test_city_case_insensitive(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(location_normalized={"city": "SEATTLE", "state": "WA"})
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True


# --- Missing contract window fallback ---


class TestContractWindowEdgeCases:
    """Cover _contract_window assumptions."""

    def test_missing_contract_window_uses_all_periods(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(
            contract_start_time=None,
            contract_end_time=None,
            resolution_time=None,
        )
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True
        assert result.selected_period_count == 2
        assert any("missing" in a.lower() for a in result.assumptions)

    def test_inverted_window_swapped_with_assumption(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(
            contract_start_time=datetime(2026, 2, 24, 23, 59, tzinfo=UTC),
            contract_end_time=datetime(2026, 2, 24, 0, 0, tzinfo=UTC),
        )
        result = matcher.match_contract(contract, _snapshot())
        # The window swap should still allow matching
        assert result.matched is True
        assert any("inverted" in a.lower() for a in result.assumptions)

    def test_start_missing_uses_24h_lookback(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(
            contract_start_time=None,
            contract_end_time=datetime(2026, 2, 25, 0, 0, tzinfo=UTC),
        )
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True
        assert any("start missing" in a.lower() for a in result.assumptions)


# --- Metric period selection ---


class TestMetricPeriodSelection:
    """Cover _select_metric_periods for different dimensions."""

    def test_unsupported_dimension_rejected(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(weather_dimension=None)
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is False
        assert result.decision_code == "unsupported_metric_dimension"

    def test_wind_periods_selected(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(weather_dimension="wind")
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True
        assert result.selected_period_count == 2

    def test_precipitation_periods_selected(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(weather_dimension="precipitation")
        result = matcher.match_contract(contract, _snapshot())
        assert result.matched is True
        assert result.selected_period_count == 2

    def test_metric_unavailable_when_data_missing(self) -> None:
        matcher = WeatherMarketMatcher()
        contract = _contract(weather_dimension="temperature")
        snap = _snapshot(
            periods=[
                WeatherForecastPeriod(
                    start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                    end_time=datetime(2026, 2, 24, 9, 0, tzinfo=UTC),
                    temperature=None,
                ),
            ]
        )
        result = matcher.match_contract(contract, snap)
        assert result.matched is False
        assert result.decision_code == "metric_data_unavailable"

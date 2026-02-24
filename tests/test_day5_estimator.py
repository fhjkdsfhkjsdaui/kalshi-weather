"""Day 5 estimator tests."""

from __future__ import annotations

from datetime import UTC, datetime

from kalshi_weather_bot.contracts.models import ParsedWeatherContract
from kalshi_weather_bot.signal.estimator import WeatherProbabilityEstimator
from kalshi_weather_bot.signal.models import MatchResult
from kalshi_weather_bot.weather.models import WeatherForecastPeriod


def _contract(**overrides: object) -> ParsedWeatherContract:
    payload = {
        "provider_market_id": "KX-WEATHER-1",
        "raw_title": "Will high temperature in Seattle, WA be above 70F?",
        "weather_candidate": True,
        "weather_dimension": "temperature",
        "metric_subtype": "high_temp",
        "threshold_operator": ">",
        "threshold_value": 70,
        "threshold_unit": "F",
        "parse_confidence": 0.9,
        "parse_status": "parsed",
    }
    payload.update(overrides)
    return ParsedWeatherContract.model_validate(payload)


def _match_result(periods: list[WeatherForecastPeriod], **overrides: object) -> MatchResult:
    payload = {
        "matched": True,
        "decision_code": "matched",
        "selected_period_indices": list(range(len(periods))),
        "selected_period_count": len(periods),
        "selected_periods": periods,
        "matcher_confidence": 0.8,
        "assumptions": [],
    }
    payload.update(overrides)
    return MatchResult.model_validate(payload)


def test_temperature_threshold_estimate_happy_path() -> None:
    estimator = WeatherProbabilityEstimator()
    periods = [
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            temperature=72,
            temperature_unit="F",
        ),
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 8, 0, tzinfo=UTC),
            temperature=68,
            temperature_unit="F",
        ),
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 8, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 9, 0, tzinfo=UTC),
            temperature=74,
            temperature_unit="F",
        ),
    ]
    result = estimator.estimate(_contract(), _match_result(periods))
    assert result.available is True
    assert result.estimate_method == "temperature_threshold_v1"
    assert result.estimated_probability is not None
    assert abs(result.estimated_probability - (2 / 3)) < 1e-6
    assert result.model_confidence > 0.5


def test_unsupported_metric_rejected_safely() -> None:
    estimator = WeatherProbabilityEstimator()
    result = estimator.estimate(
        _contract(weather_dimension="snowfall"),
        _match_result([]),
    )
    assert result.available is False
    assert result.decision_code == "unsupported_metric"


def test_low_confidence_with_assumptions() -> None:
    estimator = WeatherProbabilityEstimator()
    periods = [
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            temperature=71,
            temperature_unit="F",
        ),
    ]
    match = _match_result(periods, assumptions=["ASSUMPTION: weak time alignment"])
    result = estimator.estimate(_contract(), match)
    assert result.available is True
    assert result.model_confidence < 0.75
    assert result.assumptions


def test_missing_data_flags_for_temperature() -> None:
    estimator = WeatherProbabilityEstimator()
    periods = [
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            temperature=None,
            temperature_unit="F",
        )
    ]
    result = estimator.estimate(_contract(), _match_result(periods))
    assert result.available is False
    assert "temperature_values_missing" in result.missing_data_flags


# --- Wind estimation tests ---


class TestWindEstimator:
    """Cover the wind estimation path."""

    def test_wind_happy_path_above_threshold(self) -> None:
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="wind",
            threshold_operator=">",
            threshold_value=10,
            threshold_unit="mph",
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                wind_speed="15 mph",
            ),
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 8, 0, tzinfo=UTC),
                wind_speed="8 mph",
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        assert result.estimate_method == "wind_threshold_v1"
        assert result.estimated_probability == 0.5

    def test_wind_range_speed_uses_max(self) -> None:
        """Wind strings like '8 to 15 mph' should use max (15)."""
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="wind",
            threshold_operator=">",
            threshold_value=12,
            threshold_unit="mph",
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                wind_speed="8 to 15 mph",
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        assert result.estimated_probability == 1.0

    def test_wind_missing_speed_data(self) -> None:
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="wind",
            threshold_operator=">",
            threshold_value=10,
            threshold_unit="mph",
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                wind_speed=None,
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is False
        assert result.decision_code == "wind_data_missing"

    def test_wind_assumption_penalizes_confidence(self) -> None:
        """Wind parsing assumption should be reflected in confidence calc."""
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="wind",
            threshold_operator=">",
            threshold_value=10,
            threshold_unit="mph",
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                wind_speed="15 mph",
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        assert any("wind string" in a.lower() for a in result.assumptions)
        # The assumption count should reduce confidence
        assert result.model_confidence < 0.7

    def test_wind_no_threshold_rejects(self) -> None:
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="wind",
            threshold_operator=None,
            threshold_value=None,
            threshold_unit="mph",
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                wind_speed="15 mph",
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is False
        assert result.decision_code == "threshold_not_parseable"


# --- Precipitation estimation tests ---


class TestPrecipEstimator:
    """Cover the precipitation estimation path."""

    def test_precip_pop_proxy_happy_path(self) -> None:
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="precipitation",
            threshold_operator=None,
            threshold_value=None,
            threshold_unit=None,
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                probability_of_precipitation=50,
            ),
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 8, 0, tzinfo=UTC),
                probability_of_precipitation=30,
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        assert result.estimate_method == "precip_occurrence_proxy_v1"
        # p_any = 1 - (1 - 0.5) * (1 - 0.3) = 1 - 0.35 = 0.65
        assert result.estimated_probability is not None
        assert abs(result.estimated_probability - 0.65) < 0.01

    def test_precip_amount_not_supported(self) -> None:
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="precipitation",
            threshold_operator=">",
            threshold_value=2,
            threshold_unit="in",
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                probability_of_precipitation=60,
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is False
        assert result.decision_code == "precip_amount_not_supported"

    def test_precip_missing_pop_data(self) -> None:
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="precipitation",
            threshold_operator=None,
            threshold_value=None,
            threshold_unit=None,
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                probability_of_precipitation=None,
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is False
        assert result.decision_code == "precip_probability_missing"

    def test_precip_threshold_percent_interpretation(self) -> None:
        """When threshold_value is >1 and <=100, interpret as PoP percentage."""
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="precipitation",
            threshold_operator=">=",
            threshold_value=40,
            threshold_unit=None,
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                probability_of_precipitation=50,
            ),
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 8, 0, tzinfo=UTC),
                probability_of_precipitation=20,
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        # Period 1: 50% >= 40% → met. Period 2: 20% < 40% → not met. → 1/2 = 0.5
        assert result.estimated_probability == 0.5

    def test_precip_pop_clamped_to_valid_range(self) -> None:
        """PoP values above 100 should be clamped to 1.0 probability."""
        estimator = WeatherProbabilityEstimator()
        contract = _contract(
            weather_dimension="precipitation",
            threshold_operator=None,
            threshold_value=None,
            threshold_unit=None,
        )
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                probability_of_precipitation=100,
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        assert result.estimated_probability == 1.0


# --- Temperature unit conversion ---


class TestTemperatureConversion:
    """Cover temperature unit conversion edge cases."""

    def test_celsius_to_fahrenheit_threshold(self) -> None:
        """Periods in Celsius should be converted to contract's Fahrenheit."""
        estimator = WeatherProbabilityEstimator()
        # 25°C = 77°F > 70°F threshold
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                temperature=25,
                temperature_unit="C",
            ),
        ]
        result = estimator.estimate(_contract(), _match_result(periods))
        assert result.available is True
        assert result.estimated_probability == 1.0

    def test_fahrenheit_to_celsius_conversion(self) -> None:
        """Contract in Celsius should convert Fahrenheit forecast data."""
        estimator = WeatherProbabilityEstimator()
        contract = _contract(threshold_value=20, threshold_unit="C")
        # 60°F = 15.55°C < 20°C threshold
        periods = [
            WeatherForecastPeriod(
                start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
                end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
                temperature=60,
                temperature_unit="F",
            ),
        ]
        result = estimator.estimate(contract, _match_result(periods))
        assert result.available is True
        assert result.estimated_probability == 0.0


# --- Match failure propagation ---


def test_match_failed_returns_unavailable_estimate() -> None:
    estimator = WeatherProbabilityEstimator()
    failed = MatchResult.model_validate(
        {
            "matched": False,
            "decision_code": "location_mismatch",
            "matcher_confidence": 0.0,
        }
    )
    result = estimator.estimate(_contract(), failed)
    assert result.available is False
    assert result.decision_code == "match_failed"


# --- Between/range threshold operators ---


def test_temperature_between_operator() -> None:
    """Between operator should accept values in the range."""
    estimator = WeatherProbabilityEstimator()
    contract = _contract(
        threshold_operator="between",
        threshold_value=None,
        threshold_low=65,
        threshold_high=75,
        threshold_unit="F",
    )
    periods = [
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            temperature=70,
            temperature_unit="F",
        ),
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 8, 0, tzinfo=UTC),
            temperature=80,
            temperature_unit="F",
        ),
    ]
    result = estimator.estimate(contract, _match_result(periods))
    assert result.available is True
    assert result.estimated_probability == 0.5


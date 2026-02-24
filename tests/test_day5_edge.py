"""Day 5 edge math tests."""

from __future__ import annotations

from types import SimpleNamespace

from kalshi_weather_bot.signal.edge import EdgeCalculator
from kalshi_weather_bot.signal.models import EstimateResult
from kalshi_weather_bot.signal.pricing import (
    dollars_to_cents,
    parse_cents_field,
    parse_dollars_field,
    parse_price_field,
    parse_probability_field,
)


def _settings(**overrides: object) -> SimpleNamespace:
    payload = {"edge_spread_buffer": 0.01, "edge_fee_buffer": 0.0}
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _estimate(probability: float) -> EstimateResult:
    return EstimateResult(
        available=True,
        decision_code="estimated",
        estimated_probability=probability,
        model_confidence=0.8,
        estimate_method="test",
    )


def test_cents_probability_conversion_and_edge_sign() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"yes_ask": 40, "no_ask": 60}, _estimate(0.55))
    assert result.valid is True
    assert result.market_implied_probability == 0.4
    assert result.edge_yes is not None and result.edge_yes > 0
    assert result.recommended_side == "yes"


def test_recommended_no_side_when_model_below_market_yes() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"yes_ask": 70, "no_ask": 30}, _estimate(0.25))
    assert result.valid is True
    assert result.recommended_side == "no"
    assert result.edge_no is not None and result.edge_no > 0


def test_buffer_application_can_block_candidate() -> None:
    edge = EdgeCalculator(settings=_settings(edge_spread_buffer=0.1, edge_fee_buffer=0.05))
    result = edge.compute({"yes_ask": 45, "no_ask": 55}, _estimate(0.58))
    assert result.valid is True
    assert result.recommended_side is None
    assert "insufficient_edge_after_buffers" in result.reasons


def test_malformed_price_rejected() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"yes_ask": "bad-value"}, _estimate(0.55))
    assert result.valid is False
    assert result.decision_code == "missing_market_price"


# --- _to_cents boundary tests ---


class TestToCentsBoundary:
    """Cover field-aware cents/dollars conversion edge cases."""

    def test_cents_field_value_1_is_1_cent(self) -> None:
        assert parse_cents_field(1) == 1

    def test_cents_field_string_1_0000_is_1_cent(self) -> None:
        assert parse_cents_field("1.0000") == 1

    def test_cents_field_fractional_0_50_rejected(self) -> None:
        assert parse_cents_field("0.50") is None

    def test_cents_field_value_100_is_100_cents(self) -> None:
        assert parse_cents_field(100) == 100

    def test_cents_field_malformed_string_rejected(self) -> None:
        assert parse_cents_field("abc") is None

    def test_dollars_field_string_1_0000_is_100_cents(self) -> None:
        assert parse_dollars_field("1.0000") == 100

    def test_dollars_field_0_50_is_50_cents(self) -> None:
        assert parse_dollars_field("0.50") == 50

    def test_dollars_field_integer_1_is_100_cents(self) -> None:
        assert parse_dollars_field(1) == 100

    def test_dollars_field_100_is_invalid(self) -> None:
        assert parse_dollars_field(100) is None

    def test_dollars_field_malformed_string_rejected(self) -> None:
        assert parse_dollars_field("bad-value") is None

    def test_parse_price_field_uses_cents_semantics(self) -> None:
        assert parse_price_field("yes_ask", "1.0000") == 1

    def test_parse_price_field_uses_dollars_semantics(self) -> None:
        assert parse_price_field("yes_ask_dollars", "1.0000") == 100

    def test_dollars_to_cents_helper(self) -> None:
        assert dollars_to_cents(0.42) == 42

    def test_probability_parser_explicit_units(self) -> None:
        assert parse_probability_field(0.42, unit="fraction") == 0.42
        assert parse_probability_field(42, unit="percent") == 0.42


# --- Edge estimate unavailable ---


def test_estimate_unavailable_returns_invalid() -> None:
    edge = EdgeCalculator(settings=_settings())
    unavailable = EstimateResult(
        available=False,
        decision_code="match_failed",
        estimated_probability=None,
        model_confidence=0.0,
        estimate_method="none",
    )
    result = edge.compute({"yes_ask": 40, "no_ask": 60}, unavailable)
    assert result.valid is False
    assert result.decision_code == "estimate_unavailable"


# --- Edge derive yes from no and vice versa ---


def test_yes_price_derived_from_no_price() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"no_ask": 60}, _estimate(0.55))
    assert result.valid is True
    assert result.yes_price_cents == 40
    assert "yes_price_derived_from_no_price" in result.warnings


def test_no_price_derived_from_yes_price() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"yes_ask": 40}, _estimate(0.55))
    assert result.valid is True
    assert result.no_price_cents == 60
    assert "no_price_derived_from_yes_price" in result.warnings


# --- Edge non-actionable prices ---


def test_yes_price_at_zero_is_non_actionable() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"yes_ask": 0, "no_ask": 100}, _estimate(0.5))
    assert result.valid is False
    assert "non_actionable_yes_price" in result.reasons


def test_yes_price_at_100_is_non_actionable() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"yes_ask": 100, "no_ask": 0}, _estimate(0.5))
    assert result.valid is False
    assert "non_actionable_yes_price" in result.reasons


# --- Dollar-format prices ---


def test_dollar_format_prices_below_one() -> None:
    """Dollar-denominated fields should convert to cents."""
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute(
        {"yes_ask_dollars": "0.40", "no_ask_dollars": "0.60"},
        _estimate(0.55),
    )
    assert result.valid is True
    assert result.yes_price_cents == 40
    assert result.no_price_cents == 60


def test_last_price_field_used_as_yes_fallback() -> None:
    edge = EdgeCalculator(settings=_settings())
    result = edge.compute({"last_price": 50}, _estimate(0.70))
    assert result.valid is True
    assert result.yes_price_cents == 50

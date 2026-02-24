"""Day 5 candidate selector tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from kalshi_weather_bot.contracts.models import ParsedWeatherContract
from kalshi_weather_bot.signal.models import (
    EdgeResult,
    EstimateResult,
    MatchResult,
    SignalEvaluation,
)
from kalshi_weather_bot.signal.selector import CandidateSelector
from kalshi_weather_bot.weather.models import WeatherForecastPeriod


def _settings(**overrides: object) -> SimpleNamespace:
    payload = {
        "signal_enabled": True,
        "signal_min_parser_confidence": 0.75,
        "signal_min_matcher_confidence": 0.6,
        "signal_min_model_confidence": 0.6,
        "signal_min_edge": 0.03,
        "signal_max_candidates": 3,
        "signal_staleness_override_seconds": None,
        "risk_max_weather_age_seconds": 1800,
        "signal_allow_unsupported": False,
        "risk_min_liquidity_contracts": None,
        "risk_max_spread_cents": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _parsed_contract(
    *,
    market_id: str,
    parse_confidence: float = 0.9,
    parse_status: str = "parsed",
) -> ParsedWeatherContract:
    return ParsedWeatherContract.model_validate(
        {
            "provider_market_id": market_id,
            "ticker": market_id,
            "raw_title": f"Will high temperature in {market_id} be above 70F?",
            "weather_candidate": True,
            "weather_dimension": "temperature",
            "metric_subtype": "high_temp",
            "threshold_operator": ">",
            "threshold_value": 70,
            "threshold_unit": "F",
            "parse_confidence": parse_confidence,
            "parse_status": parse_status,
        }
    )


def _match_result(
    *,
    matched: bool = True,
    matcher_confidence: float = 0.8,
) -> MatchResult:
    periods = [
        WeatherForecastPeriod(
            start_time=datetime(2026, 2, 24, 6, 0, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 7, 0, tzinfo=UTC),
            temperature=72,
            temperature_unit="F",
        )
    ]
    return MatchResult.model_validate(
        {
            "matched": matched,
            "decision_code": "matched" if matched else "no_match",
            "selected_period_indices": [0] if matched else [],
            "selected_period_count": 1 if matched else 0,
            "selected_periods": periods if matched else [],
            "matcher_confidence": matcher_confidence,
            "assumptions": [],
        }
    )


def _estimate_result(
    *,
    available: bool = True,
    confidence: float = 0.8,
    probability: float = 0.7,
) -> EstimateResult:
    return EstimateResult.model_validate(
        {
            "available": available,
            "decision_code": "estimated" if available else "estimate_unavailable",
            "estimated_probability": probability if available else None,
            "model_confidence": confidence,
            "estimate_method": "temperature_threshold_v1",
            "assumptions": [],
            "missing_data_flags": [],
            "supporting_values": {},
            "reasons": [],
            "warnings": [],
        }
    )


def _edge_result(
    *,
    valid: bool = True,
    edge_after: float = 0.08,
    side: str | None = "yes",
    price: int | None = 40,
) -> EdgeResult:
    return EdgeResult.model_validate(
        {
            "valid": valid,
            "decision_code": "edge_computed" if valid else "missing_market_price",
            "reasons": [] if valid else ["missing_market_price"],
            "warnings": [],
            "yes_price_cents": 40,
            "no_price_cents": 60,
            "market_price_field_yes": "yes_ask",
            "market_price_field_no": "no_ask",
            "market_implied_probability": 0.4,
            "model_probability": 0.7,
            "edge_yes": 0.3,
            "edge_no": -0.3,
            "recommended_side": side,
            "recommended_price_cents": price,
            "edge_after_buffers": edge_after,
            "spread_buffer": 0.01,
            "fee_buffer": 0.0,
        }
    )


def _evaluation(
    *,
    market_id: str,
    parse_confidence: float = 0.9,
    matcher_confidence: float = 0.8,
    model_confidence: float = 0.8,
    weather_age_seconds: float = 60.0,
    edge_after: float = 0.08,
    recommended_side: str | None = "yes",
    market_raw: dict[str, Any] | None = None,
    parse_status: str = "parsed",
) -> SignalEvaluation:
    return SignalEvaluation.model_validate(
        {
            "market_id": market_id,
            "ticker": market_id,
            "title": f"Market {market_id}",
            "market_raw": market_raw or {},
            "parsed_contract": _parsed_contract(
                market_id=market_id,
                parse_confidence=parse_confidence,
                parse_status=parse_status,
            ).model_dump(mode="json"),
            "weather_snapshot_ref": "weather-fixture",
            "weather_snapshot_retrieved_at": datetime(2026, 2, 24, 12, 0, tzinfo=UTC),
            "weather_age_seconds": weather_age_seconds,
            "match_result": _match_result(
                matched=True,
                matcher_confidence=matcher_confidence,
            ).model_dump(mode="json"),
            "estimate_result": _estimate_result(
                available=True,
                confidence=model_confidence,
                probability=0.7,
            ).model_dump(mode="json"),
            "edge_result": _edge_result(
                valid=True,
                edge_after=edge_after,
                side=recommended_side,
                price=40 if recommended_side == "yes" else 60,
            ).model_dump(mode="json"),
        }
    )


def test_selector_filters_and_ranks_candidates() -> None:
    selector = CandidateSelector(settings=_settings())
    evals = [
        _evaluation(market_id="MKT-A", edge_after=0.10),
        _evaluation(market_id="MKT-B", edge_after=0.06),
        _evaluation(market_id="MKT-C", parse_confidence=0.5),
    ]
    result = selector.select(evals)

    assert len(result.selected) == 2
    assert len(result.rejected) == 1
    assert result.selected[0].market_id == "MKT-A"
    assert result.selected[1].market_id == "MKT-B"
    assert result.rejected[0].reason_code == "parser_confidence_below_threshold"
    assert result.selected[0].explainability["selected_reason"] == "all_signal_gates_passed"
    assert "key_values" in result.selected[0].explainability


def test_selector_rejects_stale_weather() -> None:
    selector = CandidateSelector(settings=_settings())
    result = selector.select([_evaluation(market_id="MKT-STALE", weather_age_seconds=5000)])
    assert len(result.selected) == 0
    assert len(result.rejected) == 1
    assert result.rejected[0].stage == "freshness"
    assert "weather_snapshot_too_stale" in result.rejected[0].reasons


def test_selector_rejects_missing_recommended_side() -> None:
    selector = CandidateSelector(settings=_settings())
    result = selector.select(
        [_evaluation(market_id="MKT-NONE", edge_after=0.0, recommended_side=None)]
    )
    assert len(result.selected) == 0
    assert len(result.rejected) == 1
    assert result.rejected[0].stage == "edge"
    assert "no_recommended_side" in result.rejected[0].reasons


def test_selector_rejects_market_quality_when_checks_enabled() -> None:
    selector = CandidateSelector(
        settings=_settings(
            risk_min_liquidity_contracts=100,
            risk_max_spread_cents=2,
        )
    )
    eval_item = _evaluation(
        market_id="MKT-QUALITY",
        market_raw={"yes_ask_dollars": "0.60", "yes_bid_dollars": "0.40"},
    )
    result = selector.select([eval_item])
    assert len(result.selected) == 0
    assert len(result.rejected) == 1
    assert result.rejected[0].stage == "market_quality"
    assert "missing_market_liquidity" in result.rejected[0].reasons
    assert "market_spread_above_limit" in result.rejected[0].reasons


# --- Signal disabled gate ---


def test_selector_rejects_all_when_signal_disabled() -> None:
    selector = CandidateSelector(settings=_settings(signal_enabled=False))
    result = selector.select([_evaluation(market_id="MKT-DIS")])
    assert len(result.selected) == 0
    assert len(result.rejected) == 1
    assert result.rejected[0].reason_code == "signal_disabled"


# --- Candidate cap ---


def test_selector_caps_candidates_to_max() -> None:
    selector = CandidateSelector(settings=_settings(signal_max_candidates=1))
    evals = [
        _evaluation(market_id="MKT-1", edge_after=0.10),
        _evaluation(market_id="MKT-2", edge_after=0.09),
    ]
    result = selector.select(evals)
    assert len(result.selected) == 1
    assert result.selected[0].market_id == "MKT-1"


# --- Edge below threshold ---


def test_selector_rejects_edge_below_min() -> None:
    selector = CandidateSelector(settings=_settings(signal_min_edge=0.10))
    result = selector.select([_evaluation(market_id="MKT-LOW-EDGE", edge_after=0.05)])
    assert len(result.selected) == 0
    assert "edge_below_threshold" in result.rejected[0].reasons


# --- Matcher confidence below threshold ---


def test_selector_rejects_low_matcher_confidence() -> None:
    selector = CandidateSelector(settings=_settings(signal_min_matcher_confidence=0.9))
    result = selector.select(
        [_evaluation(market_id="MKT-LOW-MATCH", matcher_confidence=0.5)]
    )
    assert len(result.selected) == 0
    assert "matcher_confidence_below_threshold" in result.rejected[0].reasons


# --- Model confidence below threshold ---


def test_selector_rejects_low_model_confidence() -> None:
    selector = CandidateSelector(settings=_settings(signal_min_model_confidence=0.9))
    result = selector.select(
        [_evaluation(market_id="MKT-LOW-MODEL", model_confidence=0.5)]
    )
    assert len(result.selected) == 0
    assert "model_confidence_below_threshold" in result.rejected[0].reasons


# --- Staleness override ---


def test_staleness_override_supersedes_risk_max() -> None:
    selector = CandidateSelector(
        settings=_settings(
            risk_max_weather_age_seconds=60,
            signal_staleness_override_seconds=6000,
        )
    )
    result = selector.select(
        [_evaluation(market_id="MKT-OVER", weather_age_seconds=3000)]
    )
    assert len(result.selected) == 1


# --- Score ordering ---


def test_selector_scores_ordered_by_edge_and_confidence() -> None:
    selector = CandidateSelector(settings=_settings())
    evals = [
        _evaluation(market_id="MKT-HIGH", edge_after=0.15, model_confidence=0.9),
        _evaluation(market_id="MKT-MED", edge_after=0.10, model_confidence=0.7),
        _evaluation(market_id="MKT-LOW", edge_after=0.05, model_confidence=0.65),
    ]
    result = selector.select(evals)
    assert len(result.selected) == 3
    assert result.selected[0].market_id == "MKT-HIGH"
    assert result.selected[1].market_id == "MKT-MED"
    assert result.selected[2].market_id == "MKT-LOW"
    assert result.selected[0].score > result.selected[1].score > result.selected[2].score


# --- Counts dict ---


def test_selection_result_counts_accurate() -> None:
    selector = CandidateSelector(settings=_settings())
    evals = [
        _evaluation(market_id="MKT-A", edge_after=0.10),
        _evaluation(market_id="MKT-B", parse_confidence=0.5),
    ]
    result = selector.select(evals)
    assert result.counts["total_evaluated"] == 2
    assert result.counts["selected"] == 1
    assert result.counts["rejected"] == 1


# --- Parse status unsupported gate ---


def test_selector_rejects_unparsed_status_by_default() -> None:
    selector = CandidateSelector(settings=_settings(signal_allow_unsupported=False))
    result = selector.select(
        [_evaluation(market_id="MKT-UNPARSED", parse_status="unsupported")]
    )
    assert len(result.selected) == 0
    assert "parse_status_not_supported" in result.rejected[0].reasons


def test_selector_allows_unparsed_status_when_override_enabled() -> None:
    selector = CandidateSelector(settings=_settings(signal_allow_unsupported=True))
    result = selector.select(
        [_evaluation(market_id="MKT-UNPARSED", parse_status="unsupported")]
    )
    # Should not be rejected for parse_status (may still pass other gates)
    for rejection in result.rejected:
        assert "parse_status_not_supported" not in rejection.reasons


# --- min_edge_override parameter ---


def test_select_min_edge_override() -> None:
    selector = CandidateSelector(settings=_settings(signal_min_edge=0.03))
    # Without override, edge_after=0.05 passes min_edge=0.03
    result = selector.select(
        [_evaluation(market_id="MKT-EDGE-OVER", edge_after=0.05)],
        min_edge_override=0.10,
    )
    assert len(result.selected) == 0
    assert "edge_below_threshold" in result.rejected[0].reasons


# --- Liquidity passes when present ---


def test_selector_passes_liquidity_when_sufficient() -> None:
    selector = CandidateSelector(
        settings=_settings(risk_min_liquidity_contracts=100)
    )
    result = selector.select(
        [_evaluation(market_id="MKT-LIQ", market_raw={"liquidity": 500})]
    )
    assert len(result.selected) == 1

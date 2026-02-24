"""Day 5 candidate filtering and ranking pipeline."""

from __future__ import annotations

import logging
from typing import Any

from ..config import Settings
from .models import SignalCandidate, SignalEvaluation, SignalRejection, SignalSelectionResult
from .pricing import parse_cents_field, parse_dollars_field


class CandidateSelector:
    """Apply signal gates and rank surviving trade candidates."""

    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("kalshi_weather_bot.signal.selector")

    def select(
        self,
        evaluations: list[SignalEvaluation],
        *,
        max_candidates: int | None = None,
        min_edge_override: float | None = None,
        min_model_confidence_override: float | None = None,
    ) -> SignalSelectionResult:
        """Filter + rank signal candidates with explainable rejection records."""
        selected: list[SignalCandidate] = []
        rejected: list[SignalRejection] = []

        if not self.settings.signal_enabled:
            for evaluation in evaluations:
                rejected.append(
                    self._rejection(
                        evaluation=evaluation,
                        stage="signal",
                        reason_code="signal_disabled",
                        reasons=["signal_disabled"],
                    )
                )
            return self._result(selected, rejected, len(evaluations))

        min_parser_conf = self.settings.signal_min_parser_confidence
        min_match_conf = self.settings.signal_min_matcher_confidence
        min_model_conf = (
            min_model_confidence_override
            if min_model_confidence_override is not None
            else self.settings.signal_min_model_confidence
        )
        min_edge = (
            min_edge_override
            if min_edge_override is not None
            else self.settings.signal_min_edge
        )
        staleness_limit = (
            self.settings.signal_staleness_override_seconds
            if self.settings.signal_staleness_override_seconds is not None
            else self.settings.risk_max_weather_age_seconds
        )

        for evaluation in evaluations:
            parsed = evaluation.parsed_contract
            match = evaluation.match_result
            estimate = evaluation.estimate_result
            edge = evaluation.edge_result
            reasons: list[str] = []
            stage = "signal"

            if parsed.parse_status != "parsed" and not self.settings.signal_allow_unsupported:
                stage = "parser"
                reasons.append("parse_status_not_supported")
            if parsed.parse_confidence < min_parser_conf:
                stage = "parser"
                reasons.append("parser_confidence_below_threshold")
            if not match.matched:
                stage = "matcher"
                reasons.append(match.decision_code or "matcher_failed")
            if match.matcher_confidence < min_match_conf:
                stage = "matcher"
                reasons.append("matcher_confidence_below_threshold")
            if not estimate.available:
                stage = "estimator"
                reasons.append(estimate.decision_code or "estimate_unavailable")
            if estimate.model_confidence < min_model_conf:
                stage = "estimator"
                reasons.append("model_confidence_below_threshold")
            if evaluation.weather_age_seconds > staleness_limit:
                stage = "freshness"
                reasons.append("weather_snapshot_too_stale")
            if not edge.valid:
                stage = "edge"
                reasons.extend(edge.reasons or [edge.decision_code])
            if edge.recommended_side is None:
                stage = "edge"
                reasons.append("no_recommended_side")
            if edge.edge_after_buffers is None or edge.edge_after_buffers < min_edge:
                stage = "edge"
                reasons.append("edge_below_threshold")

            if self.settings.risk_min_liquidity_contracts is not None:
                liquidity = _to_int(evaluation.market_raw.get("liquidity"))
                if liquidity is None:
                    stage = "market_quality"
                    reasons.append("missing_market_liquidity")
                elif liquidity < self.settings.risk_min_liquidity_contracts:
                    stage = "market_quality"
                    reasons.append("insufficient_market_liquidity")

            if self.settings.risk_max_spread_cents is not None:
                spread = _market_spread_cents(evaluation.market_raw)
                if spread is None:
                    stage = "market_quality"
                    reasons.append("missing_market_spread")
                elif spread > self.settings.risk_max_spread_cents:
                    stage = "market_quality"
                    reasons.append("market_spread_above_limit")

            if reasons:
                rejected.append(
                    self._rejection(
                        evaluation=evaluation,
                        stage=stage,
                        reason_code=reasons[0],
                        reasons=sorted(set(reasons)),
                    )
                )
                continue

            score = self._score_candidate(
                edge_after_buffers=edge.edge_after_buffers or 0.0,
                matcher_confidence=match.matcher_confidence,
                model_confidence=estimate.model_confidence,
            )
            if edge.recommended_price_cents is None or edge.recommended_side is None:
                rejected.append(
                    self._rejection(
                        evaluation=evaluation,
                        stage="edge",
                        reason_code="missing_recommended_trade_terms",
                        reasons=["missing_recommended_trade_terms"],
                    )
                )
                continue

            selected.append(
                SignalCandidate(
                    market_id=evaluation.market_id,
                    ticker=evaluation.ticker,
                    title=evaluation.title,
                    side=edge.recommended_side,
                    price_cents=edge.recommended_price_cents,
                    quantity=1,
                    score=score,
                    parsed_contract=parsed,
                    match_result=match,
                    estimate_result=estimate,
                    edge_result=edge,
                    explainability={
                        "selected_reason": "all_signal_gates_passed",
                        "key_values": {
                            "parser_confidence": parsed.parse_confidence,
                            "matcher_confidence": match.matcher_confidence,
                            "model_confidence": estimate.model_confidence,
                            "market_implied_probability": edge.market_implied_probability,
                            "model_probability": edge.model_probability,
                            "edge_yes": edge.edge_yes,
                            "edge_no": edge.edge_no,
                            "edge_after_buffers": edge.edge_after_buffers,
                            "weather_age_seconds": evaluation.weather_age_seconds,
                        },
                        "assumptions": estimate.assumptions,
                        "selected_period_count": match.selected_period_count,
                    },
                )
            )

        selected_sorted = sorted(selected, key=lambda candidate: candidate.score, reverse=True)
        candidate_cap = max_candidates or self.settings.signal_max_candidates
        selected_capped = selected_sorted[:candidate_cap]
        return self._result(selected_capped, rejected, len(evaluations))

    def _score_candidate(
        self,
        *,
        edge_after_buffers: float,
        matcher_confidence: float,
        model_confidence: float,
    ) -> float:
        confidence_weight = 0.4 + 0.3 * matcher_confidence + 0.3 * model_confidence
        return round(edge_after_buffers * confidence_weight, 8)

    def _rejection(
        self,
        *,
        evaluation: SignalEvaluation,
        stage: str,
        reason_code: str,
        reasons: list[str],
    ) -> SignalRejection:
        return SignalRejection(
            market_id=evaluation.market_id,
            ticker=evaluation.ticker,
            title=evaluation.title,
            stage=stage,
            reason_code=reason_code,
            reasons=reasons,
            details={
                "parser_confidence": evaluation.parsed_contract.parse_confidence,
                "matcher_confidence": evaluation.match_result.matcher_confidence,
                "model_confidence": evaluation.estimate_result.model_confidence,
                "weather_age_seconds": evaluation.weather_age_seconds,
                "edge_after_buffers": evaluation.edge_result.edge_after_buffers,
                "recommended_side": evaluation.edge_result.recommended_side,
            },
        )

    @staticmethod
    def _result(
        selected: list[SignalCandidate],
        rejected: list[SignalRejection],
        total_evaluated: int,
    ) -> SignalSelectionResult:
        counts = {
            "total_evaluated": total_evaluated,
            "selected": len(selected),
            "rejected": len(rejected),
        }
        return SignalSelectionResult(selected=selected, rejected=rejected, counts=counts)


def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _market_spread_cents(market: dict[str, Any]) -> int | None:
    ask = parse_cents_field(market.get("yes_ask"))
    bid = parse_cents_field(market.get("yes_bid"))
    if ask is None:
        ask = parse_dollars_field(market.get("yes_ask_dollars"))
    if bid is None:
        bid = parse_dollars_field(market.get("yes_bid_dollars"))
    if ask is None or bid is None:
        return None
    return max(0, ask - bid)

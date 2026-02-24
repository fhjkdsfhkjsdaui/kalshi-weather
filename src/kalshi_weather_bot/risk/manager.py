"""Day 4 production-style risk manager for dry-run execution flows."""

from __future__ import annotations

import logging

from ..config import Settings
from ..execution.models import TradeIntent
from .models import IntentFingerprint, RiskContext, RiskDecision, RiskMetricsSnapshot


class RiskManager:
    """Evaluate trade intents against explicit safety/risk rules."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def evaluate_intent(self, intent: TradeIntent, context: RiskContext) -> RiskDecision:
        """Run all configured risk checks and return a structured decision."""
        reasons: list[str] = []
        warnings: list[str] = []

        if self.settings.risk_kill_switch or context.kill_switch_active:
            reasons.append("kill_switch_active")

        weather_age_seconds = (context.now - intent.weather_snapshot_retrieved_at).total_seconds()
        if weather_age_seconds < 0:
            warnings.append("weather_snapshot_timestamp_in_future")
            weather_age_seconds = 0.0

        intent_notional_cents = intent.stake_cents
        current_market_exposure = context.exposure_by_market_cents.get(intent.market_id, 0)
        projected_total_exposure = context.total_open_exposure_cents + intent_notional_cents
        projected_market_exposure = current_market_exposure + intent_notional_cents
        projected_open_orders = context.open_order_count + 1
        duplicate_detected = self._detect_duplicate_intent(intent=intent, context=context)

        if not self.settings.risk_enabled:
            warnings.append("risk_checks_disabled")
            metrics = RiskMetricsSnapshot(
                intent_notional_cents=intent_notional_cents,
                parser_confidence=intent.parser_confidence,
                min_parser_confidence=self.settings.risk_min_parser_confidence,
                weather_age_seconds=weather_age_seconds,
                max_weather_age_seconds=self.settings.risk_max_weather_age_seconds,
                price_cents=intent.price_cents,
                quantity=intent.quantity,
                total_open_exposure_cents=context.total_open_exposure_cents,
                projected_total_exposure_cents=projected_total_exposure,
                current_market_exposure_cents=current_market_exposure,
                projected_market_exposure_cents=projected_market_exposure,
                max_stake_per_trade_cents=self.settings.risk_max_stake_per_trade_cents,
                max_total_exposure_cents=self.settings.risk_max_total_exposure_cents,
                max_exposure_per_market_cents=self.settings.risk_max_exposure_per_market_cents,
                open_order_count=context.open_order_count,
                projected_open_order_count=projected_open_orders,
                max_concurrent_open_orders=self.settings.risk_max_concurrent_open_orders,
                duplicate_intent_cooldown_seconds=self.settings.risk_duplicate_intent_cooldown_seconds,
                duplicate_intent_detected=duplicate_detected,
                market_liquidity=intent.market_liquidity,
                min_required_liquidity=self.settings.risk_min_liquidity_contracts,
                market_spread_cents=intent.market_spread_cents,
                max_allowed_spread_cents=self.settings.risk_max_spread_cents,
                kill_switch_active=self.settings.risk_kill_switch or context.kill_switch_active,
            )
            return RiskDecision(
                allowed=not reasons,
                decision_code="risk_disabled" if not reasons else reasons[0],
                reasons=reasons,
                warnings=warnings,
                computed_metrics=metrics,
            )

        if intent.parser_confidence < self.settings.risk_min_parser_confidence:
            reasons.append("parser_confidence_below_min")
        if weather_age_seconds > self.settings.risk_max_weather_age_seconds:
            reasons.append("stale_weather_snapshot")
        if intent.price_cents < self.settings.risk_price_min_cents:
            reasons.append("price_below_min_bound")
        if intent.price_cents > self.settings.risk_price_max_cents:
            reasons.append("price_above_max_bound")
        if intent_notional_cents > self.settings.risk_max_stake_per_trade_cents:
            reasons.append("max_stake_per_trade_exceeded")
        if projected_total_exposure > self.settings.risk_max_total_exposure_cents:
            reasons.append("max_total_exposure_exceeded")
        if projected_market_exposure > self.settings.risk_max_exposure_per_market_cents:
            reasons.append("max_market_exposure_exceeded")
        if projected_open_orders > self.settings.risk_max_concurrent_open_orders:
            reasons.append("max_concurrent_open_orders_exceeded")
        if duplicate_detected:
            reasons.append("duplicate_intent_cooldown")

        self._evaluate_liquidity_and_spread(intent=intent, reasons=reasons, warnings=warnings)

        metrics = RiskMetricsSnapshot(
            intent_notional_cents=intent_notional_cents,
            parser_confidence=intent.parser_confidence,
            min_parser_confidence=self.settings.risk_min_parser_confidence,
            weather_age_seconds=weather_age_seconds,
            max_weather_age_seconds=self.settings.risk_max_weather_age_seconds,
            price_cents=intent.price_cents,
            quantity=intent.quantity,
            total_open_exposure_cents=context.total_open_exposure_cents,
            projected_total_exposure_cents=projected_total_exposure,
            current_market_exposure_cents=current_market_exposure,
            projected_market_exposure_cents=projected_market_exposure,
            max_stake_per_trade_cents=self.settings.risk_max_stake_per_trade_cents,
            max_total_exposure_cents=self.settings.risk_max_total_exposure_cents,
            max_exposure_per_market_cents=self.settings.risk_max_exposure_per_market_cents,
            open_order_count=context.open_order_count,
            projected_open_order_count=projected_open_orders,
            max_concurrent_open_orders=self.settings.risk_max_concurrent_open_orders,
            duplicate_intent_cooldown_seconds=self.settings.risk_duplicate_intent_cooldown_seconds,
            duplicate_intent_detected=duplicate_detected,
            market_liquidity=intent.market_liquidity,
            min_required_liquidity=self.settings.risk_min_liquidity_contracts,
            market_spread_cents=intent.market_spread_cents,
            max_allowed_spread_cents=self.settings.risk_max_spread_cents,
            kill_switch_active=self.settings.risk_kill_switch or context.kill_switch_active,
        )

        allowed = len(reasons) == 0
        decision_code = "approved" if allowed else reasons[0]
        return RiskDecision(
            allowed=allowed,
            decision_code=decision_code,
            reasons=reasons,
            warnings=warnings,
            computed_metrics=metrics,
        )

    def _detect_duplicate_intent(self, intent: TradeIntent, context: RiskContext) -> bool:
        for recent in context.recent_intents:
            if not self._is_same_signature(intent=intent, recent=recent):
                continue
            age_seconds = (context.now - recent.timestamp).total_seconds()
            if age_seconds <= self.settings.risk_duplicate_intent_cooldown_seconds:
                return True
        return False

    @staticmethod
    def _is_same_signature(intent: TradeIntent, recent: IntentFingerprint) -> bool:
        return (
            intent.market_id == recent.market_id
            and intent.side == recent.side
            and intent.price_cents == recent.price_cents
            and intent.quantity == recent.quantity
        )

    def _evaluate_liquidity_and_spread(
        self, intent: TradeIntent, reasons: list[str], warnings: list[str]
    ) -> None:
        min_liq = self.settings.risk_min_liquidity_contracts
        if min_liq is not None:
            if intent.market_liquidity is None:
                reasons.append("missing_market_liquidity")
            elif intent.market_liquidity < min_liq:
                reasons.append("insufficient_market_liquidity")
        elif intent.market_liquidity is None:
            warnings.append("market_liquidity_unavailable_check_skipped")

        max_spread = self.settings.risk_max_spread_cents
        if max_spread is not None:
            if intent.market_spread_cents is None:
                reasons.append("missing_market_spread")
            elif intent.market_spread_cents > max_spread:
                reasons.append("market_spread_above_limit")
        elif intent.market_spread_cents is None:
            warnings.append("market_spread_unavailable_check_skipped")


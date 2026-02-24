"""Day 7 strict trade policy for guarded live_micro execution."""

from __future__ import annotations

import logging

from ..config import Settings
from .micro_models import (
    MicroPolicyContext,
    MicroTradeCandidate,
    TradePolicyDecision,
    TradePolicyMetrics,
)


class MicroTradePolicy:
    """Apply explicit Day 7 micro-live safety gates before any live order submit."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def evaluate(
        self,
        candidate: MicroTradeCandidate,
        context: MicroPolicyContext,
        *,
        max_trades_per_run: int,
    ) -> TradePolicyDecision:
        """Return structured allow/deny decision for one candidate."""
        reasons: list[str] = []
        warnings: list[str] = []

        notional_cents = candidate.notional_cents
        max_notional_cents = int(
            round(self.settings.micro_max_notional_per_trade_dollars * 100)
        )
        max_daily_gross_cents = int(
            round(self.settings.micro_max_daily_gross_exposure_dollars * 100)
        )
        max_daily_loss_cents = int(round(self.settings.micro_max_daily_realized_loss_dollars * 100))

        if candidate.parser_confidence < self.settings.micro_min_parser_confidence:
            reasons.append("parser_confidence_below_micro_min")
        if candidate.edge_after_buffers < self.settings.micro_min_edge:
            reasons.append("edge_below_micro_min")
        if candidate.weather_age_seconds > self.settings.micro_max_weather_age_seconds:
            reasons.append("weather_staleness_exceeds_micro_max")

        if notional_cents > max_notional_cents:
            reasons.append("micro_max_notional_per_trade_exceeded")
        if context.trades_executed_run >= max_trades_per_run:
            reasons.append("micro_max_trades_per_run_hit")
        if context.trades_executed_today >= self.settings.micro_max_trades_per_day:
            reasons.append("micro_max_trades_per_day_hit")
        if context.open_positions_count >= self.settings.micro_max_open_positions:
            reasons.append("micro_max_open_positions_hit")
        if context.daily_gross_exposure_cents + notional_cents > max_daily_gross_cents:
            reasons.append("micro_max_daily_gross_exposure_hit")
        if context.daily_realized_pnl_cents <= -max_daily_loss_cents:
            reasons.append("micro_max_daily_realized_loss_hit")

        seconds_since_last_trade: float | None = None
        if context.last_trade_ts is not None:
            seconds_since_last_trade = (context.now - context.last_trade_ts).total_seconds()
            if seconds_since_last_trade < 0:
                warnings.append("last_trade_timestamp_in_future")
                seconds_since_last_trade = 0.0
            if seconds_since_last_trade < self.settings.micro_min_seconds_between_trades:
                reasons.append("micro_trade_cooldown_active")

        self._apply_market_quality_checks(candidate=candidate, reasons=reasons, warnings=warnings)
        self._apply_market_scope_check(candidate=candidate, reasons=reasons)

        metrics = TradePolicyMetrics(
            notional_cents=notional_cents,
            max_notional_per_trade_cents=max_notional_cents,
            projected_daily_gross_exposure_cents=(
                context.daily_gross_exposure_cents + notional_cents
            ),
            max_daily_gross_exposure_cents=max_daily_gross_cents,
            daily_realized_pnl_cents=context.daily_realized_pnl_cents,
            max_daily_realized_loss_cents=max_daily_loss_cents,
            parser_confidence=candidate.parser_confidence,
            min_parser_confidence=self.settings.micro_min_parser_confidence,
            edge_after_buffers=candidate.edge_after_buffers,
            min_edge=self.settings.micro_min_edge,
            weather_age_seconds=candidate.weather_age_seconds,
            max_weather_age_seconds=self.settings.micro_max_weather_age_seconds,
            open_positions_count=context.open_positions_count,
            max_open_positions=self.settings.micro_max_open_positions,
            trades_executed_run=context.trades_executed_run,
            max_trades_per_run=max_trades_per_run,
            trades_executed_today=context.trades_executed_today,
            max_trades_per_day=self.settings.micro_max_trades_per_day,
            seconds_since_last_trade=seconds_since_last_trade,
            min_seconds_between_trades=self.settings.micro_min_seconds_between_trades,
        )

        allowed = len(reasons) == 0
        decision_code = "approved" if allowed else reasons[0]
        return TradePolicyDecision(
            allowed=allowed,
            decision_code=decision_code,
            reasons=reasons,
            warnings=warnings,
            computed_metrics=metrics,
        )

    def _apply_market_quality_checks(
        self,
        *,
        candidate: MicroTradeCandidate,
        reasons: list[str],
        warnings: list[str],
    ) -> None:
        min_liquidity = self.settings.risk_min_liquidity_contracts
        if min_liquidity is not None:
            if candidate.market_liquidity is None:
                reasons.append("missing_market_liquidity")
            elif candidate.market_liquidity < min_liquidity:
                reasons.append("insufficient_market_liquidity")
        elif candidate.market_liquidity is None:
            warnings.append("market_liquidity_check_skipped")

        max_spread = self.settings.risk_max_spread_cents
        if max_spread is not None:
            if candidate.market_spread_cents is None:
                reasons.append("missing_market_spread")
            elif candidate.market_spread_cents > max_spread:
                reasons.append("market_spread_above_limit")
        elif candidate.market_spread_cents is None:
            warnings.append("market_spread_check_skipped")

    def _apply_market_scope_check(
        self,
        *,
        candidate: MicroTradeCandidate,
        reasons: list[str],
    ) -> None:
        whitelist = self.settings.micro_market_scope_whitelist
        if whitelist is None:
            return
        allowed_tokens = [token.strip().lower() for token in whitelist.split(",") if token.strip()]
        if not allowed_tokens:
            return

        haystack = " ".join(
            [
                candidate.market_id,
                candidate.ticker or "",
                candidate.title,
                candidate.location_hint or "",
            ]
        ).lower()
        if not any(token in haystack for token in allowed_tokens):
            reasons.append("market_scope_not_allowed")

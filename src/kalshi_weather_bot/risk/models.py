"""Typed risk models used by the Day 4 risk manager."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class IntentFingerprint(BaseModel):
    """Signature used for duplicate intent suppression checks."""

    market_id: str
    side: str
    price_cents: int
    quantity: int
    timestamp: datetime


class RiskContext(BaseModel):
    """Runtime context required to evaluate a trade intent."""

    now: datetime
    total_open_exposure_cents: int = 0
    exposure_by_market_cents: dict[str, int] = Field(default_factory=dict)
    open_order_count: int = 0
    recent_intents: list[IntentFingerprint] = Field(default_factory=list)
    kill_switch_active: bool = False


class RiskMetricsSnapshot(BaseModel):
    """Computed metrics emitted with each risk decision."""

    intent_notional_cents: int
    parser_confidence: float
    min_parser_confidence: float
    weather_age_seconds: float
    max_weather_age_seconds: int
    price_cents: int
    quantity: int
    total_open_exposure_cents: int
    projected_total_exposure_cents: int
    current_market_exposure_cents: int
    projected_market_exposure_cents: int
    max_stake_per_trade_cents: int
    max_total_exposure_cents: int
    max_exposure_per_market_cents: int
    open_order_count: int
    projected_open_order_count: int
    max_concurrent_open_orders: int
    duplicate_intent_cooldown_seconds: int
    duplicate_intent_detected: bool
    market_liquidity: int | None = None
    min_required_liquidity: int | None = None
    market_spread_cents: int | None = None
    max_allowed_spread_cents: int | None = None
    kill_switch_active: bool = False


class RiskDecision(BaseModel):
    """Structured decision output from risk evaluation."""

    allowed: bool
    decision_code: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    computed_metrics: RiskMetricsSnapshot


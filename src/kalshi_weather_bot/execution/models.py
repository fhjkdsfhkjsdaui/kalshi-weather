"""Typed execution models for Day 4 risk-evaluated dry-run order flow."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..risk.models import IntentFingerprint, RiskDecision


class TradeIntent(BaseModel):
    """Intent to place a trade, produced by upstream strategy/plumbing layers."""

    intent_id: str | None = None
    market_id: str
    side: Literal["yes", "no"]
    price_cents: int = Field(ge=1, le=99)
    quantity: int = Field(gt=0)
    parser_confidence: float = Field(ge=0.0, le=1.0)
    parsed_contract_ref: str
    weather_snapshot_ref: str
    weather_snapshot_retrieved_at: datetime
    timestamp: datetime
    strategy_tag: str | None = None
    market_liquidity: int | None = Field(default=None, ge=0)
    market_spread_cents: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weather_snapshot_retrieved_at", "timestamp", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value

    @property
    def stake_cents(self) -> int:
        """Total cents at risk for this intent under a simple stake model."""
        return self.price_cents * self.quantity


class RiskEvaluatedIntent(BaseModel):
    """Trade intent paired with the resulting risk decision."""

    intent: TradeIntent
    risk_decision: RiskDecision


class DryRunOrder(BaseModel):
    """In-memory representation of a submitted dry-run order."""

    order_id: str
    intent_id: str
    market_id: str
    side: Literal["yes", "no"]
    price_cents: int
    quantity: int
    stake_cents: int
    status: Literal["open", "cancelled"]
    created_at: datetime
    cancelled_at: datetime | None = None
    strategy_tag: str | None = None


class ExposureSnapshot(BaseModel):
    """Current dry-run exposure state across markets."""

    total_open_exposure_cents: int
    exposure_by_market_cents: dict[str, int] = Field(default_factory=dict)
    open_order_count: int
    recent_intents: list[IntentFingerprint] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Result of submitting or cancelling a dry-run order."""

    status: Literal["accepted", "rejected", "cancelled", "not_found"]
    decision_code: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    intent: TradeIntent | None = None
    risk_decision: RiskDecision | None = None
    order: DryRunOrder | None = None


"""Typed models for Day 7 live_micro policy, execution, and position tracking."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MicroOrderOutcome = Literal[
    "skipped",
    "filled",
    "partially_filled",
    "canceled",
    "rejected",
    "unresolved",
    "error",
]


class MicroTradeCandidate(BaseModel):
    """Signal candidate mapped into Day 7 micro-live execution terms."""

    market_id: str
    ticker: str | None = None
    title: str
    side: Literal["yes", "no"]
    price_cents: int = Field(ge=1, le=99)
    quantity: int = Field(gt=0)
    parser_confidence: float = Field(ge=0.0, le=1.0)
    edge_after_buffers: float
    weather_age_seconds: float = Field(ge=0.0)
    parsed_contract_ref: str
    weather_snapshot_ref: str
    market_liquidity: int | None = Field(default=None, ge=0)
    market_spread_cents: int | None = Field(default=None, ge=0)
    strategy_tag: str | None = None
    location_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def notional_cents(self) -> int:
        """Nominal stake for one micro candidate order."""
        return self.price_cents * self.quantity


class MicroPolicyContext(BaseModel):
    """Runtime context used to evaluate a Day 7 micro-live candidate."""

    now: datetime
    today: date
    trades_executed_run: int = 0
    trades_executed_today: int = 0
    open_positions_count: int = 0
    daily_gross_exposure_cents: int = 0
    daily_realized_pnl_cents: int = 0
    last_trade_ts: datetime | None = None

    @field_validator("now", "last_trade_ts", mode="before")
    @classmethod
    def ensure_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value


class TradePolicyMetrics(BaseModel):
    """Computed values attached to each trade policy decision."""

    notional_cents: int
    max_notional_per_trade_cents: int
    projected_daily_gross_exposure_cents: int
    max_daily_gross_exposure_cents: int
    daily_realized_pnl_cents: int
    max_daily_realized_loss_cents: int
    parser_confidence: float
    min_parser_confidence: float
    edge_after_buffers: float
    min_edge: float
    weather_age_seconds: float
    max_weather_age_seconds: int
    open_positions_count: int
    max_open_positions: int
    trades_executed_run: int
    max_trades_per_run: int
    trades_executed_today: int
    max_trades_per_day: int
    seconds_since_last_trade: float | None = None
    min_seconds_between_trades: int


class TradePolicyDecision(BaseModel):
    """Allow/deny decision from the Day 7 trade policy gate."""

    allowed: bool
    decision_code: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    computed_metrics: TradePolicyMetrics


class MicroPosition(BaseModel):
    """Minimal fill-confirmed open position record."""

    market_id: str
    side: Literal["yes", "no"]
    quantity: int = Field(gt=0)
    avg_entry_price_cents: float = Field(ge=0.0)
    opened_at: datetime
    updated_at: datetime
    last_order_id: str

    @field_validator("opened_at", "updated_at", mode="before")
    @classmethod
    def ensure_position_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value


class PositionEvent(BaseModel):
    """One position-level event emitted from confirmed fill handling."""

    event_type: Literal["position_opened", "position_updated", "position_closed"]
    market_id: str
    side: Literal["yes", "no"]
    quantity: int
    avg_entry_price_cents: float
    order_id: str
    ts: datetime
    realized_pnl_delta_cents: int = 0
    note: str | None = None

    @field_validator("ts", mode="before")
    @classmethod
    def ensure_event_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value


class PositionUpdateResult(BaseModel):
    """Result of applying one confirmed fill to local position state."""

    events: list[PositionEvent] = Field(default_factory=list)
    realized_pnl_delta_cents: int = 0


class PositionSnapshot(BaseModel):
    """Snapshot of current open positions and daily totals."""

    open_positions: list[MicroPosition] = Field(default_factory=list)
    open_positions_count: int = 0
    daily_gross_exposure_cents: int = 0
    daily_realized_pnl_cents: int = 0
    trades_executed_today: int = 0
    last_trade_ts: datetime | None = None

    @field_validator("last_trade_ts", mode="before")
    @classmethod
    def ensure_snapshot_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value


class MicroOrderAttempt(BaseModel):
    """Result for one candidate considered by Day 7 live_micro runner."""

    candidate: MicroTradeCandidate
    policy_decision: TradePolicyDecision
    attempted: bool = False
    outcome: MicroOrderOutcome
    order_id: str | None = None
    terminal_status: str | None = None
    filled_quantity: int = 0
    reasons: list[str] = Field(default_factory=list)
    polled_count: int = 0
    unresolved: bool = False
    reconciliation_mismatch: bool = False
    halted: bool = False


class MicroSessionSummary(BaseModel):
    """Aggregated counters for one Day 7 supervised session run."""

    cycles_processed: int
    candidates_seen: int
    trades_allowed: int
    trades_skipped: int
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    orders_submitted: int
    fills: int
    partial_fills: int
    cancels: int
    rejects: int
    unresolved: int
    open_positions_count: int
    realized_pnl_cents: int
    daily_gross_exposure_cents: int
    gross_exposure_utilization: float
    realized_loss_utilization: float
    trades_per_run_utilization: float
    trades_per_day_utilization: float
    halted_early: bool
    halt_reason: str | None = None


class MicroSessionResult(BaseModel):
    """Detailed Day 7 run output with attempts, positions, and summary."""

    attempts: list[MicroOrderAttempt] = Field(default_factory=list)
    position_snapshot: PositionSnapshot
    summary: MicroSessionSummary

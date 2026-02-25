"""Typed models for Day 6 cancel-only live execution validation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

LifecycleState = Literal[
    "intent_created",
    "submit_requested",
    "submit_ack",
    "submit_rejected",
    "cancel_requested",
    "cancel_ack",
    "canceled",
    "partially_filled",
    "filled",
    "timeout",
    "reconciliation_mismatch",
    "terminal_error",
]

NormalizedOrderStatus = Literal[
    "pending",
    "open",
    "canceled",
    "rejected",
    "partially_filled",
    "filled",
    "unknown",
]

ErrorCategory = Literal[
    "auth",
    "permission",
    "validation",
    "rate_limit",
    "network",
    "server",
    "unknown",
]


class CancelOnlyOrderIntent(BaseModel):
    """Operator-provided order intent used for Day 6 cancel-only validation."""

    market_id: str
    side: Literal["yes", "no"]
    price_cents: int = Field(ge=1, le=99)
    quantity: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LifecycleEvent(BaseModel):
    """One lifecycle state transition with optional context."""

    state: LifecycleState
    ts: datetime
    note: str | None = None
    raw_status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts", mode="before")
    @classmethod
    def ensure_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value


class LocalOrderRecord(BaseModel):
    """Local state machine record for one submit->cancel lifecycle attempt."""

    attempt_id: str
    market_id: str
    side: Literal["yes", "no"]
    price_cents: int
    quantity: int
    remote_order_id: str | None = None
    current_state: LifecycleState = "intent_created"
    events: list[LifecycleEvent] = Field(default_factory=list)
    remote_status_raw: str | None = None
    remote_status_normalized: NormalizedOrderStatus = "unknown"
    requested_quantity: int | None = None
    filled_quantity: int = 0
    remaining_quantity: int | None = None
    submit_ts: datetime | None = None
    cancel_ts: datetime | None = None
    terminal_ts: datetime | None = None
    error_category: ErrorCategory | None = None
    error_message: str | None = None

    @field_validator("submit_ts", "cancel_ts", "terminal_ts", mode="before")
    @classmethod
    def ensure_optional_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value


class KalshiCreateOrderRequest(BaseModel):
    """Minimal order create request model for cancel-only validation."""

    market_id: str
    side: Literal["yes", "no"]
    action: Literal["buy", "sell"] = "buy"
    price_cents: int = Field(ge=1, le=99)
    quantity: int = Field(gt=0)
    client_order_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build API payload.

        TODO(KALSHI_API): confirm exact order-create field names and side/price semantics.
        """
        payload: dict[str, Any] = {
            "ticker": self.market_id,
            "action": self.action,
            "side": self.side,
            "count": self.quantity,
            "type": "limit",
        }
        if self.side == "yes":
            payload["yes_price"] = self.price_cents
        else:
            payload["no_price"] = self.price_cents
        if self.client_order_id:
            payload["client_order_id"] = self.client_order_id
        return payload


class KalshiOrderStatus(BaseModel):
    """Normalized order status snapshot extracted from Kalshi responses."""

    order_id: str
    status_raw: str | None = None
    normalized_status: NormalizedOrderStatus = "unknown"
    requested_quantity: int | None = None
    filled_quantity: int = 0
    remaining_quantity: int | None = None
    updated_at: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("updated_at", mode="before")
    @classmethod
    def ensure_optional_utc(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return value

    @property
    def has_fill(self) -> bool:
        return self.filled_quantity > 0 or self.normalized_status in {"filled", "partially_filled"}

    @property
    def is_terminal(self) -> bool:
        return self.normalized_status in {"canceled", "rejected", "partially_filled", "filled"}


class ReconciliationResult(BaseModel):
    """Result of comparing local terminal state vs remote final order status."""

    attempt_id: str
    local_terminal_state: LifecycleState
    remote_terminal_state: NormalizedOrderStatus | Literal["unresolved"]
    matched: bool
    mismatch_reason: str | None = None
    unresolved: bool = False
    order_id: str | None = None
    requested_quantity: int | None = None
    filled_quantity: int = 0
    remaining_quantity: int | None = None
    submit_to_cancel_ms: int | None = None
    submit_to_terminal_ms: int | None = None
    polled_count: int = 0


class CancelOnlyAttemptResult(BaseModel):
    """One Day 6 attempt result with lifecycle + reconciliation details."""

    attempt_id: str
    attempt_index: int
    local_record: LocalOrderRecord
    reconciliation: ReconciliationResult | None = None
    halted_early: bool = False
    halt_reason: str | None = None


class CancelOnlyBatchSummary(BaseModel):
    """Aggregated summary for a cancel-only validation run."""

    attempts_requested: int
    attempts_executed: int
    submit_ack_count: int
    cancel_success_count: int
    rejected_count: int
    unresolved_count: int
    reconciliation_mismatch_count: int
    unexpected_fill_count: int
    halted_early: bool
    halt_reason: str | None = None


class CancelOnlyBatchResult(BaseModel):
    """Batch result containing per-attempt records and summary."""

    attempts: list[CancelOnlyAttemptResult] = Field(default_factory=list)
    summary: CancelOnlyBatchSummary

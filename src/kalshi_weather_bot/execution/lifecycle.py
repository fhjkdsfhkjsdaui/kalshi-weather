"""Day 6 order lifecycle state machine."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .live_models import LifecycleEvent, LifecycleState, LocalOrderRecord

_TERMINAL_STATES: set[LifecycleState] = {
    "submit_rejected",
    "canceled",
    "partially_filled",
    "filled",
    "timeout",
    "reconciliation_mismatch",
    "terminal_error",
}

_ALLOWED_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
    "intent_created": {"submit_requested", "terminal_error"},
    "submit_requested": {"submit_ack", "submit_rejected", "timeout", "terminal_error"},
    "submit_ack": {
        "cancel_requested",
        "filled",
        "partially_filled",
        "timeout",
        "terminal_error",
    },
    "submit_rejected": set(),
    "cancel_requested": {
        "cancel_ack",
        "filled",
        "partially_filled",
        "timeout",
        "terminal_error",
    },
    "cancel_ack": {
        "canceled",
        "filled",
        "partially_filled",
        "timeout",
        "reconciliation_mismatch",
        "terminal_error",
    },
    "canceled": set(),
    "partially_filled": set(),
    "filled": set(),
    "timeout": set(),
    "reconciliation_mismatch": set(),
    "terminal_error": set(),
}


def is_terminal_lifecycle_state(state: LifecycleState) -> bool:
    """Return True when state is terminal."""
    return state in _TERMINAL_STATES


class OrderLifecycleTracker:
    """Validate and record lifecycle transitions for one order attempt."""

    def __init__(
        self,
        *,
        record: LocalOrderRecord,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.record = record
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def transition(
        self,
        new_state: LifecycleState,
        *,
        note: str | None = None,
        raw_status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Apply one validated state transition and append lifecycle event."""
        current = self.record.current_state
        if new_state == current:
            return
        allowed_next = _ALLOWED_TRANSITIONS.get(current, set())
        if new_state not in allowed_next:
            raise ValueError(f"Invalid lifecycle transition: {current} -> {new_state}")

        now = self._now()
        event = LifecycleEvent(
            state=new_state,
            ts=now,
            note=note,
            raw_status=raw_status,
            metadata=metadata or {},
        )
        self.record.events.append(event)
        self.record.current_state = new_state

        if new_state == "submit_requested":
            self.record.submit_ts = now
        if new_state == "cancel_requested":
            self.record.cancel_ts = now
        if is_terminal_lifecycle_state(new_state):
            self.record.terminal_ts = now

    def _now(self) -> datetime:
        value = self._now_provider()
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)

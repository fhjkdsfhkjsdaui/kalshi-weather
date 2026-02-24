"""Day 6 reconciliation module for local vs remote order state."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from .live_adapter import KalshiLiveOrderAdapter, OrderAdapterError
from .live_models import KalshiOrderStatus, LocalOrderRecord, ReconciliationResult


class OrderReconciler:
    """Poll and compare remote order status against local lifecycle state."""

    def __init__(
        self,
        adapter: KalshiLiveOrderAdapter,
        *,
        sleep_fn: Callable[[float], None] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.adapter = adapter
        self._sleep = sleep_fn or time.sleep
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def reconcile(
        self,
        *,
        record: LocalOrderRecord,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
        on_poll: Callable[[KalshiOrderStatus], None] | None = None,
    ) -> ReconciliationResult:
        """Reconcile local terminal state with remote terminal state."""
        if not record.remote_order_id:
            return ReconciliationResult(
                attempt_id=record.attempt_id,
                local_terminal_state=record.current_state,
                remote_terminal_state="unresolved",
                matched=False,
                mismatch_reason="missing_remote_order_id",
                unresolved=True,
                order_id=None,
                requested_quantity=record.requested_quantity,
                filled_quantity=record.filled_quantity,
                remaining_quantity=record.remaining_quantity,
                submit_to_cancel_ms=_duration_ms(record.submit_ts, record.cancel_ts),
                submit_to_terminal_ms=_duration_ms(record.submit_ts, record.terminal_ts),
                polled_count=0,
            )

        deadline = self._now().timestamp() + poll_timeout_seconds
        last_status: KalshiOrderStatus | None = None
        polled_count = 0
        while self._now().timestamp() <= deadline:
            try:
                status = self.adapter.get_order(record.remote_order_id)
            except OrderAdapterError:
                self._sleep(poll_interval_seconds)
                continue

            polled_count += 1
            last_status = status
            if on_poll is not None:
                on_poll(status)
            if status.is_terminal:
                break
            self._sleep(poll_interval_seconds)

        if last_status is None or not last_status.is_terminal:
            return ReconciliationResult(
                attempt_id=record.attempt_id,
                local_terminal_state=record.current_state,
                remote_terminal_state="unresolved",
                matched=False,
                mismatch_reason="remote_terminal_state_unconfirmed",
                unresolved=True,
                order_id=record.remote_order_id,
                requested_quantity=record.requested_quantity,
                filled_quantity=record.filled_quantity,
                remaining_quantity=record.remaining_quantity,
                submit_to_cancel_ms=_duration_ms(record.submit_ts, record.cancel_ts),
                submit_to_terminal_ms=_duration_ms(record.submit_ts, record.terminal_ts),
                polled_count=polled_count,
            )

        matched, mismatch_reason = _match_terminal_states(
            local_state=record.current_state,
            remote_state=last_status.normalized_status,
        )
        terminal_at = last_status.updated_at or record.terminal_ts
        return ReconciliationResult(
            attempt_id=record.attempt_id,
            local_terminal_state=record.current_state,
            remote_terminal_state=last_status.normalized_status,
            matched=matched,
            mismatch_reason=mismatch_reason,
            unresolved=False,
            order_id=last_status.order_id,
            requested_quantity=last_status.requested_quantity,
            filled_quantity=last_status.filled_quantity,
            remaining_quantity=last_status.remaining_quantity,
            submit_to_cancel_ms=_duration_ms(record.submit_ts, record.cancel_ts),
            submit_to_terminal_ms=_duration_ms(record.submit_ts, terminal_at),
            polled_count=polled_count,
        )

    def _now(self) -> datetime:
        now = self._now_provider()
        return now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)


def _match_terminal_states(*, local_state: str, remote_state: str) -> tuple[bool, str | None]:
    if local_state in {"canceled", "cancel_ack"} and remote_state == "canceled":
        return True, None
    if local_state == "submit_rejected" and remote_state == "rejected":
        return True, None
    if local_state == "partially_filled" and remote_state in {"partially_filled", "filled"}:
        return True, None
    if local_state == "filled" and remote_state == "filled":
        return True, None
    if local_state == "timeout":
        return False, "local_timeout"
    if local_state == "terminal_error":
        return False, "local_terminal_error"
    if local_state == "reconciliation_mismatch":
        return False, "local_already_marked_mismatch"
    return False, f"local_{local_state}_remote_{remote_state}"


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))

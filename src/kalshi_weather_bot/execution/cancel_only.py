"""Day 6 cancel-only live execution runner."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from ..config import Settings
from ..exceptions import JournalError
from ..journal import JournalWriter
from ..redaction import sanitize_text
from .lifecycle import OrderLifecycleTracker, is_terminal_lifecycle_state
from .live_adapter import KalshiLiveOrderAdapter, OrderAdapterError
from .live_models import (
    CancelOnlyAttemptResult,
    CancelOnlyBatchResult,
    CancelOnlyBatchSummary,
    CancelOnlyOrderIntent,
    KalshiOrderStatus,
    LocalOrderRecord,
    ReconciliationResult,
)
from .reconciliation import OrderReconciler

_FILL_STATES = {"filled", "partially_filled"}


class CancelOnlyRunner:
    """Run submit->cancel validation attempts with strict cancel-only safeguards."""

    def __init__(
        self,
        *,
        settings: Settings,
        adapter: KalshiLiveOrderAdapter,
        logger: logging.Logger,
        journal: JournalWriter | None = None,
        reconciler: OrderReconciler | None = None,
        now_provider: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        attempt_id_provider: Callable[[int], str] | None = None,
    ) -> None:
        self.settings = settings
        self.adapter = adapter
        self.logger = logger
        self.journal = journal
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._sleep = sleep_fn or time.sleep
        self._attempt_id_provider = attempt_id_provider or self._default_attempt_id
        self.reconciler = reconciler or OrderReconciler(
            adapter=adapter,
            sleep_fn=self._sleep,
            now_provider=self._now_provider,
        )

    def run_batch(
        self,
        *,
        intent: CancelOnlyOrderIntent,
        attempts: int,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
        cancel_delay_ms: int,
    ) -> CancelOnlyBatchResult:
        """Run cancel-only validation attempts and return detailed batch result."""
        self._validate_startup_guards(
            intent=intent,
            attempts=attempts,
            poll_timeout_seconds=poll_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            cancel_delay_ms=cancel_delay_ms,
        )

        results: list[CancelOnlyAttemptResult] = []
        halted_early = False
        halt_reason: str | None = None

        for attempt_index in range(1, attempts + 1):
            if halted_early:
                break
            if attempt_index > 1 and self.settings.cancel_only_min_delay_between_attempts_ms > 0:
                self._sleep(self.settings.cancel_only_min_delay_between_attempts_ms / 1000.0)

            attempt_id = self._attempt_id_provider(attempt_index)
            record = LocalOrderRecord(
                attempt_id=attempt_id,
                market_id=intent.market_id,
                side=intent.side,
                price_cents=intent.price_cents,
                quantity=intent.quantity,
            )
            tracker = OrderLifecycleTracker(record=record, now_provider=self._now_provider)
            self._write_event(
                "order_lifecycle_start",
                payload={
                    "attempt_id": attempt_id,
                    "attempt_index": attempt_index,
                    "market_id": intent.market_id,
                    "side": intent.side,
                    "price_cents": intent.price_cents,
                    "quantity": intent.quantity,
                },
            )

            tracker.transition("submit_requested")
            self._write_event(
                "order_submit_requested",
                payload={
                    "attempt_id": attempt_id,
                    "market_id": intent.market_id,
                    "side": intent.side,
                    "price_cents": intent.price_cents,
                    "quantity": intent.quantity,
                },
            )

            submit_status: KalshiOrderStatus | None = None
            try:
                submit_status = self.adapter.submit_order(
                    intent,
                    client_order_id=attempt_id,
                )
            except OrderAdapterError as exc:
                record.error_category = exc.category
                record.error_message = sanitize_text(str(exc))
                tracker.transition(
                    "submit_rejected",
                    note="submit_error",
                    metadata={"error_category": exc.category},
                )
                self._write_event(
                    "order_submit_rejected",
                    payload={
                        "attempt_id": attempt_id,
                        "error_category": exc.category,
                        "status_code": exc.status_code,
                        "error": sanitize_text(str(exc)),
                    },
                )
                results.append(
                    CancelOnlyAttemptResult(
                        attempt_id=attempt_id,
                        attempt_index=attempt_index,
                        local_record=record,
                    )
                )
                continue

            self._apply_remote_snapshot(record=record, status=submit_status)
            if submit_status.normalized_status == "rejected":
                tracker.transition(
                    "submit_rejected",
                    raw_status=submit_status.status_raw,
                )
                self._write_event(
                    "order_submit_rejected",
                    payload={
                        "attempt_id": attempt_id,
                        "order_id": submit_status.order_id,
                        "raw_status": submit_status.status_raw,
                        "normalized_status": submit_status.normalized_status,
                    },
                )
                results.append(
                    CancelOnlyAttemptResult(
                        attempt_id=attempt_id,
                        attempt_index=attempt_index,
                        local_record=record,
                    )
                )
                continue

            tracker.transition("submit_ack", raw_status=submit_status.status_raw)
            self._write_event(
                "order_submit_ack",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": submit_status.order_id,
                    "raw_status": submit_status.status_raw,
                    "normalized_status": submit_status.normalized_status,
                    "filled_quantity": submit_status.filled_quantity,
                },
            )

            if submit_status.normalized_status in _FILL_STATES:
                self._transition_fill_state(tracker, submit_status)
                self._write_unexpected_fill_event(attempt_id=attempt_id, status=submit_status)
                if self.settings.cancel_only_halt_on_any_fill:
                    halted_early = True
                    halt_reason = "unexpected_fill_on_submit"
                    self._write_event(
                        "cancel_only_halt",
                        payload={"attempt_id": attempt_id, "reason": halt_reason},
                    )
                results.append(
                    CancelOnlyAttemptResult(
                        attempt_id=attempt_id,
                        attempt_index=attempt_index,
                        local_record=record,
                        halted_early=halted_early,
                        halt_reason=halt_reason,
                    )
                )
                continue

            if cancel_delay_ms > 0:
                self._sleep(cancel_delay_ms / 1000.0)

            tracker.transition("cancel_requested")
            self._write_event(
                "order_cancel_requested",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": submit_status.order_id,
                    "cancel_delay_ms": cancel_delay_ms,
                },
            )

            try:
                cancel_status = self.adapter.cancel_order(submit_status.order_id)
            except OrderAdapterError as exc:
                record.error_category = exc.category
                record.error_message = sanitize_text(str(exc))
                tracker.transition(
                    "timeout",
                    note="cancel_request_failed",
                    metadata={"error_category": exc.category},
                )
                self._write_event(
                    "order_cancel_ack",
                    payload={
                        "attempt_id": attempt_id,
                        "order_id": submit_status.order_id,
                        "acknowledged": False,
                        "error_category": exc.category,
                        "status_code": exc.status_code,
                        "error": sanitize_text(str(exc)),
                    },
                )
            else:
                self._apply_remote_snapshot(record=record, status=cancel_status)
                tracker.transition("cancel_ack", raw_status=cancel_status.status_raw)
                self._write_event(
                    "order_cancel_ack",
                    payload={
                        "attempt_id": attempt_id,
                        "order_id": cancel_status.order_id,
                        "acknowledged": True,
                        "raw_status": cancel_status.status_raw,
                        "normalized_status": cancel_status.normalized_status,
                        "filled_quantity": cancel_status.filled_quantity,
                    },
                )
                if cancel_status.normalized_status == "canceled":
                    tracker.transition("canceled", raw_status=cancel_status.status_raw)
                elif cancel_status.normalized_status in _FILL_STATES:
                    self._transition_fill_state(tracker, cancel_status)
                    self._write_unexpected_fill_event(attempt_id=attempt_id, status=cancel_status)
                    if self.settings.cancel_only_halt_on_any_fill:
                        halted_early = True
                        halt_reason = "unexpected_fill_on_cancel_ack"
                        self._write_event(
                            "cancel_only_halt",
                            payload={"attempt_id": attempt_id, "reason": halt_reason},
                        )

            reconciliation = self.reconciler.reconcile(
                record=record,
                poll_timeout_seconds=poll_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                on_poll=lambda status, aid=attempt_id: self._write_event(
                    "order_status_polled",
                    payload={
                        "attempt_id": aid,
                        "order_id": status.order_id,
                        "raw_status": status.status_raw,
                        "normalized_status": status.normalized_status,
                        "filled_quantity": status.filled_quantity,
                    },
                ),
            )
            self._write_event(
                "order_reconciliation_result",
                payload=reconciliation.model_dump(mode="json"),
            )

            self._apply_reconciliation_terminal_state(
                tracker=tracker,
                reconciliation=reconciliation,
            )
            if reconciliation.remote_terminal_state in _FILL_STATES:
                self.logger.critical(
                    "UNEXPECTED FILL detected during reconciliation: attempt=%s "
                    "order=%s remote_state=%s filled=%d",
                    attempt_id,
                    reconciliation.order_id,
                    reconciliation.remote_terminal_state,
                    reconciliation.filled_quantity,
                )
                self._write_event(
                    "order_unexpected_fill_detected",
                    payload={
                        "severity": "CRITICAL",
                        "attempt_id": attempt_id,
                        "order_id": reconciliation.order_id,
                        "remote_terminal_state": reconciliation.remote_terminal_state,
                        "filled_quantity": reconciliation.filled_quantity,
                    },
                )
                if self.settings.cancel_only_halt_on_any_fill:
                    halted_early = True
                    halt_reason = "unexpected_fill_on_reconciliation"
                    self._write_event(
                        "cancel_only_halt",
                        payload={"attempt_id": attempt_id, "reason": halt_reason},
                    )

            results.append(
                CancelOnlyAttemptResult(
                    attempt_id=attempt_id,
                    attempt_index=attempt_index,
                    local_record=record,
                    reconciliation=reconciliation,
                    halted_early=halted_early,
                    halt_reason=halt_reason,
                )
            )

        summary = self._build_summary(
            attempts_requested=attempts,
            results=results,
            halted_early=halted_early,
            halt_reason=halt_reason,
        )
        self._write_event(
            "cancel_only_batch_summary",
            payload=summary.model_dump(mode="json"),
        )
        return CancelOnlyBatchResult(attempts=results, summary=summary)

    def _validate_startup_guards(
        self,
        *,
        intent: CancelOnlyOrderIntent,
        attempts: int,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
        cancel_delay_ms: int,
    ) -> None:
        if self.settings.execution_mode != "live_cancel_only":
            raise ValueError("EXECUTION_MODE must be 'live_cancel_only' for Day 6 runner.")
        if not self.settings.allow_live_api:
            raise ValueError("ALLOW_LIVE_API must be true for Day 6 runner.")
        if not self.settings.cancel_only_enabled:
            raise ValueError("CANCEL_ONLY_ENABLED must be true for Day 6 runner.")
        if attempts <= 0:
            raise ValueError("attempts must be > 0.")
        if attempts > self.settings.cancel_only_max_attempts_per_run:
            raise ValueError(
                "attempts exceeds CANCEL_ONLY_MAX_ATTEMPTS_PER_RUN "
                f"({self.settings.cancel_only_max_attempts_per_run})."
            )
        if intent.quantity > self.settings.cancel_only_max_qty:
            raise ValueError(
                "intent.quantity exceeds CANCEL_ONLY_MAX_QTY "
                f"({self.settings.cancel_only_max_qty})."
            )
        if poll_timeout_seconds <= 0:
            raise ValueError("poll_timeout_seconds must be > 0.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0.")
        if poll_interval_seconds > poll_timeout_seconds:
            raise ValueError("poll_interval_seconds cannot exceed poll_timeout_seconds.")
        if cancel_delay_ms < 0:
            raise ValueError("cancel_delay_ms must be >= 0.")

    @staticmethod
    def _apply_remote_snapshot(record: LocalOrderRecord, status: KalshiOrderStatus) -> None:
        record.remote_order_id = status.order_id
        record.remote_status_raw = status.status_raw
        record.remote_status_normalized = status.normalized_status
        record.requested_quantity = status.requested_quantity
        record.filled_quantity = status.filled_quantity
        record.remaining_quantity = status.remaining_quantity

    @staticmethod
    def _transition_fill_state(
        tracker: OrderLifecycleTracker,
        status: KalshiOrderStatus,
    ) -> None:
        if status.normalized_status == "filled":
            tracker.transition("filled", raw_status=status.status_raw)
            return
        tracker.transition("partially_filled", raw_status=status.status_raw)

    @staticmethod
    def _apply_reconciliation_terminal_state(
        *,
        tracker: OrderLifecycleTracker,
        reconciliation: ReconciliationResult,
    ) -> None:
        current = tracker.record.current_state
        if is_terminal_lifecycle_state(current):
            return
        if reconciliation.unresolved:
            tracker.transition("timeout", note="reconciliation_unresolved")
            return
        remote = reconciliation.remote_terminal_state
        if remote == "canceled":
            tracker.transition("canceled")
            return
        if remote == "rejected":
            if tracker.record.current_state == "submit_requested":
                tracker.transition("submit_rejected")
            else:
                tracker.transition("terminal_error", note="remote_rejected_after_submit_ack")
            return
        if remote == "filled":
            tracker.transition("filled")
            return
        if remote == "partially_filled":
            tracker.transition("partially_filled")
            return
        if not reconciliation.matched:
            tracker.transition("reconciliation_mismatch", note=reconciliation.mismatch_reason)
            return
        tracker.transition("terminal_error", note="unknown_terminal_resolution")

    @staticmethod
    def _build_summary(
        *,
        attempts_requested: int,
        results: list[CancelOnlyAttemptResult],
        halted_early: bool,
        halt_reason: str | None,
    ) -> CancelOnlyBatchSummary:
        submit_ack_count = 0
        cancel_success_count = 0
        rejected_count = 0
        unresolved_count = 0
        mismatch_count = 0
        unexpected_fill_count = 0

        for attempt in results:
            states = {event.state for event in attempt.local_record.events}
            if "submit_ack" in states:
                submit_ack_count += 1
            if attempt.local_record.current_state == "canceled":
                cancel_success_count += 1
            if attempt.local_record.current_state == "submit_rejected":
                rejected_count += 1

            has_fill = attempt.local_record.current_state in {"filled", "partially_filled"}
            reconciliation = attempt.reconciliation
            if reconciliation is not None:
                if reconciliation.unresolved:
                    unresolved_count += 1
                if not reconciliation.matched:
                    mismatch_count += 1
                if reconciliation.remote_terminal_state in _FILL_STATES:
                    has_fill = True
            if has_fill:
                unexpected_fill_count += 1

        return CancelOnlyBatchSummary(
            attempts_requested=attempts_requested,
            attempts_executed=len(results),
            submit_ack_count=submit_ack_count,
            cancel_success_count=cancel_success_count,
            rejected_count=rejected_count,
            unresolved_count=unresolved_count,
            reconciliation_mismatch_count=mismatch_count,
            unexpected_fill_count=unexpected_fill_count,
            halted_early=halted_early,
            halt_reason=halt_reason,
        )

    def _write_unexpected_fill_event(self, *, attempt_id: str, status: KalshiOrderStatus) -> None:
        self.logger.critical(
            "UNEXPECTED FILL in cancel-only mode: attempt=%s order=%s filled=%d status=%s",
            attempt_id,
            status.order_id,
            status.filled_quantity,
            status.normalized_status,
        )
        self._write_event(
            "order_unexpected_fill_detected",
            payload={
                "severity": "CRITICAL",
                "attempt_id": attempt_id,
                "order_id": status.order_id,
                "raw_status": status.status_raw,
                "normalized_status": status.normalized_status,
                "filled_quantity": status.filled_quantity,
            },
        )

    def _write_event(self, event_type: str, payload: dict[str, object]) -> None:
        if self.journal is None:
            return
        try:
            self.journal.write_event(event_type=event_type, payload=payload)
        except JournalError as exc:
            self.logger.error(
                "Failed to write %s journal event: %s",
                event_type,
                sanitize_text(str(exc)),
            )

    def _default_attempt_id(self, attempt_index: int) -> str:
        now = self._now_provider()
        ts = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
        return f"cancelonly-{ts:%Y%m%dT%H%M%S}-{attempt_index:03d}"

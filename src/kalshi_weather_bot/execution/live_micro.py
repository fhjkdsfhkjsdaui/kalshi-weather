"""Day 7 guarded live_micro execution runner."""

from __future__ import annotations

import inspect
import logging
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from ..config import Settings
from ..exceptions import JournalError
from ..journal import JournalWriter
from ..redaction import sanitize_text
from .lifecycle import OrderLifecycleTracker
from .live_adapter import KalshiLiveOrderAdapter, OrderAdapterError
from .live_models import CancelOnlyOrderIntent, KalshiOrderStatus, LocalOrderRecord
from .micro_models import (
    MicroOrderAttempt,
    MicroOrderOutcome,
    MicroSessionResult,
    MicroSessionSummary,
    MicroTradeCandidate,
    PositionSnapshot,
    TradePolicyDecision,
)
from .micro_policy import MicroTradePolicy
from .position_tracker import MicroPositionTracker

_FILL_STATES = {"filled", "partially_filled"}


class LiveMicroRunner:
    """Execute tiny, supervised live orders with strict Day 7 safeguards."""

    def __init__(
        self,
        *,
        settings: Settings,
        adapter: KalshiLiveOrderAdapter,
        logger: logging.Logger,
        journal: JournalWriter | None = None,
        policy: MicroTradePolicy | None = None,
        position_tracker: MicroPositionTracker | None = None,
        now_provider: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        attempt_id_provider: Callable[[int], str] | None = None,
    ) -> None:
        self.settings = settings
        self.adapter = adapter
        self.logger = logger
        self.journal = journal
        self.policy = policy or MicroTradePolicy(settings=settings, logger=logger)
        self.position_tracker = position_tracker or MicroPositionTracker(now_provider=now_provider)
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._sleep = sleep_fn or time.sleep
        self._attempt_id_provider = attempt_id_provider or self._default_attempt_id

    def run_candidates(
        self,
        *,
        candidates: list[MicroTradeCandidate],
        max_trades_this_run: int | None = None,
        poll_timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
        supervised: bool,
        cycles_processed: int = 1,
    ) -> MicroSessionResult:
        """Evaluate candidates and execute approved ones in guarded live_micro mode."""
        effective_max_trades = (
            max_trades_this_run
            if max_trades_this_run is not None
            else self.settings.micro_max_trades_per_run
        )
        timeout = (
            poll_timeout_seconds
            if poll_timeout_seconds is not None
            else self.settings.micro_poll_timeout_seconds
        )
        interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self.settings.micro_poll_interval_seconds
        )
        self._validate_startup_guards(
            supervised=supervised,
            max_trades_this_run=effective_max_trades,
            poll_timeout_seconds=timeout,
            poll_interval_seconds=interval,
        )

        attempts: list[MicroOrderAttempt] = []
        skip_reasons: Counter[str] = Counter()
        orders_submitted = 0
        trades_allowed = 0
        fills = 0
        partial_fills = 0
        cancels = 0
        rejects = 0
        unresolved = 0
        trades_submitted_run = 0
        halted_early = False
        halt_reason: str | None = None

        for idx, candidate in enumerate(candidates, start=1):
            if halted_early:
                break

            now = self._now()
            policy_context = self.position_tracker.policy_context(
                now=now,
                trades_executed_run=trades_submitted_run,
            )
            decision = self.policy.evaluate(
                candidate,
                policy_context,
                max_trades_per_run=effective_max_trades,
            )
            self._write_event(
                "trade_policy_evaluated",
                payload={
                    "candidate_market_id": candidate.market_id,
                    "allowed": decision.allowed,
                    "decision_code": decision.decision_code,
                    "reasons": decision.reasons[:5],
                    "warnings": decision.warnings[:5],
                },
            )

            if not decision.allowed:
                for reason in decision.reasons or [decision.decision_code]:
                    skip_reasons[reason] += 1
                self._write_event(
                    "trade_policy_denied",
                    payload={
                        "candidate_market_id": candidate.market_id,
                        "decision_code": decision.decision_code,
                        "reasons": decision.reasons[:5],
                    },
                )
                self._write_event(
                    "risk_cap_hit",
                    payload={
                        "candidate_market_id": candidate.market_id,
                        "decision_code": decision.decision_code,
                        "reasons": decision.reasons[:5],
                    },
                )
                attempts.append(
                    MicroOrderAttempt(
                        candidate=candidate,
                        policy_decision=decision,
                        attempted=False,
                        outcome="skipped",
                        reasons=decision.reasons,
                    )
                )
                continue

            trades_allowed += 1
            self._write_event(
                "trade_policy_approved",
                payload={
                    "candidate_market_id": candidate.market_id,
                    "decision_code": decision.decision_code,
                },
            )
            attempt = self._execute_candidate(
                candidate=candidate,
                decision=decision,
                attempt_index=idx,
                poll_timeout_seconds=timeout,
                poll_interval_seconds=interval,
            )
            attempts.append(attempt)

            if attempt.attempted and attempt.order_id:
                orders_submitted += 1
                trades_submitted_run += 1

            if attempt.outcome == "filled":
                fills += 1
            elif attempt.outcome == "partially_filled":
                partial_fills += 1
            elif attempt.outcome == "canceled":
                cancels += 1
            elif attempt.outcome == "rejected":
                rejects += 1
            elif attempt.outcome in {"unresolved", "error"}:
                unresolved += 1

            if attempt.halted:
                halted_early = True
                halt_reason = attempt.reasons[0] if attempt.reasons else "micro_halt"
                self._write_event(
                    "micro_halt",
                    payload={
                        "candidate_market_id": candidate.market_id,
                        "reason": halt_reason,
                        "order_id": attempt.order_id,
                    },
                )

            if not halted_early and self.settings.micro_min_seconds_between_trades > 0:
                self._sleep(self.settings.micro_min_seconds_between_trades)

        position_snapshot = self.position_tracker.snapshot(now=self._now())
        summary = self._build_summary(
            cycles_processed=cycles_processed,
            candidates_seen=len(candidates),
            trades_allowed=trades_allowed,
            skip_reasons=dict(skip_reasons),
            orders_submitted=orders_submitted,
            fills=fills,
            partial_fills=partial_fills,
            cancels=cancels,
            rejects=rejects,
            unresolved=unresolved,
            position_snapshot=position_snapshot,
            trades_submitted_run=trades_submitted_run,
            halted_early=halted_early,
            halt_reason=halt_reason,
            max_trades_per_run=effective_max_trades,
        )
        self._write_event("micro_session_summary", payload=summary.model_dump(mode="json"))
        return MicroSessionResult(
            attempts=attempts,
            position_snapshot=position_snapshot,
            summary=summary,
        )

    def _execute_candidate(
        self,
        *,
        candidate: MicroTradeCandidate,
        decision: TradePolicyDecision,
        attempt_index: int,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> MicroOrderAttempt:
        attempt_id = self._attempt_id_provider(attempt_index)
        record = LocalOrderRecord(
            attempt_id=attempt_id,
            market_id=candidate.market_id,
            side=candidate.side,
            price_cents=candidate.price_cents,
            quantity=candidate.quantity,
        )
        tracker = OrderLifecycleTracker(record=record, now_provider=self._now_provider)
        tracker.transition("submit_requested")

        order_intent = CancelOnlyOrderIntent(
            market_id=candidate.market_id,
            side=candidate.side,
            price_cents=candidate.price_cents,
            quantity=candidate.quantity,
            metadata={
                "strategy_tag": candidate.strategy_tag,
                "parsed_contract_ref": candidate.parsed_contract_ref,
                "weather_snapshot_ref": candidate.weather_snapshot_ref,
            },
        )

        try:
            submit_status = self.adapter.submit_order(order_intent, client_order_id=attempt_id)
        except OrderAdapterError as exc:
            record.error_category = exc.category
            record.error_message = sanitize_text(str(exc))
            tracker.transition(
                "submit_rejected",
                note="submit_error",
                metadata={"error_category": exc.category},
            )
            self._write_event(
                "micro_order_rejected",
                payload={
                    "attempt_id": attempt_id,
                    "candidate_market_id": candidate.market_id,
                    "error_category": exc.category,
                    "status_code": exc.status_code,
                    "error": sanitize_text(str(exc)),
                },
            )
            return MicroOrderAttempt(
                candidate=candidate,
                policy_decision=decision,
                attempted=False,
                outcome="rejected",
                reasons=["submit_error", exc.category],
            )

        self._apply_remote_snapshot(record=record, status=submit_status)
        tracker.transition("submit_ack", raw_status=submit_status.status_raw)
        self._write_event(
            "micro_order_submitted",
            payload={
                "attempt_id": attempt_id,
                "order_id": submit_status.order_id,
                "candidate_market_id": candidate.market_id,
                "side": candidate.side,
                "price_cents": candidate.price_cents,
                "quantity": candidate.quantity,
                "status": submit_status.normalized_status,
                "status_raw": submit_status.status_raw,
                "status_raw_keys": sorted(submit_status.raw.keys())[:20],
            },
        )
        self.position_tracker.record_trade_submission(timestamp=self._now())

        terminal_status = submit_status
        polled_count = 0
        if not submit_status.is_terminal:
            polled_status, polled_count = self._poll_terminal_status(
                order_id=submit_status.order_id,
                poll_timeout_seconds=poll_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                attempt_id=attempt_id,
            )
            if polled_status is None:
                (
                    cancel_status,
                    cancel_polled_count,
                    cancel_reason,
                ) = self._attempt_cancel_after_timeout(
                    order_id=submit_status.order_id,
                    attempt_id=attempt_id,
                    tracker=tracker,
                    record=record,
                    poll_timeout_seconds=poll_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                )
                polled_count += cancel_polled_count
                if cancel_status is None:
                    tracker.transition("timeout", note="reconciliation_unresolved")
                    unresolved_reason = cancel_reason or "remote_terminal_state_unconfirmed"
                    self._write_event(
                        "micro_order_unresolved",
                        payload={
                            "attempt_id": attempt_id,
                            "order_id": submit_status.order_id,
                            "reason": unresolved_reason,
                        },
                    )
                    halted = self.settings.micro_halt_on_reconciliation_mismatch
                    return MicroOrderAttempt(
                        candidate=candidate,
                        policy_decision=decision,
                        attempted=True,
                        outcome="unresolved",
                        order_id=submit_status.order_id,
                        terminal_status="unresolved",
                        filled_quantity=record.filled_quantity,
                        reasons=["reconciliation_unresolved"],
                        polled_count=polled_count,
                        unresolved=True,
                        reconciliation_mismatch=True,
                        halted=halted,
                    )
                terminal_status = cancel_status
            else:
                terminal_status = polled_status

        self._apply_remote_snapshot(record=record, status=terminal_status)
        outcome, reasons, mismatch = self._apply_terminal_status(
            tracker=tracker,
            status=terminal_status,
            attempt_id=attempt_id,
            candidate=candidate,
        )

        halted = False
        if mismatch and self.settings.micro_halt_on_reconciliation_mismatch:
            self.logger.critical(
                "MICRO RECONCILIATION MISMATCH: outcome=%s order=%s market=%s",
                outcome,
                terminal_status.order_id,
                candidate.market_id,
            )
            halted = True
            reasons.insert(0, "reconciliation_mismatch")
        if (
            outcome in {"filled", "partially_filled"}
            and self.settings.micro_halt_on_any_unexpected_fill_state
            and terminal_status.filled_quantity <= 0
        ):
            self.logger.critical(
                "MICRO UNEXPECTED FILL STATE: %s filled_quantity=%d order=%s market=%s",
                outcome,
                terminal_status.filled_quantity,
                terminal_status.order_id,
                candidate.market_id,
            )
            self._write_event(
                "micro_unexpected_fill_state",
                payload={
                    "severity": "CRITICAL",
                    "attempt_id": attempt_id,
                    "order_id": terminal_status.order_id,
                    "outcome": outcome,
                    "filled_quantity": terminal_status.filled_quantity,
                    "market_id": candidate.market_id,
                },
            )
            halted = True
            reasons.insert(0, "unexpected_fill_state_without_quantity")

        return MicroOrderAttempt(
            candidate=candidate,
            policy_decision=decision,
            attempted=True,
            outcome=outcome,
            order_id=terminal_status.order_id,
            terminal_status=terminal_status.normalized_status,
            filled_quantity=terminal_status.filled_quantity,
            reasons=reasons,
            polled_count=polled_count,
            unresolved=False,
            reconciliation_mismatch=mismatch,
            halted=halted,
        )

    def _poll_terminal_status(
        self,
        *,
        order_id: str,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
        attempt_id: str,
    ) -> tuple[KalshiOrderStatus | None, int]:
        deadline = self._now().timestamp() + poll_timeout_seconds
        polled_count = 0
        last_status: KalshiOrderStatus | None = None
        while self._now().timestamp() <= deadline:
            try:
                status = self._get_order_with_attempt_context(
                    order_id=order_id,
                    attempt_id=attempt_id,
                )
            except OrderAdapterError as exc:
                self._write_event(
                    "order_status_polled",
                    payload={
                        "attempt_id": attempt_id,
                        "order_id": order_id,
                        "error_category": exc.category,
                        "error": sanitize_text(str(exc)),
                        "status": "poll_error",
                    },
                )
                self._sleep(poll_interval_seconds)
                continue

            polled_count += 1
            last_status = status
            self._write_event(
                "order_status_polled",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": status.order_id,
                    "normalized_status": status.normalized_status,
                    "status_raw": status.status_raw,
                    "filled_quantity": status.filled_quantity,
                    "status_raw_keys": (
                        sorted(status.raw.keys())[:20]
                        if status.normalized_status == "unknown"
                        else []
                    ),
                },
            )
            if status.is_terminal:
                return status, polled_count
            self._sleep(poll_interval_seconds)

        return (last_status if last_status and last_status.is_terminal else None), polled_count

    def _attempt_cancel_after_timeout(
        self,
        *,
        order_id: str,
        attempt_id: str,
        tracker: OrderLifecycleTracker,
        record: LocalOrderRecord,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> tuple[KalshiOrderStatus | None, int, str | None]:
        cancel_fn = getattr(self.adapter, "cancel_order", None)
        if not callable(cancel_fn):
            return None, 0, "cancel_method_unavailable"

        tracker.transition("cancel_requested", note="poll_timeout_cancel")
        self._write_event(
            "micro_order_cancel_requested",
            payload={
                "attempt_id": attempt_id,
                "order_id": order_id,
                "reason": "poll_timeout",
            },
        )
        try:
            cancel_status = self._cancel_order_with_attempt_context(
                order_id=order_id,
                attempt_id=attempt_id,
            )
        except OrderAdapterError as exc:
            record.error_category = exc.category
            record.error_message = sanitize_text(str(exc))
            self._write_event(
                "micro_order_cancel_error",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": order_id,
                    "error_category": exc.category,
                    "status_code": exc.status_code,
                    "error": sanitize_text(str(exc)),
                },
            )
            final = self._final_reconciliation_read(
                order_id=order_id,
                attempt_id=attempt_id,
            )
            if final is not None:
                return final, 0, None
            return None, 0, "cancel_after_timeout_failed"

        self._apply_remote_snapshot(record=record, status=cancel_status)
        tracker.transition("cancel_ack", raw_status=cancel_status.status_raw)
        self._write_event(
            "micro_order_cancel_ack",
            payload={
                "attempt_id": attempt_id,
                "order_id": cancel_status.order_id,
                "normalized_status": cancel_status.normalized_status,
                "filled_quantity": cancel_status.filled_quantity,
            },
        )
        if cancel_status.is_terminal:
            return cancel_status, 0, None

        polled_status, polled_count = self._poll_terminal_status(
            order_id=order_id,
            poll_timeout_seconds=poll_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            attempt_id=attempt_id,
        )
        if polled_status is None:
            final = self._final_reconciliation_read(
                order_id=order_id,
                attempt_id=attempt_id,
            )
            if final is not None:
                return final, polled_count, None
            return None, polled_count, "remote_terminal_state_unconfirmed_after_cancel"
        return polled_status, polled_count, None

    def _get_order_with_attempt_context(
        self,
        *,
        order_id: str,
        attempt_id: str,
    ) -> KalshiOrderStatus:
        get_fn = self.adapter.get_order
        if self._supports_client_order_id(get_fn):
            return get_fn(order_id, client_order_id=attempt_id)
        return get_fn(order_id)

    def _cancel_order_with_attempt_context(
        self,
        *,
        order_id: str,
        attempt_id: str,
    ) -> KalshiOrderStatus:
        cancel_fn = self.adapter.cancel_order
        if self._supports_client_order_id(cancel_fn):
            return cancel_fn(order_id, client_order_id=attempt_id)
        return cancel_fn(order_id)

    def _final_reconciliation_read(
        self,
        *,
        order_id: str,
        attempt_id: str,
    ) -> KalshiOrderStatus | None:
        """One last status read (direct + list fallback) before declaring unresolved.

        Returns a KalshiOrderStatus only when the remote state is terminal.
        Returns None when the read fails or the order is still non-terminal.
        """
        self._write_event(
            "micro_order_final_reconciliation_read",
            payload={
                "attempt_id": attempt_id,
                "order_id": order_id,
            },
        )
        try:
            status = self._get_order_with_attempt_context(
                order_id=order_id,
                attempt_id=attempt_id,
            )
        except OrderAdapterError:
            return None
        self._write_event(
            "order_status_polled",
            payload={
                "attempt_id": attempt_id,
                "order_id": status.order_id,
                "normalized_status": status.normalized_status,
                "status_raw": status.status_raw,
                "filled_quantity": status.filled_quantity,
                "source": "final_reconciliation_read",
            },
        )
        if status.is_terminal:
            return status
        return None

    @staticmethod
    def _supports_client_order_id(callable_obj: object) -> bool:
        try:
            signature = inspect.signature(callable_obj)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return "client_order_id" in signature.parameters

    def _apply_terminal_status(
        self,
        *,
        tracker: OrderLifecycleTracker,
        status: KalshiOrderStatus,
        attempt_id: str,
        candidate: MicroTradeCandidate,
    ) -> tuple[MicroOrderOutcome, list[str], bool]:
        reasons: list[str] = []
        mismatch = False

        if status.normalized_status == "filled":
            tracker.transition("filled", raw_status=status.status_raw)
            self._write_event(
                "micro_order_fill_detected",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": status.order_id,
                    "filled_quantity": status.filled_quantity,
                    "market_id": candidate.market_id,
                },
            )
            self._apply_confirmed_fill(candidate=candidate, status=status)
            return "filled", reasons, mismatch

        if status.normalized_status == "partially_filled":
            tracker.transition("partially_filled", raw_status=status.status_raw)
            self._write_event(
                "micro_order_partial_fill_detected",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": status.order_id,
                    "filled_quantity": status.filled_quantity,
                    "market_id": candidate.market_id,
                },
            )
            self._apply_confirmed_fill(candidate=candidate, status=status)
            return "partially_filled", reasons, mismatch

        if status.normalized_status == "rejected":
            tracker.transition("terminal_error", note="remote_rejected_after_submit_ack")
            self._write_event(
                "micro_order_rejected",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": status.order_id,
                    "normalized_status": status.normalized_status,
                    "raw_status": status.status_raw,
                },
            )
            mismatch = True
            reasons.append("remote_rejected_after_submit_ack")
            return "rejected", reasons, mismatch

        if status.normalized_status == "canceled":
            if tracker.record.current_state in {"cancel_requested", "cancel_ack"}:
                tracker.transition("canceled", raw_status=status.status_raw)
                self._write_event(
                    "micro_order_canceled",
                    payload={
                        "attempt_id": attempt_id,
                        "order_id": status.order_id,
                        "market_id": candidate.market_id,
                        "reason": "timeout_cancelled",
                    },
                )
                return "canceled", reasons, mismatch

            tracker.transition("terminal_error", note="remote_canceled_without_local_cancel")
            self._write_event(
                "micro_order_unresolved",
                payload={
                    "attempt_id": attempt_id,
                    "order_id": status.order_id,
                    "reason": "remote_canceled_without_local_cancel",
                },
            )
            mismatch = True
            reasons.append("remote_canceled_without_local_cancel")
            return "canceled", reasons, mismatch

        tracker.transition("terminal_error", note="unexpected_terminal_status")
        self._write_event(
            "micro_order_unresolved",
            payload={
                "attempt_id": attempt_id,
                "order_id": status.order_id,
                "reason": "unexpected_terminal_status",
                "normalized_status": status.normalized_status,
                "raw_status": status.status_raw,
            },
        )
        mismatch = True
        reasons.append("unexpected_terminal_status")
        return "error", reasons, mismatch

    def _apply_confirmed_fill(
        self,
        *,
        candidate: MicroTradeCandidate,
        status: KalshiOrderStatus,
    ) -> None:
        filled_qty = max(0, min(status.filled_quantity, candidate.quantity))
        if filled_qty <= 0:
            return
        update = self.position_tracker.apply_confirmed_fill(
            market_id=candidate.market_id,
            side=candidate.side,
            price_cents=candidate.price_cents,
            filled_quantity=filled_qty,
            order_id=status.order_id,
            timestamp=self._now(),
        )
        for event in update.events:
            self._write_event(event.event_type, payload=event.model_dump(mode="json"))

    @staticmethod
    def _apply_remote_snapshot(record: LocalOrderRecord, status: KalshiOrderStatus) -> None:
        record.remote_order_id = status.order_id
        record.remote_status_raw = status.status_raw
        record.remote_status_normalized = status.normalized_status
        record.requested_quantity = status.requested_quantity
        record.filled_quantity = status.filled_quantity
        record.remaining_quantity = status.remaining_quantity

    def _validate_startup_guards(
        self,
        *,
        supervised: bool,
        max_trades_this_run: int,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        if self.settings.execution_mode != "live_micro":
            raise ValueError("EXECUTION_MODE must be 'live_micro' for Day 7 runner.")
        if not self.settings.allow_live_api:
            raise ValueError("ALLOW_LIVE_API must be true for Day 7 runner.")
        if not self.settings.allow_live_fills:
            raise ValueError("ALLOW_LIVE_FILLS must be true for Day 7 runner.")
        if not self.settings.micro_mode_enabled:
            raise ValueError("MICRO_MODE_ENABLED must be true for Day 7 runner.")
        if self.settings.micro_require_supervised_mode and not supervised:
            raise ValueError("MICRO_REQUIRE_SUPERVISED_MODE requires --supervised.")
        if max_trades_this_run <= 0:
            raise ValueError("max_trades_this_run must be > 0.")
        if max_trades_this_run > self.settings.micro_max_trades_per_run:
            raise ValueError(
                "max_trades_this_run exceeds MICRO_MAX_TRADES_PER_RUN "
                f"({self.settings.micro_max_trades_per_run})."
            )
        if poll_timeout_seconds <= 0:
            raise ValueError("poll_timeout_seconds must be > 0.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0.")
        if poll_interval_seconds > poll_timeout_seconds:
            raise ValueError("poll_interval_seconds cannot exceed poll_timeout_seconds.")

    def _build_summary(
        self,
        *,
        cycles_processed: int,
        candidates_seen: int,
        trades_allowed: int,
        skip_reasons: dict[str, int],
        orders_submitted: int,
        fills: int,
        partial_fills: int,
        cancels: int,
        rejects: int,
        unresolved: int,
        position_snapshot: PositionSnapshot,
        trades_submitted_run: int,
        halted_early: bool,
        halt_reason: str | None,
        max_trades_per_run: int,
    ) -> MicroSessionSummary:
        max_daily_gross = int(round(self.settings.micro_max_daily_gross_exposure_dollars * 100))
        max_daily_loss = int(round(self.settings.micro_max_daily_realized_loss_dollars * 100))
        gross_util = (
            position_snapshot.daily_gross_exposure_cents / max_daily_gross
            if max_daily_gross > 0
            else 0.0
        )
        realized_loss = max(0, -position_snapshot.daily_realized_pnl_cents)
        loss_util = realized_loss / max_daily_loss if max_daily_loss > 0 else 0.0

        return MicroSessionSummary(
            cycles_processed=cycles_processed,
            candidates_seen=candidates_seen,
            trades_allowed=trades_allowed,
            trades_skipped=sum(skip_reasons.values()),
            skip_reasons=skip_reasons,
            orders_submitted=orders_submitted,
            fills=fills,
            partial_fills=partial_fills,
            cancels=cancels,
            rejects=rejects,
            unresolved=unresolved,
            open_positions_count=position_snapshot.open_positions_count,
            realized_pnl_cents=position_snapshot.daily_realized_pnl_cents,
            daily_gross_exposure_cents=position_snapshot.daily_gross_exposure_cents,
            gross_exposure_utilization=round(gross_util, 6),
            realized_loss_utilization=round(loss_util, 6),
            trades_per_run_utilization=round(trades_submitted_run / max_trades_per_run, 6),
            trades_per_day_utilization=round(
                position_snapshot.trades_executed_today / self.settings.micro_max_trades_per_day,
                6,
            ),
            halted_early=halted_early,
            halt_reason=halt_reason,
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

    def _default_attempt_id(self, index: int) -> str:
        now = self._now()
        return f"micro-{now:%Y%m%dT%H%M%S}-{index:03d}"

    def _now(self) -> datetime:
        value = self._now_provider()
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)

"""Day 6 cancel-only lifecycle, reconciliation, and CLI tests."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from kalshi_weather_bot import day6_cli
from kalshi_weather_bot.execution.cancel_only import CancelOnlyRunner
from kalshi_weather_bot.execution.lifecycle import OrderLifecycleTracker
from kalshi_weather_bot.execution.live_adapter import OrderAdapterError
from kalshi_weather_bot.execution.live_models import (
    CancelOnlyBatchResult,
    CancelOnlyBatchSummary,
    CancelOnlyOrderIntent,
    KalshiCreateOrderRequest,
    KalshiOrderStatus,
    LocalOrderRecord,
)
from kalshi_weather_bot.journal import JournalWriter


def _settings(**overrides: object) -> SimpleNamespace:
    payload = {
        "execution_mode": "live_cancel_only",
        "allow_live_api": True,
        "cancel_only_enabled": True,
        "cancel_only_max_attempts_per_run": 3,
        "cancel_only_max_qty": 1,
        "cancel_only_min_delay_between_attempts_ms": 0,
        "cancel_only_halt_on_any_fill": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _intent(**overrides: object) -> CancelOnlyOrderIntent:
    payload = {
        "market_id": "KX-WEATHER-TEST",
        "side": "yes",
        "price_cents": 40,
        "quantity": 1,
    }
    payload.update(overrides)
    return CancelOnlyOrderIntent.model_validate(payload)


def _status(
    *,
    order_id: str = "ord-1",
    normalized: str = "open",
    raw: str | None = None,
    requested: int = 1,
    filled: int = 0,
    remaining: int | None = 1,
) -> KalshiOrderStatus:
    return KalshiOrderStatus(
        order_id=order_id,
        status_raw=raw or normalized,
        normalized_status=normalized,  # type: ignore[arg-type]
        requested_quantity=requested,
        filled_quantity=filled,
        remaining_quantity=remaining,
    )


class _StepClock:
    def __init__(self, start: datetime | None = None, step_seconds: float = 0.5) -> None:
        self.current = start or datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        self.step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        value = self.current
        self.current = self.current + self.step
        return value


class _FakeAdapter:
    def __init__(
        self,
        *,
        submit_seq: list[KalshiOrderStatus | Exception],
        cancel_seq: list[KalshiOrderStatus | Exception] | None = None,
        get_seq: list[KalshiOrderStatus | Exception] | None = None,
    ) -> None:
        self.submit_seq = list(submit_seq)
        self.cancel_seq = list(cancel_seq or [])
        self.get_seq = list(get_seq or [])

    def submit_order(
        self,
        intent: CancelOnlyOrderIntent,
        *,
        client_order_id: str | None = None,
    ) -> KalshiOrderStatus:
        item = self.submit_seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def cancel_order(self, order_id: str) -> KalshiOrderStatus:
        item = self.cancel_seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get_order(self, order_id: str) -> KalshiOrderStatus:
        if not self.get_seq:
            return _status(order_id=order_id, normalized="open", raw="open")
        item = self.get_seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_lifecycle_valid_and_invalid_transitions() -> None:
    record = LocalOrderRecord(
        attempt_id="a1",
        market_id="KX",
        side="yes",
        price_cents=40,
        quantity=1,
    )
    tracker = OrderLifecycleTracker(record=record)
    tracker.transition("submit_requested")
    tracker.transition("submit_ack")
    tracker.transition("cancel_requested")
    tracker.transition("cancel_ack")
    tracker.transition("canceled")
    assert record.current_state == "canceled"

    record2 = LocalOrderRecord(
        attempt_id="a2",
        market_id="KX",
        side="yes",
        price_cents=40,
        quantity=1,
    )
    tracker2 = OrderLifecycleTracker(record=record2)
    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        tracker2.transition("cancel_requested")


def test_startup_guard_rejects_unsafe_config_combo() -> None:
    runner = CancelOnlyRunner(
        settings=_settings(execution_mode="dry_run"),
        adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )
    with pytest.raises(ValueError, match="EXECUTION_MODE"):
        runner.run_batch(
            intent=_intent(),
            attempts=1,
            poll_timeout_seconds=1.0,
            poll_interval_seconds=0.2,
            cancel_delay_ms=0,
        )


def test_submit_then_cancel_happy_path() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o1", normalized="open", raw="open")],
        cancel_seq=[_status(order_id="o1", normalized="canceled", raw="canceled")],
        get_seq=[_status(order_id="o1", normalized="canceled", raw="canceled")],
    )
    runner = CancelOnlyRunner(
        settings=_settings(),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(),
        attempts=1,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.2,
        cancel_delay_ms=0,
    )
    summary = result.summary
    assert summary.attempts_executed == 1
    assert summary.submit_ack_count == 1
    assert summary.cancel_success_count == 1
    assert summary.rejected_count == 0
    assert summary.unresolved_count == 0
    assert summary.reconciliation_mismatch_count == 0
    assert summary.unexpected_fill_count == 0


def test_submit_reject_path() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o2", normalized="rejected", raw="rejected")],
    )
    runner = CancelOnlyRunner(
        settings=_settings(),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(),
        attempts=1,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.2,
        cancel_delay_ms=0,
    )
    assert result.summary.rejected_count == 1
    assert result.summary.submit_ack_count == 0


def test_cancel_timeout_reconciliation_unresolved() -> None:
    clock = _StepClock(step_seconds=0.6)
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o3", normalized="open", raw="open")],
        cancel_seq=[
            OrderAdapterError(
                "cancel failed timeout",
                category="network",
                status_code=None,
            )
        ],
        get_seq=[_status(order_id="o3", normalized="open", raw="open")],
    )
    runner = CancelOnlyRunner(
        settings=_settings(),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(),
        attempts=1,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.2,
        cancel_delay_ms=0,
    )
    assert result.summary.unresolved_count == 1


def test_reconciliation_mismatch_and_halt_on_fill() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o4", normalized="open", raw="open")],
        cancel_seq=[_status(order_id="o4", normalized="open", raw="open")],
        get_seq=[_status(order_id="o4", normalized="filled", raw="filled", filled=1, remaining=0)],
    )
    runner = CancelOnlyRunner(
        settings=_settings(cancel_only_halt_on_any_fill=True),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(),
        attempts=2,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.2,
        cancel_delay_ms=0,
    )
    assert result.summary.attempts_executed == 1
    assert result.summary.unexpected_fill_count == 1
    assert result.summary.reconciliation_mismatch_count == 1
    assert result.summary.halted_early is True


def test_unexpected_partial_fill_on_submit_halts() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o5", normalized="partially_filled", raw="partial", filled=1)],
    )
    runner = CancelOnlyRunner(
        settings=_settings(cancel_only_halt_on_any_fill=True),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(),
        attempts=3,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.2,
        cancel_delay_ms=0,
    )
    assert result.summary.attempts_executed == 1
    assert result.summary.unexpected_fill_count == 1
    assert result.summary.halted_early is True


def test_journal_redacts_sensitive_submit_error(tmp_path: Path) -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[
            OrderAdapterError(
                "auth failed Authorization=Bearer secret123 KALSHI-ACCESS-SIGNATURE=abc",
                category="auth",
                status_code=401,
            )
        ],
    )
    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="day6-test",
    )
    runner = CancelOnlyRunner(
        settings=_settings(),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        journal=journal,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    runner.run_batch(
        intent=_intent(),
        attempts=1,
        poll_timeout_seconds=1.0,
        poll_interval_seconds=0.2,
        cancel_delay_ms=0,
    )
    content = journal.events_path.read_text(encoding="utf-8")
    assert "secret123" not in content
    assert "[REDACTED]" in content


def _set_required_env(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("KALSHI_AUTH_MODE", "bearer")
    monkeypatch.setenv("KALSHI_BEARER_TOKEN", "test-token")
    monkeypatch.setenv("NWS_USER_AGENT", "test-agent")
    monkeypatch.setenv("JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("RAW_PAYLOAD_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("WEATHER_RAW_PAYLOAD_DIR", str(tmp_path / "raw_weather"))
    monkeypatch.setenv("DRY_RUN_MODE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "live_cancel_only")
    monkeypatch.setenv("ALLOW_LIVE_API", "true")
    monkeypatch.setenv("CANCEL_ONLY_ENABLED", "true")


def test_day6_cli_offline_smoke_with_mocked_runner(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    _set_required_env(monkeypatch, tmp_path)

    class _FakeRunner:
        def run_batch(
            self,
            *,
            intent: CancelOnlyOrderIntent,
            attempts: int,
            poll_timeout_seconds: float,
            poll_interval_seconds: float,
            cancel_delay_ms: int,
        ) -> CancelOnlyBatchResult:
            del intent, attempts, poll_timeout_seconds, poll_interval_seconds, cancel_delay_ms
            attempt = {
                "attempt_id": "a1",
                "attempt_index": 1,
                "local_record": {
                    "attempt_id": "a1",
                    "market_id": "KX",
                    "side": "yes",
                    "price_cents": 40,
                    "quantity": 1,
                    "current_state": "canceled",
                    "events": [{"state": "submit_ack", "ts": datetime.now(UTC)}],
                },
                "reconciliation": {
                    "attempt_id": "a1",
                    "local_terminal_state": "canceled",
                    "remote_terminal_state": "canceled",
                    "matched": True,
                    "unresolved": False,
                },
            }
            summary = CancelOnlyBatchSummary(
                attempts_requested=1,
                attempts_executed=1,
                submit_ack_count=1,
                cancel_success_count=1,
                rejected_count=0,
                unresolved_count=0,
                reconciliation_mismatch_count=0,
                unexpected_fill_count=0,
                halted_early=False,
            )
            return CancelOnlyBatchResult.model_validate(
                {"attempts": [attempt], "summary": summary.model_dump(mode="json")}
            )

    class _FakeClient:
        def close(self) -> None:
            return

    monkeypatch.setattr(
        day6_cli,
        "_build_runner",
        lambda settings, logger, journal: (_FakeRunner(), _FakeClient()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day6",
            "--market",
            "KX-WEATHER",
            "--side",
            "yes",
            "--price-cents",
            "40",
            "--qty",
            "1",
            "--attempts",
            "1",
        ],
    )
    exit_code = day6_cli.main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "attempts_requested=1" in output
    assert "cancel_success=1" in output


def test_day6_cli_guard_rejects_unsafe_live_flags(monkeypatch: Any, tmp_path: Path) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ALLOW_LIVE_API", "false")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day6",
            "--market",
            "KX-WEATHER",
            "--side",
            "yes",
            "--price-cents",
            "40",
            "--qty",
            "1",
        ],
    )
    exit_code = day6_cli.main()
    assert exit_code == 2


# ── Additional lifecycle state machine tests ──


class TestLifecycleStateMachine:
    """Cover lifecycle edge cases not in original test suite."""

    def test_terminal_states_have_no_successors(self) -> None:
        """All terminal states should have empty allowed-transition sets."""
        from kalshi_weather_bot.execution.lifecycle import (
            _ALLOWED_TRANSITIONS,
            _TERMINAL_STATES,
        )

        for state in _TERMINAL_STATES:
            assert _ALLOWED_TRANSITIONS[state] == set(), f"{state} should have no successors"

    def test_duplicate_transition_is_noop(self) -> None:
        record = LocalOrderRecord(
            attempt_id="dup1", market_id="KX", side="yes", price_cents=40, quantity=1
        )
        tracker = OrderLifecycleTracker(record=record)
        tracker.transition("submit_requested")
        event_count = len(record.events)
        tracker.transition("submit_requested")
        assert len(record.events) == event_count

    def test_terminal_state_blocks_further_transitions(self) -> None:
        record = LocalOrderRecord(
            attempt_id="term1", market_id="KX", side="yes", price_cents=40, quantity=1
        )
        tracker = OrderLifecycleTracker(record=record)
        tracker.transition("submit_requested")
        tracker.transition("submit_rejected")
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            tracker.transition("cancel_requested")

    def test_timestamps_set_on_submit_cancel_terminal(self) -> None:
        clock = _StepClock()
        record = LocalOrderRecord(
            attempt_id="ts1", market_id="KX", side="yes", price_cents=40, quantity=1
        )
        tracker = OrderLifecycleTracker(record=record, now_provider=clock)
        tracker.transition("submit_requested")
        assert record.submit_ts is not None
        tracker.transition("submit_ack")
        tracker.transition("cancel_requested")
        assert record.cancel_ts is not None
        tracker.transition("cancel_ack")
        tracker.transition("canceled")
        assert record.terminal_ts is not None

    def test_submit_ack_to_filled_transition_valid(self) -> None:
        record = LocalOrderRecord(
            attempt_id="fill1", market_id="KX", side="yes", price_cents=40, quantity=1
        )
        tracker = OrderLifecycleTracker(record=record)
        tracker.transition("submit_requested")
        tracker.transition("submit_ack")
        tracker.transition("filled")
        assert record.current_state == "filled"
        assert record.terminal_ts is not None


# ── Additional startup guard tests ──


class TestStartupGuards:
    """Cover all guard rejection paths in CancelOnlyRunner."""

    def _runner(self, **overrides: object) -> CancelOnlyRunner:
        return CancelOnlyRunner(
            settings=_settings(**overrides),
            adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
            logger=logging.getLogger("test"),
        )

    def test_guard_rejects_allow_live_api_false(self) -> None:
        runner = self._runner(allow_live_api=False)
        with pytest.raises(ValueError, match="ALLOW_LIVE_API"):
            runner.run_batch(
                intent=_intent(), attempts=1,
                poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
            )

    def test_guard_rejects_cancel_only_disabled(self) -> None:
        runner = self._runner(cancel_only_enabled=False)
        with pytest.raises(ValueError, match="CANCEL_ONLY_ENABLED"):
            runner.run_batch(
                intent=_intent(), attempts=1,
                poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
            )

    def test_guard_rejects_attempts_over_max(self) -> None:
        runner = self._runner(cancel_only_max_attempts_per_run=2)
        with pytest.raises(ValueError, match="CANCEL_ONLY_MAX_ATTEMPTS_PER_RUN"):
            runner.run_batch(
                intent=_intent(), attempts=3,
                poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
            )

    def test_guard_rejects_qty_over_max(self) -> None:
        runner = self._runner(cancel_only_max_qty=1)
        with pytest.raises(ValueError, match="CANCEL_ONLY_MAX_QTY"):
            runner.run_batch(
                intent=_intent(quantity=5), attempts=1,
                poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
            )

    def test_guard_rejects_negative_cancel_delay(self) -> None:
        runner = self._runner()
        with pytest.raises(ValueError, match="cancel_delay_ms"):
            runner.run_batch(
                intent=_intent(), attempts=1,
                poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=-1,
            )

    def test_guard_rejects_poll_interval_exceeding_timeout(self) -> None:
        runner = self._runner()
        with pytest.raises(ValueError, match="poll_interval_seconds"):
            runner.run_batch(
                intent=_intent(), attempts=1,
                poll_timeout_seconds=1.0, poll_interval_seconds=2.0, cancel_delay_ms=0,
            )


# ── Reconciliation correctness tests ──


class TestReconciliation:
    """Cover reconciliation matching and edge cases."""

    def test_cancel_ack_matches_remote_canceled(self) -> None:
        """cancel_ack local state should match remote canceled (Patch 1)."""
        from kalshi_weather_bot.execution.reconciliation import _match_terminal_states

        matched, reason = _match_terminal_states(
            local_state="cancel_ack", remote_state="canceled"
        )
        assert matched is True
        assert reason is None

    def test_timeout_local_is_mismatch(self) -> None:
        from kalshi_weather_bot.execution.reconciliation import _match_terminal_states

        matched, reason = _match_terminal_states(
            local_state="timeout", remote_state="canceled"
        )
        assert matched is False
        assert reason == "local_timeout"

    def test_submit_rejected_matches_remote_rejected(self) -> None:
        from kalshi_weather_bot.execution.reconciliation import _match_terminal_states

        matched, reason = _match_terminal_states(
            local_state="submit_rejected", remote_state="rejected"
        )
        assert matched is True

    def test_submit_ack_vs_remote_canceled_is_mismatch(self) -> None:
        """Exchange-initiated cancel while we hadn't requested cancel."""
        from kalshi_weather_bot.execution.reconciliation import _match_terminal_states

        matched, reason = _match_terminal_states(
            local_state="submit_ack", remote_state="canceled"
        )
        assert matched is False
        assert "submit_ack" in (reason or "")

    def test_reconciliation_missing_remote_order_id(self) -> None:
        from kalshi_weather_bot.execution.reconciliation import OrderReconciler

        record = LocalOrderRecord(
            attempt_id="no-id",
            market_id="KX",
            side="yes",
            price_cents=40,
            quantity=1,
            remote_order_id=None,
        )
        adapter = _FakeAdapter(submit_seq=[], get_seq=[])
        reconciler = OrderReconciler(adapter=adapter)  # type: ignore[arg-type]
        result = reconciler.reconcile(
            record=record, poll_timeout_seconds=1.0, poll_interval_seconds=0.2
        )
        assert result.unresolved is True
        assert result.mismatch_reason == "missing_remote_order_id"


# ── Normalize status edge cases ──


class TestNormalizeStatus:
    """Cover _normalize_status branches."""

    def test_canceled_variants(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw="canceled", requested_quantity=1, filled_quantity=0, remaining_quantity=1
        ) == "canceled"
        assert _normalize_status(
            status_raw="cancelled", requested_quantity=1, filled_quantity=0, remaining_quantity=1
        ) == "canceled"

    def test_rejected_variants(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw="rejected", requested_quantity=1, filled_quantity=0, remaining_quantity=1
        ) == "rejected"
        assert _normalize_status(
            status_raw="invalid", requested_quantity=1, filled_quantity=0, remaining_quantity=1
        ) == "rejected"

    def test_partial_fill_status(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw="partial_fill", requested_quantity=2, filled_quantity=1, remaining_quantity=1
        ) == "partially_filled"

    def test_fill_status(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw="filled", requested_quantity=1, filled_quantity=1, remaining_quantity=0
        ) == "filled"

    def test_open_variants(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        for raw in ("open", "resting", "live", "active"):
            assert _normalize_status(
                status_raw=raw, requested_quantity=1, filled_quantity=0, remaining_quantity=1
            ) == "open"

    def test_pending_variants(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        for raw in ("pending", "new", "received", "accepted", "queued"):
            assert _normalize_status(
                status_raw=raw, requested_quantity=1, filled_quantity=0, remaining_quantity=1
            ) == "pending"

    def test_infer_filled_from_quantities(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw=None, requested_quantity=2, filled_quantity=2, remaining_quantity=0
        ) == "filled"

    def test_infer_partial_from_quantities(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw=None, requested_quantity=2, filled_quantity=1, remaining_quantity=1
        ) == "partially_filled"

    def test_unknown_status(self) -> None:
        from kalshi_weather_bot.execution.live_adapter import _normalize_status

        assert _normalize_status(
            status_raw="something_weird", requested_quantity=None,
            filled_quantity=0, remaining_quantity=None,
        ) == "unknown"


# ── Full fill on cancel_ack: verify halt and summary ──


def test_full_fill_on_cancel_ack_halts_and_counts() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o6", normalized="open", raw="open")],
        cancel_seq=[
            _status(order_id="o6", normalized="filled", raw="filled", filled=1, remaining=0),
        ],
        get_seq=[
            _status(order_id="o6", normalized="filled", raw="filled", filled=1, remaining=0),
        ],
    )
    runner = CancelOnlyRunner(
        settings=_settings(cancel_only_halt_on_any_fill=True),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(), attempts=3,
        poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
    )
    assert result.summary.attempts_executed == 1
    assert result.summary.unexpected_fill_count == 1
    assert result.summary.halted_early is True
    assert result.summary.halt_reason in {
        "unexpected_fill_on_cancel_ack",
        "unexpected_fill_on_reconciliation",
    }


# ── Multi-attempt batch with delay ──


def test_multi_attempt_batch_executes_sequentially() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[
            _status(order_id="o-a", normalized="open", raw="open"),
            _status(order_id="o-b", normalized="open", raw="open"),
        ],
        cancel_seq=[
            _status(order_id="o-a", normalized="canceled", raw="canceled"),
            _status(order_id="o-b", normalized="canceled", raw="canceled"),
        ],
        get_seq=[
            _status(order_id="o-a", normalized="canceled", raw="canceled"),
            _status(order_id="o-b", normalized="canceled", raw="canceled"),
        ],
    )
    runner = CancelOnlyRunner(
        settings=_settings(cancel_only_max_attempts_per_run=3),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(), attempts=2,
        poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
    )
    assert result.summary.attempts_executed == 2
    assert result.summary.cancel_success_count == 2
    assert result.summary.submit_ack_count == 2


# ── Submit network error (OrderAdapterError) produces submit_rejected ──


def test_submit_network_error_records_rejection() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[
            OrderAdapterError("Connection refused", category="network", status_code=None)
        ],
    )
    runner = CancelOnlyRunner(
        settings=_settings(),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_batch(
        intent=_intent(), attempts=1,
        poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
    )
    assert result.summary.rejected_count == 1
    record = result.attempts[0].local_record
    assert record.current_state == "submit_rejected"
    assert record.error_category == "network"


# ── KalshiOrderStatus has_fill and is_terminal properties ──


class TestKalshiOrderStatusProperties:
    """Cover has_fill and is_terminal property edge cases."""

    def test_has_fill_true_when_filled_quantity(self) -> None:
        s = _status(normalized="open", filled=1)
        assert s.has_fill is True

    def test_has_fill_true_when_status_filled(self) -> None:
        s = _status(normalized="filled", filled=0)
        assert s.has_fill is True

    def test_has_fill_false_when_zero_and_open(self) -> None:
        s = _status(normalized="open", filled=0)
        assert s.has_fill is False

    def test_is_terminal_for_all_terminal_statuses(self) -> None:
        for status in ("canceled", "rejected", "partially_filled", "filled"):
            s = _status(normalized=status)
            assert s.is_terminal is True, f"{status} should be terminal"

    def test_is_not_terminal_for_open(self) -> None:
        assert _status(normalized="open").is_terminal is False

    def test_is_not_terminal_for_pending(self) -> None:
        assert _status(normalized="pending").is_terminal is False

    def test_is_not_terminal_for_unknown(self) -> None:
        assert _status(normalized="unknown").is_terminal is False


# ── CLI input validation tests ──


class TestDay6CLIInputValidation:
    """Cover CLI argument validation from Patch 3."""

    def test_cli_rejects_price_below_1(self, monkeypatch: Any, tmp_path: Path) -> None:
        _set_required_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["kalshi-weather-day6", "--market", "KX", "--side", "yes",
             "--price-cents", "0", "--qty", "1"],
        )
        assert day6_cli.main() == 2

    def test_cli_rejects_price_above_99(self, monkeypatch: Any, tmp_path: Path) -> None:
        _set_required_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["kalshi-weather-day6", "--market", "KX", "--side", "yes",
             "--price-cents", "100", "--qty", "1"],
        )
        assert day6_cli.main() == 2

    def test_cli_rejects_qty_zero(self, monkeypatch: Any, tmp_path: Path) -> None:
        _set_required_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["kalshi-weather-day6", "--market", "KX", "--side", "yes",
             "--price-cents", "40", "--qty", "0"],
        )
        assert day6_cli.main() == 2

    def test_cli_rejects_negative_attempts(self, monkeypatch: Any, tmp_path: Path) -> None:
        _set_required_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["kalshi-weather-day6", "--market", "KX", "--side", "yes",
             "--price-cents", "40", "--qty", "1", "--attempts", "-1"],
        )
        assert day6_cli.main() == 2


# ── Journal event severity for fills (Patch 2) ──


def test_fill_journal_events_include_critical_severity(tmp_path: Path) -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[
            _status(order_id="osev", normalized="partially_filled", raw="partial", filled=1),
        ],
    )
    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="day6-sev",
    )
    runner = CancelOnlyRunner(
        settings=_settings(),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        journal=journal,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    runner.run_batch(
        intent=_intent(), attempts=1,
        poll_timeout_seconds=1.0, poll_interval_seconds=0.2, cancel_delay_ms=0,
    )
    content = journal.events_path.read_text(encoding="utf-8")
    assert '"severity": "CRITICAL"' in content
    assert "order_unexpected_fill_detected" in content


# ── Pydantic model validation tests ──


class TestModelValidation:
    """Cover Pydantic model constraints."""

    def test_intent_rejects_price_zero(self) -> None:
        with pytest.raises(ValidationError):
            CancelOnlyOrderIntent(market_id="KX", side="yes", price_cents=0, quantity=1)

    def test_intent_rejects_price_100(self) -> None:
        with pytest.raises(ValidationError):
            CancelOnlyOrderIntent(market_id="KX", side="yes", price_cents=100, quantity=1)

    def test_intent_rejects_qty_zero(self) -> None:
        with pytest.raises(ValidationError):
            CancelOnlyOrderIntent(market_id="KX", side="yes", price_cents=40, quantity=0)

    def test_intent_accepts_valid_range(self) -> None:
        intent = CancelOnlyOrderIntent(market_id="KX", side="yes", price_cents=50, quantity=1)
        assert intent.price_cents == 50

    def test_intent_accepts_boundary_values(self) -> None:
        low = CancelOnlyOrderIntent(market_id="KX", side="yes", price_cents=1, quantity=1)
        high = CancelOnlyOrderIntent(market_id="KX", side="yes", price_cents=99, quantity=1)
        assert low.price_cents == 1
        assert high.price_cents == 99

    def test_create_order_payload_yes_side_uses_yes_price(self) -> None:
        request = KalshiCreateOrderRequest(
            market_id="KXHIGHNY-26FEB25-T39",
            side="yes",
            price_cents=8,
            quantity=1,
            client_order_id="test-order-1",
        )
        payload = request.to_payload()
        assert payload["action"] == "buy"
        assert payload["ticker"] == "KXHIGHNY-26FEB25-T39"
        assert payload["side"] == "yes"
        assert payload["yes_price"] == 8
        assert "no_price" not in payload
        assert payload["count"] == 1
        assert payload["type"] == "limit"

    def test_create_order_payload_no_side_uses_no_price(self) -> None:
        request = KalshiCreateOrderRequest(
            market_id="KXHIGHNY-26FEB25-T39",
            side="no",
            price_cents=12,
            quantity=1,
            client_order_id="test-order-2",
        )
        payload = request.to_payload()
        assert payload["action"] == "buy"
        assert payload["ticker"] == "KXHIGHNY-26FEB25-T39"
        assert payload["side"] == "no"
        assert payload["no_price"] == 12
        assert "yes_price" not in payload
        assert payload["count"] == 1
        assert payload["type"] == "limit"


# ── No secret leakage in journal payloads ──


def test_no_secret_leakage_in_startup_journal(tmp_path: Path) -> None:
    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="leak-test",
    )
    journal.write_event(
        "day6_cancel_only_startup",
        payload={
            "mode": "live_cancel_only",
            "market": "KX",
            "side": "yes",
            "price_cents": 40,
            "qty": 1,
            "attempts": 1,
        },
    )
    content = journal.events_path.read_text(encoding="utf-8")
    for sensitive in ("Bearer", "KALSHI_API_KEY_SECRET", "KALSHI_BEARER_TOKEN",
                      "Authorization", "SIGNATURE", "-----BEGIN"):
        assert sensitive not in content

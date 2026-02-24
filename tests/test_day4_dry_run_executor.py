"""Day 4 tests for dry-run executor behavior."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from kalshi_weather_bot.execution.dry_run import DryRunExecutor
from kalshi_weather_bot.execution.models import TradeIntent
from kalshi_weather_bot.risk.manager import RiskManager


class _Clock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: int) -> None:
        self._now += timedelta(seconds=seconds)


def _settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "risk_enabled": True,
        "risk_kill_switch": False,
        "risk_min_parser_confidence": 0.75,
        "risk_max_weather_age_seconds": 1800,
        "risk_max_stake_per_trade_cents": 50,
        "risk_max_total_exposure_cents": 500,
        "risk_max_exposure_per_market_cents": 200,
        "risk_max_concurrent_open_orders": 5,
        "risk_duplicate_intent_cooldown_seconds": 60,
        "risk_min_liquidity_contracts": None,
        "risk_max_spread_cents": None,
        "risk_price_min_cents": 1,
        "risk_price_max_cents": 99,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _intent(**overrides: object) -> TradeIntent:
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    payload = {
        "market_id": "KX-WEATHER-TEST",
        "side": "yes",
        "price_cents": 25,
        "quantity": 1,
        "parser_confidence": 0.9,
        "parsed_contract_ref": "parsed-1",
        "weather_snapshot_ref": "weather-1",
        "weather_snapshot_retrieved_at": now - timedelta(seconds=90),
        "timestamp": now,
        "strategy_tag": "test",
        "market_liquidity": 400,
        "market_spread_cents": 2,
    }
    payload.update(overrides)
    return TradeIntent.model_validate(payload)


def _executor(
    clock: _Clock | None = None,
    **settings_overrides: object,
) -> DryRunExecutor:
    settings = _settings(**settings_overrides)
    risk_manager = RiskManager(settings=settings, logger=logging.getLogger("day4-risk"))
    if clock is None:
        clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
    return DryRunExecutor(
        settings=settings,
        risk_manager=risk_manager,
        logger=logging.getLogger("day4-executor"),
        now_provider=clock.now,
        id_provider=lambda prefix, index: f"{prefix}-{index:03d}",
    )


def test_submit_accepted_intent_creates_open_order() -> None:
    executor = _executor()
    result = executor.submit_intent(_intent())
    assert result.status == "accepted"
    assert result.order is not None
    assert result.order.order_id == "dryrun-002"
    open_orders = executor.list_open_dry_run_orders()
    assert len(open_orders) == 1
    assert open_orders[0].intent_id == "intent-001"


def test_rejected_intent_does_not_create_order() -> None:
    executor = _executor(risk_min_parser_confidence=0.95)
    result = executor.submit_intent(_intent(parser_confidence=0.5))
    assert result.status == "rejected"
    assert result.order is None
    assert executor.list_open_dry_run_orders() == []


def test_cancel_open_order_removes_it_from_open_list() -> None:
    executor = _executor()
    accepted = executor.submit_intent(_intent())
    assert accepted.order is not None

    cancelled = executor.cancel_dry_run_order(accepted.order.order_id)
    assert cancelled.status == "cancelled"
    assert cancelled.order is not None
    assert cancelled.order.status == "cancelled"
    assert executor.list_open_dry_run_orders() == []


def test_cancel_missing_order_returns_not_found() -> None:
    executor = _executor()
    result = executor.cancel_dry_run_order("dryrun-999")
    assert result.status == "not_found"
    assert result.decision_code == "order_not_found"


def test_duplicate_intent_suppressed_within_cooldown() -> None:
    clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
    executor = _executor(clock=clock, risk_duplicate_intent_cooldown_seconds=120)

    first = executor.submit_intent(_intent())
    assert first.status == "accepted"
    second = executor.submit_intent(_intent())
    assert second.status == "rejected"
    assert second.risk_decision is not None
    assert "duplicate_intent_cooldown" in second.risk_decision.reasons


def test_duplicate_intent_allowed_after_cooldown() -> None:
    clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
    executor = _executor(
        clock=clock,
        risk_duplicate_intent_cooldown_seconds=30,
        risk_max_concurrent_open_orders=10,
        risk_max_total_exposure_cents=10_000,
        risk_max_exposure_per_market_cents=10_000,
    )

    first = executor.submit_intent(_intent())
    assert first.status == "accepted"
    clock.advance(31)
    second = executor.submit_intent(_intent())
    assert second.status == "accepted"


# --- New tests: exposure tracking, cancel edge cases, kill switch, serialization ---


class TestExposureTracking:
    """Verify exposure snapshot reflects open orders correctly."""

    def test_exposure_increases_after_accepted_order(self) -> None:
        executor = _executor(
            risk_max_total_exposure_cents=10_000,
            risk_max_exposure_per_market_cents=10_000,
            risk_max_concurrent_open_orders=10,
        )
        snap_before = executor.exposure_snapshot()
        assert snap_before.total_open_exposure_cents == 0
        assert snap_before.open_order_count == 0

        executor.submit_intent(_intent(price_cents=25, quantity=2))
        snap_after = executor.exposure_snapshot()
        assert snap_after.total_open_exposure_cents == 50
        assert snap_after.open_order_count == 1
        assert snap_after.exposure_by_market_cents["KX-WEATHER-TEST"] == 50

    def test_exposure_decreases_after_cancel(self) -> None:
        executor = _executor(
            risk_max_total_exposure_cents=10_000,
            risk_max_exposure_per_market_cents=10_000,
            risk_max_concurrent_open_orders=10,
        )
        result = executor.submit_intent(_intent(price_cents=30, quantity=1))
        assert result.order is not None
        executor.cancel_dry_run_order(result.order.order_id)
        snap = executor.exposure_snapshot()
        assert snap.total_open_exposure_cents == 0
        assert snap.open_order_count == 0

    def test_rejected_intent_does_not_change_exposure(self) -> None:
        executor = _executor(risk_min_parser_confidence=0.99)
        executor.submit_intent(_intent(parser_confidence=0.5))
        snap = executor.exposure_snapshot()
        assert snap.total_open_exposure_cents == 0
        assert snap.open_order_count == 0

    def test_multi_market_exposure_tracked_separately(self) -> None:
        executor = _executor(
            risk_max_total_exposure_cents=10_000,
            risk_max_exposure_per_market_cents=10_000,
            risk_max_concurrent_open_orders=10,
            risk_duplicate_intent_cooldown_seconds=0,
        )
        executor.submit_intent(_intent(market_id="MKT-A", price_cents=20, quantity=1))
        executor.submit_intent(_intent(market_id="MKT-B", price_cents=30, quantity=1))
        snap = executor.exposure_snapshot()
        assert snap.total_open_exposure_cents == 50
        assert snap.exposure_by_market_cents["MKT-A"] == 20
        assert snap.exposure_by_market_cents["MKT-B"] == 30


class TestCancelEdgeCases:
    """Double cancel, cancelled_at timestamp."""

    def test_double_cancel_returns_not_found(self) -> None:
        executor = _executor()
        result = executor.submit_intent(_intent())
        assert result.order is not None
        order_id = result.order.order_id

        first_cancel = executor.cancel_dry_run_order(order_id)
        assert first_cancel.status == "cancelled"

        second_cancel = executor.cancel_dry_run_order(order_id)
        assert second_cancel.status == "not_found"

    def test_cancelled_order_has_cancelled_at_timestamp(self) -> None:
        clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
        executor = _executor(clock=clock)
        result = executor.submit_intent(_intent())
        assert result.order is not None
        clock.advance(60)
        cancelled = executor.cancel_dry_run_order(result.order.order_id)
        assert cancelled.order is not None
        assert cancelled.order.cancelled_at is not None
        assert cancelled.order.cancelled_at > result.order.created_at


class TestKillSwitchExecutor:
    """Kill switch at executor level rejects all intents."""

    def test_executor_kill_switch_rejects(self) -> None:
        executor = _executor()
        executor.set_kill_switch(True)
        result = executor.submit_intent(_intent())
        assert result.status == "rejected"
        assert result.risk_decision is not None
        assert "kill_switch_active" in result.risk_decision.reasons

    def test_executor_kill_switch_toggle(self) -> None:
        executor = _executor()
        executor.set_kill_switch(True)
        result1 = executor.submit_intent(_intent())
        assert result1.status == "rejected"

        executor.set_kill_switch(False)
        result2 = executor.submit_intent(_intent())
        assert result2.status == "accepted"


class TestAutoIntentIdGeneration:
    """Verify intent_id is auto-assigned when not provided."""

    def test_auto_generated_intent_id(self) -> None:
        executor = _executor()
        intent = _intent()
        assert intent.intent_id is None  # not pre-set in _intent helper
        result = executor.submit_intent(intent)
        assert result.intent is not None
        assert result.intent.intent_id is not None
        assert result.intent.intent_id.startswith("intent-")

    def test_explicit_intent_id_preserved(self) -> None:
        executor = _executor()
        result = executor.submit_intent(_intent(intent_id="my-intent-42"))
        assert result.intent is not None
        assert result.intent.intent_id == "my-intent-42"


class TestDryRunOrderSerialization:
    """Verify DryRunOrder and ExecutionResult serialize cleanly."""

    def test_accepted_result_serializes_to_json(self) -> None:
        import json

        executor = _executor()
        result = executor.submit_intent(_intent())
        data = result.model_dump(mode="json")
        # Should round-trip through JSON without error
        serialized = json.dumps(data)
        assert "accepted" in serialized
        assert data["order"]["order_id"].startswith("dryrun-")

    def test_rejected_result_serializes_to_json(self) -> None:
        import json

        executor = _executor(risk_min_parser_confidence=0.99)
        result = executor.submit_intent(_intent(parser_confidence=0.1))
        data = result.model_dump(mode="json")
        serialized = json.dumps(data)
        assert "rejected" in serialized
        assert data["order"] is None


class TestListOpenOrders:
    """Verify list_open_dry_run_orders ordering and content."""

    def test_orders_sorted_by_creation_time(self) -> None:
        clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
        executor = _executor(
            clock=clock,
            risk_max_concurrent_open_orders=10,
            risk_max_total_exposure_cents=10_000,
            risk_max_exposure_per_market_cents=10_000,
            risk_duplicate_intent_cooldown_seconds=0,
        )
        executor.submit_intent(_intent(market_id="MKT-A"))
        clock.advance(10)
        executor.submit_intent(_intent(market_id="MKT-B"))
        orders = executor.list_open_dry_run_orders()
        assert len(orders) == 2
        assert orders[0].market_id == "MKT-A"
        assert orders[1].market_id == "MKT-B"
        assert orders[0].created_at < orders[1].created_at

    def test_empty_after_cancel_all(self) -> None:
        executor = _executor(
            risk_max_concurrent_open_orders=10,
            risk_max_total_exposure_cents=10_000,
            risk_max_exposure_per_market_cents=10_000,
            risk_duplicate_intent_cooldown_seconds=0,
        )
        r1 = executor.submit_intent(_intent(market_id="MKT-A"))
        r2 = executor.submit_intent(_intent(market_id="MKT-B"))
        assert r1.order is not None and r2.order is not None
        executor.cancel_dry_run_order(r1.order.order_id)
        executor.cancel_dry_run_order(r2.order.order_id)
        assert executor.list_open_dry_run_orders() == []


class TestJournalingIntegration:
    """Verify journal events are written for accepted and rejected intents."""

    def test_journal_receives_events_on_accept(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json as json_mod

        from kalshi_weather_bot.journal import JournalWriter

        journal = JournalWriter(
            journal_dir=tmp_path / "journal",
            raw_payload_dir=tmp_path / "raw",
            session_id="test-session",
        )
        settings = _settings()
        risk_manager = RiskManager(
            settings=settings, logger=logging.getLogger("j-risk"),
        )
        clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
        executor = DryRunExecutor(
            settings=settings,
            risk_manager=risk_manager,
            logger=logging.getLogger("j-exec"),
            journal=journal,
            now_provider=clock.now,
            id_provider=lambda prefix, idx: f"{prefix}-{idx:03d}",
        )
        executor.submit_intent(_intent())
        journal_files = list((tmp_path / "journal").glob("*.jsonl"))
        assert journal_files
        lines = journal_files[0].read_text("utf-8").strip().splitlines()
        event_types = [json_mod.loads(line)["event_type"] for line in lines]
        assert "risk_evaluation" in event_types
        assert "dry_run_order_submitted" in event_types

    def test_journal_receives_events_on_reject(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json as json_mod

        from kalshi_weather_bot.journal import JournalWriter

        journal = JournalWriter(
            journal_dir=tmp_path / "journal",
            raw_payload_dir=tmp_path / "raw",
            session_id="test-session",
        )
        settings = _settings(risk_min_parser_confidence=0.99)
        risk_manager = RiskManager(
            settings=settings, logger=logging.getLogger("j-risk"),
        )
        clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
        executor = DryRunExecutor(
            settings=settings,
            risk_manager=risk_manager,
            logger=logging.getLogger("j-exec"),
            journal=journal,
            now_provider=clock.now,
            id_provider=lambda prefix, idx: f"{prefix}-{idx:03d}",
        )
        executor.submit_intent(_intent(parser_confidence=0.1))
        journal_files = list((tmp_path / "journal").glob("*.jsonl"))
        lines = journal_files[0].read_text("utf-8").strip().splitlines()
        event_types = [json_mod.loads(line)["event_type"] for line in lines]
        assert "risk_evaluation" in event_types
        assert "risk_rejection" in event_types

    def test_stake_cents_in_journal_payload(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Verify stake_cents appears in the journaled risk_evaluation event."""
        import json as json_mod

        from kalshi_weather_bot.journal import JournalWriter

        journal = JournalWriter(
            journal_dir=tmp_path / "journal",
            raw_payload_dir=tmp_path / "raw",
            session_id="test-session",
        )
        settings = _settings()
        risk_manager = RiskManager(
            settings=settings, logger=logging.getLogger("j-risk"),
        )
        clock = _Clock(datetime(2026, 2, 24, 12, 0, tzinfo=UTC))
        executor = DryRunExecutor(
            settings=settings,
            risk_manager=risk_manager,
            logger=logging.getLogger("j-exec"),
            journal=journal,
            now_provider=clock.now,
            id_provider=lambda prefix, idx: f"{prefix}-{idx:03d}",
        )
        executor.submit_intent(_intent(price_cents=30, quantity=2))
        journal_files = list((tmp_path / "journal").glob("*.jsonl"))
        lines = journal_files[0].read_text("utf-8").strip().splitlines()
        for line in lines:
            event = json_mod.loads(line)
            if event["event_type"] == "risk_evaluation":
                assert event["payload"]["intent"]["stake_cents"] == 60
                break
        else:
            raise AssertionError("risk_evaluation event not found")


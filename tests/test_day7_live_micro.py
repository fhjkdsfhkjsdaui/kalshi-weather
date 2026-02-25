"""Day 7 live_micro policy, runner, and CLI tests."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kalshi_weather_bot import day7_cli
from kalshi_weather_bot.config import load_settings
from kalshi_weather_bot.contracts.models import ParsedWeatherContract
from kalshi_weather_bot.exceptions import KalshiAPIError
from kalshi_weather_bot.execution.live_adapter import OrderAdapterError
from kalshi_weather_bot.execution.live_micro import LiveMicroRunner
from kalshi_weather_bot.execution.live_models import KalshiOrderStatus
from kalshi_weather_bot.execution.micro_models import (
    MicroSessionResult,
    MicroSessionSummary,
    MicroTradeCandidate,
    PositionSnapshot,
)
from kalshi_weather_bot.execution.micro_policy import MicroTradePolicy
from kalshi_weather_bot.execution.position_tracker import MicroPositionTracker
from kalshi_weather_bot.journal import JournalWriter


def _settings(**overrides: object) -> SimpleNamespace:
    payload = {
        "execution_mode": "live_micro",
        "allow_live_api": True,
        "allow_live_fills": True,
        "micro_mode_enabled": True,
        "micro_require_supervised_mode": True,
        "micro_max_notional_per_trade_dollars": 0.50,
        "micro_max_trades_per_run": 1,
        "micro_max_trades_per_day": 3,
        "micro_max_open_positions": 1,
        "micro_max_daily_gross_exposure_dollars": 5.0,
        "micro_max_daily_realized_loss_dollars": 2.0,
        "micro_min_seconds_between_trades": 0,
        "micro_halt_on_reconciliation_mismatch": True,
        "micro_halt_on_any_unexpected_fill_state": True,
        "micro_min_parser_confidence": 0.8,
        "micro_min_edge": 0.05,
        "micro_max_weather_age_seconds": 900,
        "micro_poll_timeout_seconds": 2.0,
        "micro_poll_interval_seconds": 0.1,
        "micro_market_scope_whitelist": None,
        "risk_min_liquidity_contracts": None,
        "risk_max_spread_cents": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _candidate(**overrides: object) -> MicroTradeCandidate:
    payload = {
        "market_id": "KX-WEATHER-D7",
        "ticker": "KXWXD7",
        "title": "Will high temp exceed 70F in Seattle?",
        "side": "yes",
        "price_cents": 40,
        "quantity": 1,
        "parser_confidence": 0.9,
        "edge_after_buffers": 0.12,
        "weather_age_seconds": 60,
        "parsed_contract_ref": "parsed-d7",
        "weather_snapshot_ref": "weather-d7",
        "market_liquidity": 200,
        "market_spread_cents": 2,
        "location_hint": "Seattle, WA",
    }
    payload.update(overrides)
    return MicroTradeCandidate.model_validate(payload)


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
    def __init__(self, start: datetime | None = None, step_seconds: float = 0.2) -> None:
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
        intent: object,
        *,
        client_order_id: str | None = None,
    ) -> KalshiOrderStatus:
        del intent, client_order_id
        item = self.submit_seq.pop(0)
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

    def cancel_order(self, order_id: str) -> KalshiOrderStatus:
        if not self.cancel_seq:
            return _status(order_id=order_id, normalized="open", raw="open")
        item = self.cancel_seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_startup_guard_rejects_allow_live_fills_false() -> None:
    runner = LiveMicroRunner(
        settings=_settings(allow_live_fills=False),
        adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )
    with pytest.raises(ValueError, match="ALLOW_LIVE_FILLS"):
        runner.run_candidates(candidates=[_candidate()], supervised=True)


def test_startup_guard_rejects_unsupervised_when_required() -> None:
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=True),
        adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )
    with pytest.raises(ValueError, match="MICRO_REQUIRE_SUPERVISED_MODE"):
        runner.run_candidates(candidates=[_candidate()], supervised=False)


def test_policy_rejects_parser_confidence_edge_stale() -> None:
    settings = _settings()
    policy = MicroTradePolicy(settings=settings, logger=logging.getLogger("test"))  # type: ignore[arg-type]
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    context = SimpleNamespace(
        now=now,
        today=now.date(),
        trades_executed_run=0,
        trades_executed_today=0,
        open_positions_count=0,
        daily_gross_exposure_cents=0,
        daily_realized_pnl_cents=0,
        last_trade_ts=None,
    )
    decision = policy.evaluate(
        _candidate(parser_confidence=0.4, edge_after_buffers=0.01, weather_age_seconds=7200),
        context,  # type: ignore[arg-type]
        max_trades_per_run=1,
    )
    assert decision.allowed is False
    assert "parser_confidence_below_micro_min" in decision.reasons
    assert "edge_below_micro_min" in decision.reasons
    assert "weather_staleness_exceeds_micro_max" in decision.reasons


def test_policy_rejects_cooldown_and_trade_caps() -> None:
    settings = _settings(micro_min_seconds_between_trades=60)
    policy = MicroTradePolicy(settings=settings, logger=logging.getLogger("test"))  # type: ignore[arg-type]
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    context = SimpleNamespace(
        now=now,
        today=now.date(),
        trades_executed_run=1,
        trades_executed_today=3,
        open_positions_count=1,
        daily_gross_exposure_cents=10,
        daily_realized_pnl_cents=0,
        last_trade_ts=now - timedelta(seconds=10),
    )
    decision = policy.evaluate(_candidate(), context, max_trades_per_run=1)  # type: ignore[arg-type]
    assert decision.allowed is False
    assert "micro_max_trades_per_run_hit" in decision.reasons
    assert "micro_max_trades_per_day_hit" in decision.reasons
    assert "micro_max_open_positions_hit" in decision.reasons
    assert "micro_trade_cooldown_active" in decision.reasons


def test_policy_rejects_daily_loss_cap() -> None:
    settings = _settings(micro_max_daily_realized_loss_dollars=0.05)
    policy = MicroTradePolicy(settings=settings, logger=logging.getLogger("test"))  # type: ignore[arg-type]
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    context = SimpleNamespace(
        now=now,
        today=now.date(),
        trades_executed_run=0,
        trades_executed_today=0,
        open_positions_count=0,
        daily_gross_exposure_cents=0,
        daily_realized_pnl_cents=-6,
        last_trade_ts=None,
    )
    decision = policy.evaluate(_candidate(), context, max_trades_per_run=1)  # type: ignore[arg-type]
    assert decision.allowed is False
    assert "micro_max_daily_realized_loss_hit" in decision.reasons


def test_live_micro_updates_positions_after_confirmed_fill() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o1", normalized="open", raw="open")],
        get_seq=[_status(order_id="o1", normalized="filled", raw="filled", filled=1, remaining=0)],
    )
    tracker = MicroPositionTracker(now_provider=clock)
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        position_tracker=tracker,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(candidates=[_candidate()], supervised=True)
    assert result.summary.fills == 1
    assert result.position_snapshot.open_positions_count == 1
    assert result.attempts[0].outcome == "filled"


def test_live_micro_partial_fill_records_partial_position() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o2", normalized="open", raw="open", requested=2)],
        get_seq=[
            _status(
                order_id="o2",
                normalized="partially_filled",
                raw="partial_fill",
                requested=2,
                filled=1,
                remaining=1,
            )
        ],
    )
    tracker = MicroPositionTracker(now_provider=clock)
    runner = LiveMicroRunner(
        settings=_settings(
            micro_require_supervised_mode=False,
            micro_max_notional_per_trade_dollars=1.0,
        ),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        position_tracker=tracker,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate(quantity=2, price_cents=45)],
        supervised=True,
        max_trades_this_run=1,
    )
    assert result.summary.partial_fills == 1
    assert result.position_snapshot.open_positions_count == 1
    assert result.position_snapshot.open_positions[0].quantity == 1


def test_reconciliation_mismatch_halts_session() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o3", normalized="open", raw="open")],
        get_seq=[_status(order_id="o3", normalized="canceled", raw="canceled")],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate(), _candidate(market_id="KX-WEATHER-D7-2")],
        supervised=True,
        max_trades_this_run=1,
    )
    assert result.summary.halted_early is True
    assert "reconciliation_mismatch" in (result.summary.halt_reason or "")
    assert result.summary.candidates_seen == 2


def test_per_day_submission_cap_blocks_second_attempt() -> None:
    clock = _StepClock()
    tracker = MicroPositionTracker(now_provider=clock)
    tracker.record_trade_submission(timestamp=clock())
    settings = _settings(micro_max_trades_per_day=1, micro_require_supervised_mode=False)
    policy = MicroTradePolicy(settings=settings, logger=logging.getLogger("test"))  # type: ignore[arg-type]
    context = tracker.policy_context(now=clock(), trades_executed_run=0)
    decision = policy.evaluate(_candidate(), context, max_trades_per_run=2)
    assert decision.allowed is False
    assert "micro_max_trades_per_day_hit" in decision.reasons


def test_market_scope_whitelist_blocks_out_of_scope_market() -> None:
    settings = _settings(micro_market_scope_whitelist="denver", micro_require_supervised_mode=False)
    policy = MicroTradePolicy(settings=settings, logger=logging.getLogger("test"))  # type: ignore[arg-type]
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    context = SimpleNamespace(
        now=now,
        today=now.date(),
        trades_executed_run=0,
        trades_executed_today=0,
        open_positions_count=0,
        daily_gross_exposure_cents=0,
        daily_realized_pnl_cents=0,
        last_trade_ts=None,
    )
    decision = policy.evaluate(
        _candidate(location_hint="Seattle, WA"),
        context,  # type: ignore[arg-type]
        max_trades_per_run=1,
    )
    assert decision.allowed is False
    assert "market_scope_not_allowed" in decision.reasons


def test_journal_redacts_sensitive_submit_error(tmp_path: Path) -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[
            OrderAdapterError(
                "auth failed Authorization=Bearer secret123 KALSHI-ACCESS-SIGNATURE=abc",
                category="auth",
                status_code=401,
            )
        ]
    )
    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="day7-redact",
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        journal=journal,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    runner.run_candidates(candidates=[_candidate()], supervised=True)
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
    monkeypatch.setenv("EXECUTION_MODE", "live_micro")
    monkeypatch.setenv("ALLOW_LIVE_API", "true")
    monkeypatch.setenv("ALLOW_LIVE_FILLS", "true")
    monkeypatch.setenv("MICRO_MODE_ENABLED", "true")


def test_day7_cli_offline_smoke_with_mocked_runner(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    _set_required_env(monkeypatch, tmp_path)

    fake_candidate = _candidate()

    def _fake_eval(*, args: Any, settings: Any, logger: Any, journal: Any) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [fake_candidate], [], {
            "scanned": 1,
            "weather_candidates": 1,
            "matched": 1,
            "estimated": 1,
            "filtered": 0,
            "selected": 1,
        }

    class _FakeRunner:
        def run_candidates(self, **kwargs: Any) -> MicroSessionResult:
            del kwargs
            snapshot = PositionSnapshot(
                open_positions=[],
                open_positions_count=0,
                daily_gross_exposure_cents=40,
                daily_realized_pnl_cents=0,
                trades_executed_today=1,
                last_trade_ts=datetime.now(UTC),
            )
            summary = MicroSessionSummary(
                cycles_processed=1,
                candidates_seen=1,
                trades_allowed=1,
                trades_skipped=0,
                skip_reasons={},
                orders_submitted=1,
                fills=1,
                partial_fills=0,
                cancels=0,
                rejects=0,
                unresolved=0,
                open_positions_count=0,
                realized_pnl_cents=0,
                daily_gross_exposure_cents=40,
                gross_exposure_utilization=0.08,
                realized_loss_utilization=0.0,
                trades_per_run_utilization=1.0,
                trades_per_day_utilization=0.333333,
                halted_early=False,
            )
            return MicroSessionResult(attempts=[], position_snapshot=snapshot, summary=summary)

    class _FakeClient:
        def close(self) -> None:
            return

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(
        day7_cli,
        "_build_live_micro_runner",
        lambda settings, logger, journal: (_FakeRunner(), _FakeClient()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--mode",
            "live_micro",
            "--supervised",
            "--max-cycles",
            "1",
            "--max-trades-this-run",
            "1",
        ],
    )

    exit_code = day7_cli.main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "mode=live_micro" in output
    assert "orders_submitted=1" in output
    assert "fills=1" in output


def test_day6_config_backwards_compat_without_day7_env(monkeypatch: Any, tmp_path: Path) -> None:
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

    settings = load_settings()
    assert settings.execution_mode == "live_cancel_only"
    assert settings.allow_live_fills is False


# --- Startup guard exhaustive tests ---


class TestStartupGuards:
    """Verify every startup guard path in LiveMicroRunner."""

    def _runner(self, **overrides: object) -> LiveMicroRunner:
        return LiveMicroRunner(
            settings=_settings(**overrides),
            adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
            logger=logging.getLogger("test"),
            sleep_fn=lambda _: None,
        )

    def test_rejects_wrong_execution_mode(self) -> None:
        with pytest.raises(ValueError, match="EXECUTION_MODE"):
            self._runner(execution_mode="dry_run").run_candidates(
                candidates=[], supervised=True
            )

    def test_rejects_allow_live_api_false(self) -> None:
        with pytest.raises(ValueError, match="ALLOW_LIVE_API"):
            self._runner(allow_live_api=False).run_candidates(
                candidates=[], supervised=True
            )

    def test_rejects_micro_mode_enabled_false(self) -> None:
        with pytest.raises(ValueError, match="MICRO_MODE_ENABLED"):
            self._runner(micro_mode_enabled=False).run_candidates(
                candidates=[], supervised=True
            )

    def test_rejects_max_trades_exceeding_config(self) -> None:
        with pytest.raises(ValueError, match="max_trades_this_run exceeds"):
            self._runner(micro_max_trades_per_run=1).run_candidates(
                candidates=[], supervised=True, max_trades_this_run=5
            )

    def test_rejects_poll_interval_exceeding_timeout(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds cannot exceed"):
            self._runner().run_candidates(
                candidates=[],
                supervised=True,
                poll_timeout_seconds=1.0,
                poll_interval_seconds=5.0,
            )

    def test_rejects_zero_poll_timeout(self) -> None:
        with pytest.raises(ValueError, match="poll_timeout_seconds must be > 0"):
            self._runner().run_candidates(
                candidates=[], supervised=True, poll_timeout_seconds=0.0
            )

    def test_rejects_zero_poll_interval(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
            self._runner().run_candidates(
                candidates=[], supervised=True, poll_interval_seconds=0.0
            )


# --- Parameter resolution (None vs 0) ---


class TestParameterResolution:
    """Verify None falls back to settings, but explicit 0 raises."""

    def test_none_max_trades_uses_settings_default(self) -> None:
        clock = _StepClock()
        runner = LiveMicroRunner(
            settings=_settings(
                micro_require_supervised_mode=False,
                micro_max_trades_per_run=2,
            ),
            adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
            logger=logging.getLogger("test"),
            now_provider=clock,
            sleep_fn=lambda _: None,
        )
        result = runner.run_candidates(
            candidates=[], supervised=True, max_trades_this_run=None
        )
        assert result.summary.candidates_seen == 0

    def test_zero_max_trades_raises(self) -> None:
        runner = LiveMicroRunner(
            settings=_settings(micro_require_supervised_mode=False),
            adapter=_FakeAdapter(submit_seq=[]),  # type: ignore[arg-type]
            logger=logging.getLogger("test"),
            sleep_fn=lambda _: None,
        )
        with pytest.raises(ValueError, match="max_trades_this_run must be > 0"):
            runner.run_candidates(
                candidates=[], supervised=True, max_trades_this_run=0
            )


# --- Submit error path ---


def test_submit_network_error_records_rejected() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[
            OrderAdapterError("connection reset", category="network", status_code=None)
        ]
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(candidates=[_candidate()], supervised=True)
    assert result.summary.rejects == 1
    assert result.attempts[0].outcome == "rejected"
    assert "submit_error" in result.attempts[0].reasons


# --- Poll timeout → unresolved ---


def test_poll_timeout_returns_unresolved_and_halts() -> None:
    clock = _StepClock(step_seconds=5.0)
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-timeout", normalized="open")],
        get_seq=[
            _status(order_id="o-timeout", normalized="open"),
            _status(order_id="o-timeout", normalized="open"),
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate()],
        supervised=True,
        poll_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )
    assert result.attempts[0].outcome == "unresolved"
    assert result.attempts[0].unresolved is True
    assert result.summary.halted_early is True


def test_poll_timeout_then_cancel_resolves_to_canceled() -> None:
    clock = _StepClock(step_seconds=5.0)
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-timeout-cancel", normalized="open")],
        cancel_seq=[_status(order_id="o-timeout-cancel", normalized="canceled", raw="canceled")],
        get_seq=[
            _status(order_id="o-timeout-cancel", normalized="open"),
            _status(order_id="o-timeout-cancel", normalized="open"),
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate()],
        supervised=True,
        poll_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )
    assert result.attempts[0].outcome == "canceled"
    assert result.attempts[0].unresolved is False
    assert result.summary.cancels == 1
    assert result.summary.halted_early is False


# --- Position tracker opposite-side pairing ---


class TestPositionTracker:
    """Fill-confirmed position tracking edge cases."""

    def test_opposite_side_pairing_realized_pnl(self) -> None:
        clock = _StepClock()
        tracker = MicroPositionTracker(now_provider=clock)
        tracker.apply_confirmed_fill(
            market_id="MKT-A",
            side="yes",
            price_cents=40,
            filled_quantity=2,
            order_id="o-open",
        )
        assert tracker.snapshot(now=clock()).open_positions_count == 1
        update = tracker.apply_confirmed_fill(
            market_id="MKT-A",
            side="no",
            price_cents=30,
            filled_quantity=1,
            order_id="o-close",
        )
        assert update.realized_pnl_delta_cents == 30
        snap = tracker.snapshot(now=clock())
        assert snap.open_positions_count == 1
        assert snap.open_positions[0].quantity == 1

    def test_full_close_removes_position(self) -> None:
        clock = _StepClock()
        tracker = MicroPositionTracker(now_provider=clock)
        tracker.apply_confirmed_fill(
            market_id="MKT-B",
            side="yes",
            price_cents=50,
            filled_quantity=1,
            order_id="o1",
        )
        update = tracker.apply_confirmed_fill(
            market_id="MKT-B",
            side="no",
            price_cents=40,
            filled_quantity=1,
            order_id="o2",
        )
        assert update.realized_pnl_delta_cents == 10
        snap = tracker.snapshot(now=clock())
        assert snap.open_positions_count == 0

    def test_weighted_average_entry_on_add(self) -> None:
        clock = _StepClock()
        tracker = MicroPositionTracker(now_provider=clock)
        tracker.apply_confirmed_fill(
            market_id="MKT-C",
            side="yes",
            price_cents=40,
            filled_quantity=1,
            order_id="o1",
        )
        tracker.apply_confirmed_fill(
            market_id="MKT-C",
            side="yes",
            price_cents=60,
            filled_quantity=1,
            order_id="o2",
        )
        snap = tracker.snapshot(now=clock())
        pos = snap.open_positions[0]
        assert pos.quantity == 2
        assert pos.avg_entry_price_cents == pytest.approx(50.0)

    def test_zero_fill_quantity_is_noop(self) -> None:
        clock = _StepClock()
        tracker = MicroPositionTracker(now_provider=clock)
        result = tracker.apply_confirmed_fill(
            market_id="MKT-D",
            side="yes",
            price_cents=50,
            filled_quantity=0,
            order_id="o1",
        )
        assert len(result.events) == 0
        assert tracker.snapshot(now=clock()).open_positions_count == 0

    def test_invalid_side_raises(self) -> None:
        tracker = MicroPositionTracker()
        with pytest.raises(ValueError, match="side must be"):
            tracker.apply_confirmed_fill(
                market_id="MKT",
                side="maybe",
                price_cents=50,
                filled_quantity=1,
                order_id="o1",
            )

    def test_daily_gross_exposure_accumulates_both_sides(self) -> None:
        clock = _StepClock()
        tracker = MicroPositionTracker(now_provider=clock)
        tracker.apply_confirmed_fill(
            market_id="MKT-E",
            side="yes",
            price_cents=40,
            filled_quantity=1,
            order_id="o1",
        )
        tracker.apply_confirmed_fill(
            market_id="MKT-E",
            side="no",
            price_cents=30,
            filled_quantity=1,
            order_id="o2",
        )
        snap = tracker.snapshot(now=clock())
        assert snap.daily_gross_exposure_cents == 70

    def test_record_trade_submission_increments_daily_count(self) -> None:
        clock = _StepClock()
        tracker = MicroPositionTracker(now_provider=clock)
        tracker.record_trade_submission(timestamp=clock())
        tracker.record_trade_submission(timestamp=clock())
        snap = tracker.snapshot(now=clock())
        assert snap.trades_executed_today == 2
        assert snap.last_trade_ts is not None


# --- Policy: notional cap ---


def test_policy_rejects_notional_exceeding_cap() -> None:
    settings = _settings(micro_max_notional_per_trade_dollars=0.30)
    policy = MicroTradePolicy(
        settings=settings, logger=logging.getLogger("test")  # type: ignore[arg-type]
    )
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    context = SimpleNamespace(
        now=now,
        today=now.date(),
        trades_executed_run=0,
        trades_executed_today=0,
        open_positions_count=0,
        daily_gross_exposure_cents=0,
        daily_realized_pnl_cents=0,
        last_trade_ts=None,
    )
    decision = policy.evaluate(
        _candidate(price_cents=40, quantity=1),
        context,  # type: ignore[arg-type]
        max_trades_per_run=1,
    )
    assert decision.allowed is False
    assert "micro_max_notional_per_trade_exceeded" in decision.reasons


# --- Policy: daily gross exposure cap ---


def test_policy_rejects_daily_gross_exposure_exceeded() -> None:
    settings = _settings(micro_max_daily_gross_exposure_dollars=0.50)
    policy = MicroTradePolicy(
        settings=settings, logger=logging.getLogger("test")  # type: ignore[arg-type]
    )
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    context = SimpleNamespace(
        now=now,
        today=now.date(),
        trades_executed_run=0,
        trades_executed_today=0,
        open_positions_count=0,
        daily_gross_exposure_cents=30,
        daily_realized_pnl_cents=0,
        last_trade_ts=None,
    )
    decision = policy.evaluate(
        _candidate(price_cents=40, quantity=1),
        context,  # type: ignore[arg-type]
        max_trades_per_run=1,
    )
    assert decision.allowed is False
    assert "micro_max_daily_gross_exposure_hit" in decision.reasons


# --- MicroTradeCandidate notional property ---


def test_candidate_notional_cents_property() -> None:
    c = _candidate(price_cents=40, quantity=3)
    assert c.notional_cents == 120


# --- Runner: remote rejected after submit_ack ---


def test_remote_rejected_after_submit_ack_is_mismatch() -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-rej", normalized="open")],
        get_seq=[
            _status(
                order_id="o-rej",
                normalized="rejected",
                raw="rejected",
            )
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(candidates=[_candidate()], supervised=True)
    assert result.attempts[0].outcome == "rejected"
    assert result.attempts[0].reconciliation_mismatch is True
    assert result.summary.halted_early is True


# --- Config: micro cross-cap validators ---


class TestMicroConfigCrossCaps:
    """Config validators for micro cross-cap consistency."""

    def test_micro_notional_exceeds_risk_stake(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _set_required_env(monkeypatch, tmp_path)
        monkeypatch.setenv("RISK_MAX_STAKE_PER_TRADE_CENTS", "30")
        monkeypatch.setenv("MICRO_MAX_NOTIONAL_PER_TRADE_DOLLARS", "0.50")
        from pydantic import ValidationError

        with pytest.raises(
            (ValidationError, Exception),
        ):
            load_settings()

    def test_micro_positions_exceeds_risk_concurrent(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _set_required_env(monkeypatch, tmp_path)
        monkeypatch.setenv("RISK_MAX_CONCURRENT_OPEN_ORDERS", "1")
        monkeypatch.setenv("MICRO_MAX_OPEN_POSITIONS", "5")
        from pydantic import ValidationError

        with pytest.raises(
            (ValidationError, Exception),
        ):
            load_settings()


# --- CLI: --dry-run flag forces dry_run mode ---


def test_cli_dry_run_flag_forces_dry_run_mode(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
        }

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day7", "--dry-run", "--max-cycles", "1"],
    )

    exit_code = day7_cli.main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "mode=dry_run" in output


def test_cli_live_micro_mode_overrides_conflicting_config(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")

    captured_mode: dict[str, str] = {}

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
        }

    class _FakeRunner:
        def run_candidates(self, **kwargs: Any) -> MicroSessionResult:
            del kwargs
            snapshot = PositionSnapshot(
                open_positions=[],
                open_positions_count=0,
                daily_gross_exposure_cents=0,
                daily_realized_pnl_cents=0,
                trades_executed_today=0,
                last_trade_ts=None,
            )
            summary = MicroSessionSummary(
                cycles_processed=1,
                candidates_seen=0,
                trades_allowed=0,
                trades_skipped=0,
                skip_reasons={},
                orders_submitted=0,
                fills=0,
                partial_fills=0,
                cancels=0,
                rejects=0,
                unresolved=0,
                open_positions_count=0,
                realized_pnl_cents=0,
                daily_gross_exposure_cents=0,
                gross_exposure_utilization=0.0,
                realized_loss_utilization=0.0,
                trades_per_run_utilization=0.0,
                trades_per_day_utilization=0.0,
                halted_early=False,
            )
            return MicroSessionResult(attempts=[], position_snapshot=snapshot, summary=summary)

    class _FakeClient:
        def close(self) -> None:
            return

    def _fake_build_runner(*, settings: Any, logger: Any, journal: Any) -> tuple[Any, Any]:
        del logger, journal
        captured_mode["execution_mode"] = settings.execution_mode
        return _FakeRunner(), _FakeClient()

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(day7_cli, "_build_live_micro_runner", _fake_build_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--mode",
            "live_micro",
            "--no-dry-run",
            "--supervised",
            "--max-cycles",
            "1",
        ],
    )
    exit_code = day7_cli.main()
    assert exit_code == 0
    assert captured_mode["execution_mode"] == "live_micro"
    output = capsys.readouterr().out
    assert "mode=live_micro" in output


def test_cli_live_micro_mode_overrides_live_cancel_only_config(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "live_cancel_only")
    monkeypatch.setenv("CANCEL_ONLY_ENABLED", "true")

    captured_mode: dict[str, str] = {}

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
        }

    class _FakeRunner:
        def run_candidates(self, **kwargs: Any) -> MicroSessionResult:
            del kwargs
            snapshot = PositionSnapshot(
                open_positions=[],
                open_positions_count=0,
                daily_gross_exposure_cents=0,
                daily_realized_pnl_cents=0,
                trades_executed_today=0,
                last_trade_ts=None,
            )
            summary = MicroSessionSummary(
                cycles_processed=1,
                candidates_seen=0,
                trades_allowed=0,
                trades_skipped=0,
                skip_reasons={},
                orders_submitted=0,
                fills=0,
                partial_fills=0,
                cancels=0,
                rejects=0,
                unresolved=0,
                open_positions_count=0,
                realized_pnl_cents=0,
                daily_gross_exposure_cents=0,
                gross_exposure_utilization=0.0,
                realized_loss_utilization=0.0,
                trades_per_run_utilization=0.0,
                trades_per_day_utilization=0.0,
                halted_early=False,
            )
            return MicroSessionResult(attempts=[], position_snapshot=snapshot, summary=summary)

    class _FakeClient:
        def close(self) -> None:
            return

    def _fake_build_runner(*, settings: Any, logger: Any, journal: Any) -> tuple[Any, Any]:
        del logger, journal
        captured_mode["execution_mode"] = settings.execution_mode
        return _FakeRunner(), _FakeClient()

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(day7_cli, "_build_live_micro_runner", _fake_build_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--mode",
            "live_micro",
            "--no-dry-run",
            "--supervised",
            "--max-cycles",
            "1",
        ],
    )
    exit_code = day7_cli.main()
    assert exit_code == 0
    assert captured_mode["execution_mode"] == "live_micro"
    output = capsys.readouterr().out
    assert "mode=live_micro" in output


def test_cli_live_micro_with_dry_run_conflict_fails_before_side_effects(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")

    called = {"build_runner": False, "evaluate": False}

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        called["evaluate"] = True
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
        }

    def _fake_build_runner(*, settings: Any, logger: Any, journal: Any) -> tuple[Any, Any]:
        del settings, logger, journal
        called["build_runner"] = True
        raise AssertionError("Runner should not be built on startup conflict.")

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(day7_cli, "_build_live_micro_runner", _fake_build_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--mode",
            "live_micro",
            "--dry-run",
            "--supervised",
            "--max-cycles",
            "1",
        ],
    )
    exit_code = day7_cli.main()
    assert exit_code == 2
    assert called["build_runner"] is False
    assert called["evaluate"] is False
    stderr = capsys.readouterr().err.lower()
    assert "config execution_mode=dry_run" in stderr
    assert "conflicting runtime flags" in stderr
    assert "cli --mode=live_micro" in stderr
    assert "use --no-dry-run" in stderr


def test_cli_live_micro_no_dry_run_startup_journal_has_effective_state(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
        }

    class _FakeRunner:
        def run_candidates(self, **kwargs: Any) -> MicroSessionResult:
            del kwargs
            snapshot = PositionSnapshot(
                open_positions=[],
                open_positions_count=0,
                daily_gross_exposure_cents=0,
                daily_realized_pnl_cents=0,
                trades_executed_today=0,
                last_trade_ts=None,
            )
            summary = MicroSessionSummary(
                cycles_processed=1,
                candidates_seen=0,
                trades_allowed=0,
                trades_skipped=0,
                skip_reasons={},
                orders_submitted=0,
                fills=0,
                partial_fills=0,
                cancels=0,
                rejects=0,
                unresolved=0,
                open_positions_count=0,
                realized_pnl_cents=0,
                daily_gross_exposure_cents=0,
                gross_exposure_utilization=0.0,
                realized_loss_utilization=0.0,
                trades_per_run_utilization=0.0,
                trades_per_day_utilization=0.0,
                halted_early=False,
            )
            return MicroSessionResult(attempts=[], position_snapshot=snapshot, summary=summary)

    class _FakeClient:
        def close(self) -> None:
            return

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(
        day7_cli,
        "_build_live_micro_runner",
        lambda settings, logger, journal: (_FakeRunner(), _FakeClient()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--mode",
            "live_micro",
            "--no-dry-run",
            "--supervised",
            "--max-cycles",
            "1",
        ],
    )
    exit_code = day7_cli.main()
    assert exit_code == 0

    journal_files = sorted((tmp_path / "journal").glob("*.jsonl"))
    assert journal_files
    lines = journal_files[0].read_text(encoding="utf-8").splitlines()
    startup_events = [json.loads(line) for line in lines if line.strip()]
    start_event = next(
        item for item in startup_events if item["event_type"] == "micro_session_start"
    )
    payload = start_event["payload"]
    assert payload["config_execution_mode"] == "dry_run"
    assert payload["cli_mode"] == "live_micro"
    assert payload["effective_execution_mode"] == "live_micro"
    assert payload["effective_dry_run"] is False
    assert payload["supervised"] is True


def test_day7_cycle_summary_includes_fetch_diagnostics(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [], [], {
            "scanned": 2,
            "weather_candidates": 1,
            "matched": 1,
            "estimated": 1,
            "filtered": 1,
            "selected": 0,
            "pages_fetched": 3,
            "total_markets_fetched": 450,
            "weather_markets_after_filter": 12,
            "candidates_generated": 0,
        }

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--dry-run",
            "--max-cycles",
            "1",
        ],
    )

    exit_code = day7_cli.main()
    assert exit_code == 0

    journal_files = sorted((tmp_path / "journal").glob("*.jsonl"))
    assert journal_files
    events = [
        json.loads(line)
        for line in journal_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cycle_event = next(
        item for item in events if item["event_type"] == "micro_cycle_signal_summary"
    )
    payload = cycle_event["payload"]
    assert payload["pages_fetched"] == 3
    assert payload["total_markets_fetched"] == 450
    assert payload["weather_markets_after_filter"] == 12
    assert payload["candidates_generated"] == 0


# --- CRITICAL severity journal event for unexpected fill state ---


def test_critical_severity_on_unexpected_fill_state(tmp_path: Path) -> None:
    clock = _StepClock()
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-crit", normalized="open")],
        get_seq=[
            _status(
                order_id="o-crit",
                normalized="filled",
                raw="filled",
                filled=0,
                remaining=0,
            )
        ],
    )
    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="day7-critical",
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        journal=journal,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(candidates=[_candidate()], supervised=True)
    assert result.summary.halted_early is True
    content = journal.events_path.read_text(encoding="utf-8")
    assert '"severity": "CRITICAL"' in content
    assert "micro_unexpected_fill_state" in content


# --- No secret leakage in startup journal ---


def test_no_secret_leakage_in_micro_session_journal(tmp_path: Path) -> None:
    clock = _StepClock()
    journal = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="day7-secrets",
    )
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-sec", normalized="open")],
        get_seq=[
            _status(
                order_id="o-sec",
                normalized="filled",
                filled=1,
                remaining=0,
            )
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        journal=journal,
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    runner.run_candidates(candidates=[_candidate()], supervised=True)
    content = journal.events_path.read_text(encoding="utf-8")
    for secret_pattern in (
        "KALSHI_BEARER_TOKEN",
        "KALSHI_API_KEY_SECRET",
        "bearer ",
    ):
        assert secret_pattern.lower() not in content.lower()


def test_weather_filter_rejection_reason_counting() -> None:
    parsed_non_weather = ParsedWeatherContract(
        provider_market_id="m1",
        raw_title="Will candidate X win?",
        weather_candidate=False,
        parse_confidence=0.1,
        parse_status="rejected",
    )
    parsed_closed_non_weather = ParsedWeatherContract(
        provider_market_id="m2",
        raw_title="Political market",
        weather_candidate=False,
        parse_confidence=0.1,
        parse_status="rejected",
    )
    parsed_weather_parse_fail = ParsedWeatherContract(
        provider_market_id="m3",
        raw_title="NYC temperature above 90F?",
        weather_candidate=True,
        parse_confidence=0.55,
        parse_status="ambiguous",
    )

    reasons_1 = day7_cli._weather_filter_rejection_reasons(
        raw_market={"title": "Will candidate X win?"},
        parsed=parsed_non_weather,
    )
    reasons_2 = day7_cli._weather_filter_rejection_reasons(
        raw_market={
            "title": "Political market",
            "ticker": "POL-1",
            "category": "Politics",
            "status": "closed",
        },
        parsed=parsed_closed_non_weather,
    )
    reasons_3 = day7_cli._weather_filter_rejection_reasons(
        raw_market={
            "title": "NYC temperature above 90F?",
            "ticker": "WX-NYC",
            "category": "Weather",
            "status": "open",
        },
        parsed=parsed_weather_parse_fail,
    )

    assert "missing_category_field" in reasons_1
    assert "missing_title_ticker" in reasons_1
    assert "category_mismatch" in reasons_2
    assert "closed_inactive" in reasons_2
    assert "parse_failure" in reasons_3


def test_day7_cycle_continues_after_market_fetch_timeout(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _set_required_env(monkeypatch, tmp_path)
    calls = {"n": 0}

    def _flaky_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        calls["n"] += 1
        if calls["n"] == 1:
            raise KalshiAPIError("Kalshi API request failed: The read operation timed out")
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
            "pages_fetched": 1,
            "total_markets_fetched": 0,
            "weather_markets_after_filter": 0,
            "candidates_generated": 0,
            "had_more_pages": 0,
            "deduped_count": 0,
        }

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _flaky_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--dry-run",
            "--ui-mode",
            "plain",
            "--max-cycles",
            "2",
        ],
    )

    exit_code = day7_cli.main()
    assert exit_code == 0
    assert calls["n"] == 2

    journal_files = sorted((tmp_path / "journal").glob("*.jsonl"))
    assert journal_files
    events = [
        json.loads(line)
        for line in journal_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_types = [event["event_type"] for event in events]
    assert "micro_cycle_data_failure" in event_types
    summary_event = next(item for item in events if item["event_type"] == "micro_session_summary")
    assert summary_event["payload"]["cycle_data_failures"] == 1


# --- _resolve_runtime_mode tests ---


class TestResolveRuntimeMode:
    """Verify mode precedence, dry-run override, and conflict detection."""

    def _args(self, **overrides: Any) -> SimpleNamespace:
        defaults: dict[str, Any] = {
            "mode": None,
            "dry_run": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _settings_ns(self, **overrides: Any) -> Any:
        """Build a settings-like object for _resolve_runtime_mode."""
        defaults: dict[str, Any] = {
            "execution_mode": "dry_run",
            "allow_live_api": True,
            "allow_live_fills": True,
            "micro_mode_enabled": True,
        }
        defaults.update(overrides)
        ns = _settings(**defaults)

        def _model_copy(update: dict[str, Any] | None = None) -> Any:
            merged = {k: v for k, v in ns.__dict__.items() if k != "model_copy"}
            if update:
                merged.update(update)
            copy = SimpleNamespace(**merged)
            copy.model_copy = _model_copy  # type: ignore[attr-defined]
            return copy

        ns.model_copy = _model_copy  # type: ignore[attr-defined]
        return ns

    def test_default_dry_run_from_config(self) -> None:
        settings, mode, dry_run = day7_cli._resolve_runtime_mode(
            settings=self._settings_ns(execution_mode="dry_run"),
            args=self._args(),
            logger=logging.getLogger("test"),
        )
        assert mode == "dry_run"
        assert dry_run is True

    def test_cli_mode_overrides_config(self) -> None:
        settings, mode, dry_run = day7_cli._resolve_runtime_mode(
            settings=self._settings_ns(execution_mode="dry_run"),
            args=self._args(mode="live_micro", dry_run=False),
            logger=logging.getLogger("test"),
        )
        assert mode == "live_micro"
        assert dry_run is False

    def test_dry_run_flag_forces_dry_run(self) -> None:
        settings, mode, dry_run = day7_cli._resolve_runtime_mode(
            settings=self._settings_ns(execution_mode="dry_run"),
            args=self._args(dry_run=True),
            logger=logging.getLogger("test"),
        )
        assert mode == "dry_run"
        assert dry_run is True

    def test_live_micro_with_dry_run_true_raises(self) -> None:
        with pytest.raises(ValueError, match="Conflicting runtime flags"):
            day7_cli._resolve_runtime_mode(
                settings=self._settings_ns(execution_mode="dry_run"),
                args=self._args(mode="live_micro", dry_run=True),
                logger=logging.getLogger("test"),
            )

    def test_cancel_only_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="cancel-only"):
            day7_cli._resolve_runtime_mode(
                settings=self._settings_ns(execution_mode="live_cancel_only"),
                args=self._args(),
                logger=logging.getLogger("test"),
            )

    def test_no_dry_run_without_live_micro_raises(self) -> None:
        with pytest.raises(ValueError, match="dry_run=false requires"):
            day7_cli._resolve_runtime_mode(
                settings=self._settings_ns(execution_mode="dry_run"),
                args=self._args(mode="dry_run", dry_run=False),
                logger=logging.getLogger("test"),
            )

    def test_settings_updated_with_effective_mode(self) -> None:
        settings, mode, dry_run = day7_cli._resolve_runtime_mode(
            settings=self._settings_ns(execution_mode="live_micro"),
            args=self._args(dry_run=True),
            logger=logging.getLogger("test"),
        )
        assert settings.execution_mode == "dry_run"
        assert mode == "dry_run"
        assert dry_run is True


# --- _weather_near_miss_sample bounds ---


def test_weather_near_miss_sample_truncates_title() -> None:
    parsed = ParsedWeatherContract(
        provider_market_id="m1",
        raw_title="X" * 200,
        weather_candidate=False,
        parse_confidence=0.1,
        parse_status="rejected",
    )
    sample = day7_cli._weather_near_miss_sample(
        raw_market={
            "title": "X" * 200,
            "ticker": "ABC",
            "tags": [f"tag{i}" for i in range(10)],
        },
        parsed=parsed,
        rejection_reasons=["missing_category_field"],
    )
    assert len(sample["title"]) <= 120
    assert len(sample["tags"]) <= 5


def test_weather_filter_returns_empty_for_valid_weather_contract() -> None:
    parsed = ParsedWeatherContract(
        provider_market_id="m1",
        raw_title="NYC high temp above 90?",
        weather_candidate=True,
        parse_confidence=0.9,
        parse_status="parsed",
    )
    reasons = day7_cli._weather_filter_rejection_reasons(
        raw_market={
            "title": "NYC high temp above 90?",
            "ticker": "WX-NYC",
            "category": "Weather",
            "status": "open",
        },
        parsed=parsed,
    )
    assert reasons == []


def test_normalized_weather_age_seconds_clamps_future_timestamp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    future = now + timedelta(seconds=0.6)
    logger = logging.getLogger("test")

    with caplog.at_level(logging.WARNING):
        age = day7_cli._normalized_weather_age_seconds(
            now=now,
            retrieval_timestamp=future,
            logger=logger,
        )

    assert age == 0.0
    assert any("clamping age to 0" in record.message.lower() for record in caplog.records)


def test_normalized_weather_age_seconds_preserves_positive_value() -> None:
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    past = now - timedelta(seconds=123)
    logger = logging.getLogger("test")

    age = day7_cli._normalized_weather_age_seconds(
        now=now,
        retrieval_timestamp=past,
        logger=logger,
    )

    assert age == 123.0


# --- Final reconciliation read after cancel failure ---


def test_final_reconciliation_resolves_canceled_after_cancel_error() -> None:
    """Cancel API call fails, but final get_order shows the order is canceled."""
    clock = _StepClock(step_seconds=5.0)
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-frc", normalized="open")],
        cancel_seq=[
            OrderAdapterError(
                "cancel 404: not found",
                category="validation",
                status_code=404,
            )
        ],
        get_seq=[
            # _final_reconciliation_read returns terminal
            # (poll loop exits without calling get_order because clock jumps past deadline)
            _status(
                order_id="o-frc",
                normalized="canceled",
                raw="canceled",
                filled=0,
                remaining=1,
            ),
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate()],
        supervised=True,
        poll_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )
    assert result.attempts[0].outcome == "canceled"
    assert result.attempts[0].unresolved is False
    assert result.summary.cancels == 1
    assert result.summary.halted_early is False


def test_final_reconciliation_resolves_filled_after_post_cancel_poll_timeout() -> None:
    """Cancel succeeds but returns non-terminal; post-cancel poll times out.
    Final reconciliation read discovers the order filled on the exchange."""
    clock = _StepClock(step_seconds=5.0)
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-frp", normalized="open")],
        cancel_seq=[
            _status(order_id="o-frp", normalized="open", raw="open")
        ],
        get_seq=[
            # _final_reconciliation_read discovers fill
            # (both poll loops exit without calling get_order because clock jumps)
            _status(
                order_id="o-frp",
                normalized="filled",
                raw="filled",
                filled=1,
                remaining=0,
            ),
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate()],
        supervised=True,
        poll_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )
    assert result.attempts[0].outcome == "filled"
    assert result.attempts[0].unresolved is False
    assert result.summary.fills == 1


def test_final_reconciliation_still_unresolved_when_status_non_terminal() -> None:
    """Final reconciliation read also returns non-terminal → stays unresolved + halt."""
    clock = _StepClock(step_seconds=5.0)
    adapter = _FakeAdapter(
        submit_seq=[_status(order_id="o-fru", normalized="open")],
        cancel_seq=[
            OrderAdapterError(
                "cancel 404: not found",
                category="validation",
                status_code=404,
            )
        ],
        get_seq=[
            # _final_reconciliation_read still non-terminal
            # (poll loop exits without calling get_order because clock jumps)
            _status(order_id="o-fru", normalized="open"),
        ],
    )
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate()],
        supervised=True,
        poll_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )
    assert result.attempts[0].outcome == "unresolved"
    assert result.attempts[0].unresolved is True
    assert result.summary.halted_early is True
    assert "reconciliation_unresolved" in (result.summary.halt_reason or "")


def test_final_reconciliation_still_unresolved_when_read_fails() -> None:
    """Final reconciliation read throws → stays unresolved + halt."""
    clock = _StepClock(step_seconds=5.0)

    class _FailingGetAdapter(_FakeAdapter):
        def __init__(self) -> None:
            super().__init__(
                submit_seq=[_status(order_id="o-frf", normalized="open")],
                cancel_seq=[
                    OrderAdapterError(
                        "cancel 404: not found",
                        category="validation",
                        status_code=404,
                    )
                ],
                get_seq=[],
            )

        def get_order(self, order_id: str, **kwargs: Any) -> KalshiOrderStatus:
            # Every get_order call fails — poll loop exits via clock,
            # final reconciliation read also fails
            raise OrderAdapterError(
                "get order 500: internal error",
                category="server",
                status_code=500,
            )

    adapter = _FailingGetAdapter()
    runner = LiveMicroRunner(
        settings=_settings(micro_require_supervised_mode=False),
        adapter=adapter,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        now_provider=clock,
        sleep_fn=lambda _: None,
    )
    result = runner.run_candidates(
        candidates=[_candidate()],
        supervised=True,
        poll_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
    )
    assert result.attempts[0].outcome == "unresolved"
    assert result.attempts[0].unresolved is True
    assert result.summary.halted_early is True

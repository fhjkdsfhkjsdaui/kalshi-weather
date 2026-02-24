"""Day 4 tests for risk manager decision logic."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from kalshi_weather_bot.execution.models import TradeIntent
from kalshi_weather_bot.risk.manager import RiskManager
from kalshi_weather_bot.risk.models import IntentFingerprint, RiskContext


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
        "intent_id": "intent-1",
        "market_id": "KX-WEATHER-TEST",
        "side": "yes",
        "price_cents": 25,
        "quantity": 1,
        "parser_confidence": 0.9,
        "parsed_contract_ref": "parsed-1",
        "weather_snapshot_ref": "weather-1",
        "weather_snapshot_retrieved_at": now - timedelta(seconds=60),
        "timestamp": now,
        "market_liquidity": 500,
        "market_spread_cents": 2,
    }
    payload.update(overrides)
    return TradeIntent.model_validate(payload)


def _context(**overrides: object) -> RiskContext:
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    payload = {
        "now": now,
        "total_open_exposure_cents": 0,
        "exposure_by_market_cents": {},
        "open_order_count": 0,
        "recent_intents": [],
        "kill_switch_active": False,
    }
    payload.update(overrides)
    return RiskContext.model_validate(payload)


def _manager(settings: SimpleNamespace) -> RiskManager:
    return RiskManager(settings=settings, logger=logging.getLogger("test_day4_risk"))


def test_happy_path_approved() -> None:
    manager = _manager(_settings())
    decision = manager.evaluate_intent(_intent(), _context())
    assert decision.allowed is True
    assert decision.decision_code == "approved"
    assert decision.reasons == []
    assert decision.computed_metrics.intent_notional_cents == 25


def test_low_parser_confidence_rejected() -> None:
    manager = _manager(_settings(risk_min_parser_confidence=0.8))
    decision = manager.evaluate_intent(_intent(parser_confidence=0.5), _context())
    assert decision.allowed is False
    assert "parser_confidence_below_min" in decision.reasons


def test_stale_weather_rejected() -> None:
    manager = _manager(_settings(risk_max_weather_age_seconds=120))
    intent = _intent(weather_snapshot_retrieved_at=datetime(2026, 2, 24, 11, 40, tzinfo=UTC))
    decision = manager.evaluate_intent(intent, _context())
    assert decision.allowed is False
    assert "stale_weather_snapshot" in decision.reasons


def test_max_stake_per_trade_rejected() -> None:
    manager = _manager(_settings(risk_max_stake_per_trade_cents=40))
    decision = manager.evaluate_intent(_intent(price_cents=41, quantity=1), _context())
    assert decision.allowed is False
    assert "max_stake_per_trade_exceeded" in decision.reasons


def test_total_exposure_limit_rejected() -> None:
    manager = _manager(_settings(risk_max_total_exposure_cents=60))
    context = _context(total_open_exposure_cents=50)
    decision = manager.evaluate_intent(_intent(price_cents=20, quantity=1), context)
    assert decision.allowed is False
    assert "max_total_exposure_exceeded" in decision.reasons


def test_exposure_per_market_limit_rejected() -> None:
    manager = _manager(_settings(risk_max_exposure_per_market_cents=70))
    context = _context(exposure_by_market_cents={"KX-WEATHER-TEST": 60})
    decision = manager.evaluate_intent(_intent(price_cents=20, quantity=1), context)
    assert decision.allowed is False
    assert "max_market_exposure_exceeded" in decision.reasons


def test_max_concurrent_open_orders_rejected() -> None:
    manager = _manager(_settings(risk_max_concurrent_open_orders=2))
    context = _context(open_order_count=2)
    decision = manager.evaluate_intent(_intent(), context)
    assert decision.allowed is False
    assert "max_concurrent_open_orders_exceeded" in decision.reasons


def test_duplicate_intent_rejected() -> None:
    manager = _manager(_settings(risk_duplicate_intent_cooldown_seconds=120))
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    recent = IntentFingerprint(
        market_id="KX-WEATHER-TEST",
        side="yes",
        price_cents=25,
        quantity=1,
        timestamp=now - timedelta(seconds=30),
    )
    context = _context(now=now, recent_intents=[recent])
    decision = manager.evaluate_intent(_intent(), context)
    assert decision.allowed is False
    assert "duplicate_intent_cooldown" in decision.reasons


def test_kill_switch_rejected() -> None:
    manager = _manager(_settings())
    context = _context(kill_switch_active=True)
    decision = manager.evaluate_intent(_intent(), context)
    assert decision.allowed is False
    assert "kill_switch_active" in decision.reasons


def test_optional_liquidity_and_spread_checks_fail_safe() -> None:
    manager = _manager(
        _settings(
            risk_min_liquidity_contracts=100,
            risk_max_spread_cents=3,
        )
    )
    decision = manager.evaluate_intent(
        _intent(market_liquidity=None, market_spread_cents=None),
        _context(),
    )
    assert decision.allowed is False
    assert "missing_market_liquidity" in decision.reasons
    assert "missing_market_spread" in decision.reasons


def test_price_bounds_rejected() -> None:
    manager = _manager(_settings(risk_price_min_cents=10, risk_price_max_cents=90))
    decision = manager.evaluate_intent(_intent(price_cents=5), _context())
    assert decision.allowed is False
    assert "price_below_min_bound" in decision.reasons


# --- New tests: boundary values, unit consistency, multi-reason, settings kill switch ---


class TestStakeCentsUnitConsistency:
    """Verify stake_cents = price_cents * quantity (all in cents)."""

    def test_stake_single_contract(self) -> None:
        intent = _intent(price_cents=30, quantity=1)
        assert intent.stake_cents == 30

    def test_stake_multiple_contracts(self) -> None:
        intent = _intent(price_cents=40, quantity=3)
        assert intent.stake_cents == 120

    def test_risk_metrics_match_stake(self) -> None:
        manager = _manager(_settings())
        intent = _intent(price_cents=25, quantity=2)
        decision = manager.evaluate_intent(intent, _context())
        assert decision.computed_metrics.intent_notional_cents == 50
        assert decision.computed_metrics.price_cents == 25
        assert decision.computed_metrics.quantity == 2


class TestBoundaryValues:
    """Verify exact-boundary behavior for each limit."""

    def test_stake_exactly_at_limit_is_approved(self) -> None:
        """stake == max_stake_per_trade_cents should pass (not >)."""
        manager = _manager(_settings(risk_max_stake_per_trade_cents=25))
        decision = manager.evaluate_intent(_intent(price_cents=25, quantity=1), _context())
        assert decision.allowed is True

    def test_stake_one_cent_over_limit_is_rejected(self) -> None:
        manager = _manager(_settings(risk_max_stake_per_trade_cents=25))
        decision = manager.evaluate_intent(_intent(price_cents=26, quantity=1), _context())
        assert decision.allowed is False
        assert "max_stake_per_trade_exceeded" in decision.reasons

    def test_total_exposure_exactly_at_limit_is_approved(self) -> None:
        manager = _manager(_settings(risk_max_total_exposure_cents=75))
        context = _context(total_open_exposure_cents=50)
        decision = manager.evaluate_intent(_intent(price_cents=25, quantity=1), context)
        assert decision.allowed is True

    def test_total_exposure_one_cent_over_is_rejected(self) -> None:
        manager = _manager(_settings(risk_max_total_exposure_cents=74))
        context = _context(total_open_exposure_cents=50)
        decision = manager.evaluate_intent(_intent(price_cents=25, quantity=1), context)
        assert decision.allowed is False

    def test_market_exposure_exactly_at_limit_is_approved(self) -> None:
        manager = _manager(_settings(risk_max_exposure_per_market_cents=85))
        context = _context(exposure_by_market_cents={"KX-WEATHER-TEST": 60})
        decision = manager.evaluate_intent(_intent(price_cents=25, quantity=1), context)
        assert decision.allowed is True

    def test_concurrent_orders_exactly_at_limit_approved(self) -> None:
        """open_order_count + 1 == max means rejected (> check)."""
        manager = _manager(_settings(risk_max_concurrent_open_orders=3))
        context = _context(open_order_count=2)
        decision = manager.evaluate_intent(_intent(), context)
        assert decision.allowed is True

    def test_concurrent_orders_one_over_rejected(self) -> None:
        manager = _manager(_settings(risk_max_concurrent_open_orders=3))
        context = _context(open_order_count=3)
        decision = manager.evaluate_intent(_intent(), context)
        assert decision.allowed is False

    def test_parser_confidence_exactly_at_min_is_approved(self) -> None:
        """confidence == min should pass (not < min)."""
        manager = _manager(_settings(risk_min_parser_confidence=0.75))
        decision = manager.evaluate_intent(_intent(parser_confidence=0.75), _context())
        assert decision.allowed is True

    def test_parser_confidence_one_below_min_is_rejected(self) -> None:
        manager = _manager(_settings(risk_min_parser_confidence=0.75))
        decision = manager.evaluate_intent(_intent(parser_confidence=0.74), _context())
        assert decision.allowed is False


class TestMultipleRejectionReasons:
    """Verify that all violated rules accumulate, not just the first."""

    def test_multiple_reasons_all_collected(self) -> None:
        manager = _manager(_settings(
            risk_max_stake_per_trade_cents=10,
            risk_min_parser_confidence=0.95,
        ))
        decision = manager.evaluate_intent(
            _intent(price_cents=25, quantity=1, parser_confidence=0.5),
            _context(),
        )
        assert decision.allowed is False
        assert "parser_confidence_below_min" in decision.reasons
        assert "max_stake_per_trade_exceeded" in decision.reasons
        assert len(decision.reasons) >= 2

    def test_decision_code_is_first_reason(self) -> None:
        """decision_code should be the first rejection reason."""
        manager = _manager(_settings())
        context = _context(kill_switch_active=True)
        decision = manager.evaluate_intent(
            _intent(parser_confidence=0.1), context,
        )
        # kill_switch is checked first, so it should be the decision_code
        assert decision.decision_code == "kill_switch_active"
        assert "parser_confidence_below_min" in decision.reasons


class TestSettingsKillSwitch:
    """Verify the settings-level kill switch (separate from context)."""

    def test_settings_kill_switch_rejects(self) -> None:
        manager = _manager(_settings(risk_kill_switch=True))
        decision = manager.evaluate_intent(_intent(), _context())
        assert decision.allowed is False
        assert "kill_switch_active" in decision.reasons

    def test_kill_switch_overrides_risk_disabled(self) -> None:
        """Even with risk_enabled=False, kill switch should reject."""
        manager = _manager(_settings(risk_enabled=False, risk_kill_switch=True))
        decision = manager.evaluate_intent(_intent(), _context())
        assert decision.allowed is False
        assert "kill_switch_active" in decision.reasons


class TestRiskDisabledPath:
    """Verify behavior when risk_enabled=False."""

    def test_risk_disabled_approves_with_warning(self) -> None:
        manager = _manager(_settings(risk_enabled=False))
        decision = manager.evaluate_intent(_intent(), _context())
        assert decision.allowed is True
        assert decision.decision_code == "risk_disabled"
        assert "risk_checks_disabled" in decision.warnings

    def test_risk_disabled_skips_checks(self) -> None:
        """Should approve even with low confidence when risk is off."""
        manager = _manager(_settings(risk_enabled=False))
        decision = manager.evaluate_intent(
            _intent(parser_confidence=0.1),
            _context(total_open_exposure_cents=99999),
        )
        assert decision.allowed is True


class TestWeatherTimestampEdgeCases:
    """Future timestamps, exact boundary."""

    def test_future_weather_timestamp_warns(self) -> None:
        now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        intent = _intent(weather_snapshot_retrieved_at=now + timedelta(seconds=60))
        manager = _manager(_settings())
        decision = manager.evaluate_intent(intent, _context(now=now))
        assert decision.allowed is True
        assert "weather_snapshot_timestamp_in_future" in decision.warnings
        # weather_age_seconds clamped to 0, so not stale
        assert decision.computed_metrics.weather_age_seconds == 0.0

    def test_weather_age_exactly_at_max_is_approved(self) -> None:
        now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        intent = _intent(
            weather_snapshot_retrieved_at=now - timedelta(seconds=1800),
        )
        manager = _manager(_settings(risk_max_weather_age_seconds=1800))
        decision = manager.evaluate_intent(intent, _context(now=now))
        # age == max → not > max → approved
        assert decision.allowed is True


class TestDuplicateCooldownEdgeCases:
    """Different market/side/price should not trigger duplicate."""

    def test_different_market_not_duplicate(self) -> None:
        manager = _manager(_settings(risk_duplicate_intent_cooldown_seconds=120))
        now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        recent = IntentFingerprint(
            market_id="KX-WEATHER-OTHER",
            side="yes", price_cents=25, quantity=1,
            timestamp=now - timedelta(seconds=30),
        )
        context = _context(now=now, recent_intents=[recent])
        decision = manager.evaluate_intent(_intent(), context)
        assert decision.allowed is True

    def test_different_side_not_duplicate(self) -> None:
        manager = _manager(_settings(risk_duplicate_intent_cooldown_seconds=120))
        now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        recent = IntentFingerprint(
            market_id="KX-WEATHER-TEST",
            side="no", price_cents=25, quantity=1,
            timestamp=now - timedelta(seconds=30),
        )
        context = _context(now=now, recent_intents=[recent])
        decision = manager.evaluate_intent(_intent(side="yes"), context)
        assert decision.allowed is True

    def test_different_price_not_duplicate(self) -> None:
        manager = _manager(_settings(risk_duplicate_intent_cooldown_seconds=120))
        now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
        recent = IntentFingerprint(
            market_id="KX-WEATHER-TEST",
            side="yes", price_cents=30, quantity=1,
            timestamp=now - timedelta(seconds=30),
        )
        context = _context(now=now, recent_intents=[recent])
        decision = manager.evaluate_intent(_intent(price_cents=25), context)
        assert decision.allowed is True


class TestLiquiditySpreadPassPaths:
    """Verify that liquidity/spread checks pass when values are sufficient."""

    def test_sufficient_liquidity_passes(self) -> None:
        manager = _manager(_settings(risk_min_liquidity_contracts=100))
        decision = manager.evaluate_intent(
            _intent(market_liquidity=200, market_spread_cents=2),
            _context(),
        )
        assert decision.allowed is True

    def test_insufficient_liquidity_rejected(self) -> None:
        manager = _manager(_settings(risk_min_liquidity_contracts=100))
        decision = manager.evaluate_intent(
            _intent(market_liquidity=50, market_spread_cents=2),
            _context(),
        )
        assert decision.allowed is False
        assert "insufficient_market_liquidity" in decision.reasons

    def test_spread_above_max_rejected(self) -> None:
        manager = _manager(_settings(risk_max_spread_cents=3))
        decision = manager.evaluate_intent(
            _intent(market_liquidity=500, market_spread_cents=5),
            _context(),
        )
        assert decision.allowed is False
        assert "market_spread_above_limit" in decision.reasons

    def test_no_liquidity_check_when_not_configured(self) -> None:
        """When risk_min_liquidity_contracts=None but liquidity also None, just warn."""
        manager = _manager(_settings(risk_min_liquidity_contracts=None))
        decision = manager.evaluate_intent(
            _intent(market_liquidity=None),
            _context(),
        )
        assert decision.allowed is True
        assert "market_liquidity_unavailable_check_skipped" in decision.warnings

    def test_price_above_max_rejected(self) -> None:
        manager = _manager(_settings(risk_price_min_cents=1, risk_price_max_cents=90))
        decision = manager.evaluate_intent(_intent(price_cents=95), _context())
        assert decision.allowed is False
        assert "price_above_max_bound" in decision.reasons


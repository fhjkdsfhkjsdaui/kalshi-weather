"""Dry-run execution engine for Day 4 safety and plumbing validation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from ..config import Settings
from ..exceptions import JournalError
from ..journal import JournalWriter
from ..redaction import sanitize_text
from ..risk.manager import RiskManager
from ..risk.models import IntentFingerprint, RiskContext
from .models import DryRunOrder, ExecutionResult, ExposureSnapshot, RiskEvaluatedIntent, TradeIntent


class DryRunExecutor:
    """Apply risk checks and persist dry-run order state without live API calls."""

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        logger: logging.Logger,
        journal: JournalWriter | None = None,
        now_provider: Callable[[], datetime] | None = None,
        id_provider: Callable[[str, int], str] | None = None,
    ) -> None:
        self.settings = settings
        self.risk_manager = risk_manager
        self.logger = logger
        self.journal = journal
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        self._id_provider = id_provider or self._default_id_provider
        self._open_orders: dict[str, DryRunOrder] = {}
        self._closed_orders: dict[str, DryRunOrder] = {}
        self._recent_intents: list[IntentFingerprint] = []
        self._id_counter = 0
        self._kill_switch_active = False

    def submit_intent(self, intent: TradeIntent) -> ExecutionResult:
        """Run risk checks and, if allowed, store a dry-run order."""
        now = self._now()
        intent_to_run = self._ensure_intent_id(intent)
        context = self._build_risk_context(now)
        decision = self.risk_manager.evaluate_intent(intent_to_run, context)
        evaluated = RiskEvaluatedIntent(intent=intent_to_run, risk_decision=decision)

        intent_payload = intent_to_run.model_dump(mode="json")
        intent_payload["stake_cents"] = intent_to_run.stake_cents
        self._write_event(
            "risk_evaluation",
            payload={
                "intent": intent_payload,
                "decision": decision.model_dump(mode="json"),
            },
        )

        if not decision.allowed:
            self._write_event(
                "risk_rejection",
                payload={
                    "intent_id": intent_to_run.intent_id,
                    "decision_code": decision.decision_code,
                    "reasons": decision.reasons,
                    "warnings": decision.warnings,
                },
            )
            return ExecutionResult(
                status="rejected",
                decision_code=decision.decision_code,
                reasons=decision.reasons,
                warnings=decision.warnings,
                intent=evaluated.intent,
                risk_decision=evaluated.risk_decision,
            )

        order = DryRunOrder(
            order_id=self._next_id("dryrun"),
            intent_id=intent_to_run.intent_id or self._next_id("intent"),
            market_id=intent_to_run.market_id,
            side=intent_to_run.side,
            price_cents=intent_to_run.price_cents,
            quantity=intent_to_run.quantity,
            stake_cents=intent_to_run.stake_cents,
            status="open",
            created_at=now,
            strategy_tag=intent_to_run.strategy_tag,
        )
        self._open_orders[order.order_id] = order
        self._track_recent_intent(intent_to_run, now)
        self._write_event(
            "dry_run_order_submitted",
            payload={
                "order": order.model_dump(mode="json"),
                "decision_code": decision.decision_code,
            },
        )
        return ExecutionResult(
            status="accepted",
            decision_code=decision.decision_code,
            warnings=decision.warnings,
            intent=evaluated.intent,
            risk_decision=evaluated.risk_decision,
            order=order,
        )

    def cancel_dry_run_order(self, order_id: str) -> ExecutionResult:
        """Cancel an open dry-run order."""
        order = self._open_orders.pop(order_id, None)
        if order is None:
            return ExecutionResult(
                status="not_found",
                decision_code="order_not_found",
                reasons=[f"Order {order_id} is not open."],
            )

        cancelled = order.model_copy(
            update={"status": "cancelled", "cancelled_at": self._now()},
        )
        self._closed_orders[cancelled.order_id] = cancelled
        self._write_event(
            "dry_run_order_cancelled",
            payload={"order": cancelled.model_dump(mode="json")},
        )
        return ExecutionResult(
            status="cancelled",
            decision_code="cancelled",
            order=cancelled,
        )

    def list_open_dry_run_orders(self) -> list[DryRunOrder]:
        """Return current open dry-run orders sorted by creation time."""
        return sorted(self._open_orders.values(), key=lambda order: order.created_at)

    def exposure_snapshot(self) -> ExposureSnapshot:
        """Build the current open exposure snapshot."""
        exposure_by_market: dict[str, int] = {}
        for order in self._open_orders.values():
            exposure_by_market[order.market_id] = (
                exposure_by_market.get(order.market_id, 0) + order.stake_cents
            )
        return ExposureSnapshot(
            total_open_exposure_cents=sum(exposure_by_market.values()),
            exposure_by_market_cents=exposure_by_market,
            open_order_count=len(self._open_orders),
            recent_intents=list(self._recent_intents),
        )

    def set_kill_switch(self, active: bool) -> None:
        """Set executor-level kill switch state."""
        self._kill_switch_active = active

    def _build_risk_context(self, now: datetime) -> RiskContext:
        snapshot = self.exposure_snapshot()
        return RiskContext(
            now=now,
            total_open_exposure_cents=snapshot.total_open_exposure_cents,
            exposure_by_market_cents=snapshot.exposure_by_market_cents,
            open_order_count=snapshot.open_order_count,
            recent_intents=snapshot.recent_intents,
            kill_switch_active=self._kill_switch_active,
        )

    def _track_recent_intent(self, intent: TradeIntent, now: datetime) -> None:
        self._recent_intents.append(
            IntentFingerprint(
                market_id=intent.market_id,
                side=intent.side,
                price_cents=intent.price_cents,
                quantity=intent.quantity,
                timestamp=now,
            )
        )
        cooldown = self.settings.risk_duplicate_intent_cooldown_seconds
        self._recent_intents = [
            finger
            for finger in self._recent_intents
            if (now - finger.timestamp).total_seconds() <= cooldown
        ]

    def _ensure_intent_id(self, intent: TradeIntent) -> TradeIntent:
        if intent.intent_id:
            return intent
        return intent.model_copy(update={"intent_id": self._next_id("intent")})

    def _now(self) -> datetime:
        current = self._now_provider()
        return current.astimezone(UTC) if current.tzinfo else current.replace(tzinfo=UTC)

    def _next_id(self, prefix: str) -> str:
        self._id_counter += 1
        return self._id_provider(prefix, self._id_counter)

    @staticmethod
    def _default_id_provider(prefix: str, index: int) -> str:
        return f"{prefix}-{index:06d}"

    def _write_event(self, event_type: str, payload: dict[str, object]) -> None:
        if self.journal is None:
            return
        try:
            self.journal.write_event(event_type=event_type, payload=payload)
        except JournalError as exc:
            safe_msg = sanitize_text(str(exc))
            self.logger.error("Failed to write %s journal event: %s", event_type, safe_msg)


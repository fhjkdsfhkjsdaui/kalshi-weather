"""Day 7 minimal fill-confirmed position tracking."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from .micro_models import (
    MicroPolicyContext,
    MicroPosition,
    PositionEvent,
    PositionSnapshot,
    PositionUpdateResult,
)


class MicroPositionTracker:
    """Track micro-live positions and daily exposure/PnL counters from confirmed fills."""

    def __init__(self, now_provider: Callable[[], datetime] | None = None) -> None:
        self._positions: dict[str, MicroPosition] = {}
        self._daily_gross_exposure_cents: dict[date, int] = {}
        self._daily_realized_pnl_cents: dict[date, int] = {}
        self._daily_trade_count: dict[date, int] = {}
        self._last_trade_ts: datetime | None = None
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def apply_confirmed_fill(
        self,
        *,
        market_id: str,
        side: str,
        price_cents: int,
        filled_quantity: int,
        order_id: str,
        timestamp: datetime | None = None,
    ) -> PositionUpdateResult:
        """Apply one confirmed fill to position state.

        Assumption: when opposite-side inventory is paired, realized PnL is computed as
        ``100 - (entry_price + opposing_fill_price)`` per matched contract.
        """
        if filled_quantity <= 0:
            return PositionUpdateResult()

        ts = self._ensure_utc(timestamp or self._now_provider())
        side_value = side.lower().strip()
        if side_value not in {"yes", "no"}:
            raise ValueError("side must be 'yes' or 'no'.")

        today = ts.date()
        self._daily_gross_exposure_cents[today] = (
            self._daily_gross_exposure_cents.get(today, 0) + (price_cents * filled_quantity)
        )

        remaining = filled_quantity
        realized_delta = 0
        events: list[PositionEvent] = []

        opposite_side = "no" if side_value == "yes" else "yes"
        opposite_key = self._key(market_id, opposite_side)
        opposite = self._positions.get(opposite_key)
        if opposite is not None and opposite.quantity > 0:
            close_qty = min(opposite.quantity, remaining)
            if close_qty > 0:
                realized_delta += int(
                    round((100 - opposite.avg_entry_price_cents - price_cents) * close_qty)
                )
                new_qty = opposite.quantity - close_qty
                if new_qty <= 0:
                    events.append(
                        PositionEvent(
                            event_type="position_closed",
                            market_id=opposite.market_id,
                            side=opposite.side,
                            quantity=0,
                            avg_entry_price_cents=opposite.avg_entry_price_cents,
                            order_id=order_id,
                            ts=ts,
                            realized_pnl_delta_cents=realized_delta,
                            note="paired_close_assumption",
                        )
                    )
                    self._positions.pop(opposite_key, None)
                else:
                    updated = opposite.model_copy(
                        update={
                            "quantity": new_qty,
                            "updated_at": ts,
                            "last_order_id": order_id,
                        }
                    )
                    self._positions[opposite_key] = updated
                    events.append(
                        PositionEvent(
                            event_type="position_updated",
                            market_id=updated.market_id,
                            side=updated.side,
                            quantity=updated.quantity,
                            avg_entry_price_cents=updated.avg_entry_price_cents,
                            order_id=order_id,
                            ts=ts,
                            realized_pnl_delta_cents=realized_delta,
                            note="paired_close_partial_assumption",
                        )
                    )
                remaining -= close_qty

        if remaining > 0:
            key = self._key(market_id, side_value)
            current = self._positions.get(key)
            if current is None:
                opened = MicroPosition(
                    market_id=market_id,
                    side=side_value,  # type: ignore[arg-type]
                    quantity=remaining,
                    avg_entry_price_cents=float(price_cents),
                    opened_at=ts,
                    updated_at=ts,
                    last_order_id=order_id,
                )
                self._positions[key] = opened
                events.append(
                    PositionEvent(
                        event_type="position_opened",
                        market_id=opened.market_id,
                        side=opened.side,
                        quantity=opened.quantity,
                        avg_entry_price_cents=opened.avg_entry_price_cents,
                        order_id=order_id,
                        ts=ts,
                        note="fill_open",
                    )
                )
            else:
                total_qty = current.quantity + remaining
                weighted_avg = (
                    (current.avg_entry_price_cents * current.quantity) + (price_cents * remaining)
                ) / total_qty
                updated = current.model_copy(
                    update={
                        "quantity": total_qty,
                        "avg_entry_price_cents": weighted_avg,
                        "updated_at": ts,
                        "last_order_id": order_id,
                    }
                )
                self._positions[key] = updated
                events.append(
                    PositionEvent(
                        event_type="position_updated",
                        market_id=updated.market_id,
                        side=updated.side,
                        quantity=updated.quantity,
                        avg_entry_price_cents=updated.avg_entry_price_cents,
                        order_id=order_id,
                        ts=ts,
                        note="fill_add",
                    )
                )

        if realized_delta != 0:
            self._daily_realized_pnl_cents[today] = (
                self._daily_realized_pnl_cents.get(today, 0) + realized_delta
            )

        return PositionUpdateResult(events=events, realized_pnl_delta_cents=realized_delta)

    def record_trade_submission(self, *, timestamp: datetime | None = None) -> None:
        """Record one submitted live_micro order for per-day and cooldown caps."""
        ts = self._ensure_utc(timestamp or self._now_provider())
        today = ts.date()
        self._daily_trade_count[today] = self._daily_trade_count.get(today, 0) + 1
        self._last_trade_ts = ts

    def snapshot(self, *, now: datetime | None = None) -> PositionSnapshot:
        """Return current open positions and daily aggregates."""
        ts = self._ensure_utc(now or self._now_provider())
        today = ts.date()
        open_positions = sorted(
            self._positions.values(),
            key=lambda item: (item.market_id, item.side, item.opened_at),
        )
        return PositionSnapshot(
            open_positions=open_positions,
            open_positions_count=len(open_positions),
            daily_gross_exposure_cents=self._daily_gross_exposure_cents.get(today, 0),
            daily_realized_pnl_cents=self._daily_realized_pnl_cents.get(today, 0),
            trades_executed_today=self._daily_trade_count.get(today, 0),
            last_trade_ts=self._last_trade_ts,
        )

    def policy_context(
        self,
        *,
        now: datetime,
        trades_executed_run: int,
    ) -> MicroPolicyContext:
        """Build policy context using current tracker state."""
        ts = self._ensure_utc(now)
        snap = self.snapshot(now=ts)
        return MicroPolicyContext(
            now=ts,
            today=ts.date(),
            trades_executed_run=trades_executed_run,
            trades_executed_today=snap.trades_executed_today,
            open_positions_count=snap.open_positions_count,
            daily_gross_exposure_cents=snap.daily_gross_exposure_cents,
            daily_realized_pnl_cents=snap.daily_realized_pnl_cents,
            last_trade_ts=snap.last_trade_ts,
        )

    @staticmethod
    def _key(market_id: str, side: str) -> str:
        return f"{market_id}:{side}"

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)

"""Day 6 live order adapter for cancel-only lifecycle validation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..exceptions import KalshiAPIError, KalshiRequestError
from ..kalshi_client import KalshiClient
from .live_models import (
    CancelOnlyOrderIntent,
    ErrorCategory,
    KalshiCreateOrderRequest,
    KalshiOrderStatus,
    NormalizedOrderStatus,
)


class OrderAdapterError(Exception):
    """Categorized wrapper for order API failures."""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = "unknown",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


class KalshiLiveOrderAdapter:
    """Thin live adapter for submit/cancel/status calls."""

    def __init__(self, client: KalshiClient) -> None:
        self.client = client

    def submit_order(
        self,
        intent: CancelOnlyOrderIntent,
        *,
        client_order_id: str | None = None,
    ) -> KalshiOrderStatus:
        """Submit one limit order and normalize the response."""
        request = KalshiCreateOrderRequest(
            market_id=intent.market_id,
            side=intent.side,
            price_cents=intent.price_cents,
            quantity=intent.quantity,
            client_order_id=client_order_id,
        )
        try:
            raw = self.client.create_order_raw(request.to_payload())
        except KalshiRequestError as exc:
            raise OrderAdapterError(
                str(exc),
                category=self._safe_category(exc.category),
                status_code=exc.status_code,
            ) from exc
        except KalshiAPIError as exc:
            raise OrderAdapterError(str(exc), category="unknown", status_code=None) from exc
        fallback_id = client_order_id if client_order_id else None
        return self._normalize_order(raw, fallback_order_id=fallback_id)

    def cancel_order(self, order_id: str) -> KalshiOrderStatus:
        """Send cancel request for one order id and normalize response."""
        try:
            raw = self.client.cancel_order_raw(order_id)
        except KalshiRequestError as exc:
            raise OrderAdapterError(
                str(exc),
                category=self._safe_category(exc.category),
                status_code=exc.status_code,
            ) from exc
        except KalshiAPIError as exc:
            raise OrderAdapterError(str(exc), category="unknown", status_code=None) from exc
        return self._normalize_order(raw, fallback_order_id=order_id)

    def get_order(self, order_id: str) -> KalshiOrderStatus:
        """Fetch and normalize one order status by id."""
        try:
            raw = self.client.get_order_raw(order_id)
        except KalshiRequestError as exc:
            raise OrderAdapterError(
                str(exc),
                category=self._safe_category(exc.category),
                status_code=exc.status_code,
            ) from exc
        except KalshiAPIError as exc:
            raise OrderAdapterError(str(exc), category="unknown", status_code=None) from exc
        return self._normalize_order(raw, fallback_order_id=order_id)

    def _normalize_order(
        self,
        raw_response: dict[str, Any],
        *,
        fallback_order_id: str | None,
    ) -> KalshiOrderStatus:
        payload = _unwrap_order_payload(raw_response)
        order_id = _first_string(payload, ("order_id", "id", "client_order_id"))
        if order_id is None:
            order_id = fallback_order_id
        if not order_id:
            raise OrderAdapterError(
                "Order payload missing order id.",
                category="unknown",
                status_code=None,
            )

        status_raw = _first_string(payload, ("status", "order_status", "state"))
        requested_qty = _first_int(
            payload,
            ("count", "quantity", "size", "requested_quantity", "requested_count"),
        )
        filled_qty = _first_int(
            payload,
            ("filled_quantity", "filled_count", "filled_size", "matched_count"),
        )
        remaining_qty = _first_int(
            payload,
            (
                "remaining_quantity",
                "remaining_count",
                "remaining_size",
                "unfilled_count",
            ),
        )
        if filled_qty is None:
            filled_qty = 0
        normalized = _normalize_status(
            status_raw=status_raw,
            requested_quantity=requested_qty,
            filled_quantity=filled_qty,
            remaining_quantity=remaining_qty,
        )

        return KalshiOrderStatus(
            order_id=order_id,
            status_raw=status_raw,
            normalized_status=normalized,
            requested_quantity=requested_qty,
            filled_quantity=filled_qty,
            remaining_quantity=remaining_qty,
            updated_at=_first_datetime(
                payload,
                ("updated_time", "updated_at", "last_updated_time", "created_time"),
            ),
            raw=payload,
        )

    @staticmethod
    def _safe_category(value: str) -> ErrorCategory:
        if value in {"auth", "permission", "validation", "rate_limit", "network", "server"}:
            return value
        return "unknown"


def _unwrap_order_payload(raw: dict[str, Any]) -> dict[str, Any]:
    for key in ("order", "data", "result"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _first_datetime(payload: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        value = payload.get(key)
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _normalize_status(
    *,
    status_raw: str | None,
    requested_quantity: int | None,
    filled_quantity: int,
    remaining_quantity: int | None,
) -> NormalizedOrderStatus:
    status = (status_raw or "").strip().lower()
    if any(token in status for token in ("cancel", "canceled", "cancelled")):
        return "canceled"
    if any(token in status for token in ("reject", "rejected", "invalid")):
        return "rejected"
    if "partial" in status:
        return "partially_filled"
    if "fill" in status and "partial" not in status:
        return "filled"
    if any(token in status for token in ("open", "resting", "live", "active")):
        return "open"
    if any(token in status for token in ("pending", "new", "received", "accepted", "queued")):
        return "pending"

    if requested_quantity is not None and requested_quantity > 0:
        if filled_quantity >= requested_quantity:
            return "filled"
        if filled_quantity > 0:
            return "partially_filled"
        if remaining_quantity is not None and remaining_quantity <= 0:
            return "filled"
    elif filled_quantity > 0:
        return "partially_filled"

    return "unknown"

"""Focused tests for live adapter order status fallback lookup behavior."""

from __future__ import annotations

from typing import Any

import pytest

from kalshi_weather_bot.exceptions import KalshiRequestError
from kalshi_weather_bot.execution.live_adapter import KalshiLiveOrderAdapter, OrderAdapterError


class _FakeClient:
    def __init__(self, *, list_payload: dict[str, Any] | list[Any]) -> None:
        self._list_payload = list_payload

    def get_order_raw(self, order_id: str) -> dict[str, Any]:
        del order_id
        raise KalshiRequestError(
            "Kalshi API client error 404: not found",
            category="validation",
            status_code=404,
        )

    def list_orders_raw(self, limit: int = 200) -> dict[str, Any] | list[Any]:
        del limit
        return self._list_payload

    def cancel_order_raw(self, order_id: str) -> dict[str, Any]:
        del order_id
        raise KalshiRequestError(
            "Kalshi API client error 404: not found",
            category="validation",
            status_code=404,
        )


class _DirectClient:
    def __init__(self, *, order_payload: dict[str, Any]) -> None:
        self._order_payload = order_payload

    def get_order_raw(self, order_id: str) -> dict[str, Any]:
        del order_id
        return self._order_payload


class _CancelFallbackClient:
    def __init__(
        self,
        *,
        cancel_responses: dict[str, dict[str, Any] | Exception],
        list_payload: dict[str, Any] | list[Any],
    ) -> None:
        self.cancel_responses = cancel_responses
        self._list_payload = list_payload

    def cancel_order_raw(self, order_id: str) -> dict[str, Any]:
        item = self.cancel_responses.get(order_id)
        if item is None:
            raise KalshiRequestError(
                "Kalshi API client error 404: not found",
                category="validation",
                status_code=404,
            )
        if isinstance(item, Exception):
            raise item
        return item

    def list_orders_raw(self, limit: int = 200) -> dict[str, Any] | list[Any]:
        del limit
        return self._list_payload


def test_get_order_falls_back_to_list_lookup_on_404() -> None:
    client = _FakeClient(
        list_payload={
            "orders": [
                {
                    "order_id": "ord-123",
                    "status": "open",
                    "count": 1,
                    "filled_quantity": 0,
                    "remaining_quantity": 1,
                }
            ]
        }
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.get_order("ord-123")
    assert status.order_id == "ord-123"
    assert status.normalized_status == "open"


def test_get_order_raises_when_not_found_in_list_fallback() -> None:
    client = _FakeClient(
        list_payload={
            "orders": [
                {
                    "order_id": "other-1",
                    "status": "open",
                }
            ]
        }
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    with pytest.raises(OrderAdapterError, match="404"):
        adapter.get_order("ord-xyz")


def test_get_order_uses_top_level_status_when_nested_order_missing_it() -> None:
    client = _DirectClient(
        order_payload={
            "status": "resting",
            "filled_quantity": 0,
            "remaining_quantity": 1,
            "order": {
                "order_id": "ord-456",
                "count": 1,
            },
        }
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.get_order("ord-456")
    assert status.order_id == "ord-456"
    assert status.normalized_status == "open"
    assert status.requested_quantity == 1


def test_list_lookup_matches_nested_order_id() -> None:
    client = _FakeClient(
        list_payload={
            "orders": [
                {
                    "status": "resting",
                    "filled_quantity": 0,
                    "remaining_quantity": 1,
                    "order": {
                        "id": "ord-789",
                        "count": 1,
                    },
                }
            ]
        }
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.get_order("ord-789")
    assert status.order_id == "ord-789"
    assert status.normalized_status == "open"


def test_get_order_unknown_uses_list_lookup_with_client_order_id() -> None:
    client = _DirectClient(
        order_payload={
            "order": {
                "order_id": "ord-901",
            }
        }
    )
    client.list_orders_raw = lambda limit=200: {  # type: ignore[method-assign]
        "orders": [
            {
                "order_id": "ord-901",
                "client_order_id": "cli-901",
                "status": "open",
                "count": 1,
                "filled_quantity": 0,
                "remaining_quantity": 1,
            }
        ]
    }
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.get_order("ord-901", client_order_id="cli-901")
    assert status.order_id == "ord-901"
    assert status.normalized_status == "open"


def test_cancel_order_retries_with_client_order_id_on_404() -> None:
    not_found = KalshiRequestError(
        "Kalshi API client error 404: not found",
        category="validation",
        status_code=404,
    )
    client = _CancelFallbackClient(
        cancel_responses={
            "ord-404": not_found,
            "cli-404": {
                "order": {
                    "order_id": "ord-404",
                    "status": "canceled",
                    "count": 1,
                    "filled_quantity": 0,
                    "remaining_quantity": 1,
                }
            },
        },
        list_payload={"orders": []},
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.cancel_order("ord-404", client_order_id="cli-404")
    assert status.order_id == "ord-404"
    assert status.normalized_status == "canceled"


def test_get_order_maps_executed_with_initial_and_fill_counts_to_open() -> None:
    client = _DirectClient(
        order_payload={
            "status": "executed",
            "order": {
                "order_id": "ord-exec-open",
                "initial_count": 1,
                "fill_count": 0,
                "remaining_count": 1,
            },
        }
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.get_order("ord-exec-open")
    assert status.order_id == "ord-exec-open"
    assert status.requested_quantity == 1
    assert status.filled_quantity == 0
    assert status.remaining_quantity == 1
    assert status.normalized_status == "open"


def test_get_order_maps_executed_with_full_fill_counts_to_filled() -> None:
    client = _DirectClient(
        order_payload={
            "status": "executed",
            "order": {
                "order_id": "ord-exec-filled",
                "initial_count": 1,
                "fill_count": 1,
                "remaining_count": 0,
            },
        }
    )
    adapter = KalshiLiveOrderAdapter(client=client)  # type: ignore[arg-type]
    status = adapter.get_order("ord-exec-filled")
    assert status.order_id == "ord-exec-filled"
    assert status.requested_quantity == 1
    assert status.filled_quantity == 1
    assert status.remaining_quantity == 0
    assert status.normalized_status == "filled"

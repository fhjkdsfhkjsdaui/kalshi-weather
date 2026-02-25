"""Order endpoint fallback tests for KalshiClient status/cancel compatibility."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from kalshi_weather_bot.exceptions import KalshiRequestError
from kalshi_weather_bot.kalshi_client import KalshiClient


def _settings(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "kalshi_api_base_url": "https://api.example.com",
        "kalshi_auth_mode": "bearer",
        "kalshi_bearer_token": "test-token-abc",
        "kalshi_api_key_id": None,
        "kalshi_api_key_secret": None,
        "kalshi_markets_endpoint": "/trade-api/v2/markets",
        "kalshi_timeout_seconds": 5.0,
        "kalshi_default_limit": 100,
        "kalshi_orders_endpoint": "/trade-api/v2/portfolio/orders",
        "kalshi_order_status_endpoint_template": "/trade-api/v2/portfolio/orders/{order_id}",
        "kalshi_order_cancel_endpoint_template": "/trade-api/v2/portfolio/orders/{order_id}",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


def _client(**setting_overrides: object) -> KalshiClient:
    return KalshiClient(
        settings=_settings(**setting_overrides),
        logger=logging.getLogger("test"),
    )


def test_get_order_raw_tries_status_suffix_when_primary_shape_invalid() -> None:
    client = _client()
    calls: list[tuple[str, str]] = []

    def fake_request_json(
        method: str,
        endpoint: str,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del params, json_body
        calls.append((method, endpoint))
        if endpoint.endswith("/status"):
            return {"status": "open", "order_id": "ord-1", "count": 1}
        return {"request_id": "abc-123"}  # not a recognized order status shape

    client._request_json = fake_request_json  # type: ignore[method-assign]
    payload = client.get_order_raw("ord-1")
    assert payload["status"] == "open"
    assert calls == [
        ("GET", "/trade-api/v2/portfolio/orders/ord-1"),
        ("GET", "/trade-api/v2/portfolio/orders/ord-1/status"),
    ]


def test_cancel_order_raw_falls_back_to_post_cancel_suffix_after_404() -> None:
    client = _client()
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_request_json(
        method: str,
        endpoint: str,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del params
        calls.append((method, endpoint, json_body))
        if method == "POST" and endpoint.endswith("/cancel"):
            return {"status": "canceled", "order_id": "ord-2"}
        raise KalshiRequestError(
            "Kalshi API client error 404: not found",
            category="validation",
            status_code=404,
        )

    client._request_json = fake_request_json  # type: ignore[method-assign]
    payload = client.cancel_order_raw("ord-2")
    assert payload["status"] == "canceled"
    assert calls[:4] == [
        ("DELETE", "/trade-api/v2/portfolio/orders/ord-2", None),
        ("POST", "/trade-api/v2/portfolio/orders/ord-2", {}),
        ("DELETE", "/trade-api/v2/portfolio/orders/ord-2/cancel", None),
        ("POST", "/trade-api/v2/portfolio/orders/ord-2/cancel", {}),
    ]


def test_cancel_order_raw_raises_non_404_without_fallback() -> None:
    client = _client()

    def fake_request_json(
        method: str,
        endpoint: str,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del method, endpoint, params, json_body
        raise KalshiRequestError(
            "Kalshi API client error 400: invalid order",
            category="validation",
            status_code=400,
        )

    client._request_json = fake_request_json  # type: ignore[method-assign]
    with pytest.raises(KalshiRequestError, match="400"):
        client.cancel_order_raw("ord-3")

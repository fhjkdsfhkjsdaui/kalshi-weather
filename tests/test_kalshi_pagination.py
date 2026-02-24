"""Kalshi market pagination tests for Day 7 market discovery path."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from kalshi_weather_bot.kalshi_client import KalshiClient


def _settings(**overrides: Any) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "kalshi_api_base_url": "https://api.example.com",
        "kalshi_auth_mode": "bearer",
        "kalshi_bearer_token": "test-token",
        "kalshi_api_key_id": None,
        "kalshi_api_key_secret": None,
        "kalshi_markets_endpoint": "/trade-api/v2/markets",
        "kalshi_timeout_seconds": 5.0,
        "kalshi_default_limit": 100,
        "kalshi_enable_pagination": True,
        "kalshi_max_pages_per_fetch": 10,
        "kalshi_max_markets_fetch": 2000,
        "kalshi_page_sleep_ms": 0,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _client(**setting_overrides: Any) -> KalshiClient:
    logger = logging.getLogger("test.kalshi.pagination")
    return KalshiClient(settings=_settings(**setting_overrides), logger=logger)


def test_fetch_markets_paginates_three_pages_with_summary(monkeypatch: Any) -> None:
    client = _client()
    responses: list[dict[str, Any]] = [
        {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c2"},
        {"markets": [{"ticker": "C"}], "cursor": "c3"},
        {"markets": [{"ticker": "D"}]},
    ]
    call_params: list[dict[str, Any] | None] = []

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del method, endpoint, json_body
        call_params.append(params)
        return responses[len(call_params) - 1]

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=50)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item["ticker"] for item in payload["markets"]] == ["A", "B", "C", "D"]
    assert payload["pagination"]["pages_fetched"] == 3
    assert payload["pagination"]["total_markets_fetched"] == 4
    assert payload["pagination"]["had_more_pages"] is False
    assert payload["pagination"]["deduped_count"] == 0
    assert len(call_params) == 3
    assert call_params[0] == {"limit": 50}
    assert call_params[1] == {"limit": 50, "cursor": "c2"}
    assert call_params[2] == {"limit": 50, "cursor": "c3"}


def test_fetch_markets_stops_at_max_pages(monkeypatch: Any) -> None:
    client = _client(kalshi_max_pages_per_fetch=2)
    responses: list[dict[str, Any]] = [
        {"markets": [{"ticker": "A"}], "cursor": "c2"},
        {"markets": [{"ticker": "B"}], "cursor": "c3"},
        {"markets": [{"ticker": "C"}]},
    ]
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=25)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item["ticker"] for item in payload["markets"]] == ["A", "B"]
    assert payload["pagination"]["pages_fetched"] == 2
    assert payload["pagination"]["had_more_pages"] is True
    assert calls == 2


def test_fetch_markets_stops_at_max_markets(monkeypatch: Any) -> None:
    client = _client(kalshi_max_markets_fetch=3)
    responses: list[dict[str, Any]] = [
        {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c2"},
        {"markets": [{"ticker": "C"}, {"ticker": "D"}], "cursor": "c3"},
    ]
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item["ticker"] for item in payload["markets"]] == ["A", "B", "C"]
    assert payload["pagination"]["pages_fetched"] == 2
    assert payload["pagination"]["total_markets_fetched"] == 3
    assert payload["pagination"]["had_more_pages"] is True
    assert calls == 2


def test_fetch_markets_dedupes_across_pages(monkeypatch: Any) -> None:
    client = _client()
    responses: list[dict[str, Any]] = [
        {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c2"},
        {"markets": [{"ticker": "B"}, {"id": "123"}], "cursor": "c3"},
        {"markets": [{"market_id": "123"}, {"ticker": "C"}]},
    ]
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item.get("ticker") or item.get("id") for item in payload["markets"]] == [
        "A",
        "B",
        "123",
        "C",
    ]
    assert payload["pagination"]["deduped_count"] == 2


def test_fetch_markets_pagination_disabled_warns_once(monkeypatch: Any, caplog: Any) -> None:
    client = _client(kalshi_enable_pagination=False)
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        calls += 1
        return {"markets": [{"ticker": "A"}], "cursor": "c2"}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    caplog.set_level(logging.INFO)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert payload["pagination"]["pages_fetched"] == 1
    assert payload["pagination"]["had_more_pages"] is True
    assert calls == 1
    warning_lines = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "pagination disabled" in record.getMessage().lower()
    ]
    assert len(warning_lines) == 1
    summary_lines = [
        record
        for record in caplog.records
        if "kalshi markets fetch summary" in record.getMessage().lower()
    ]
    assert len(summary_lines) == 1


def test_repeated_cursor_stops_pagination(monkeypatch: Any, caplog: Any) -> None:
    """If the API returns the same cursor twice, pagination stops early."""
    client = _client()
    responses: list[dict[str, Any]] = [
        {"markets": [{"ticker": "A"}], "cursor": "STUCK"},
        {"markets": [{"ticker": "B"}], "cursor": "STUCK"},
        {"markets": [{"ticker": "C"}]},
    ]
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(client, "_request_json", _fake_request)
    caplog.set_level(logging.WARNING)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    # Only 2 pages fetched: first page + one with cursor "STUCK",
    # then loop breaks on repeated cursor before a 3rd fetch.
    assert payload["pagination"]["pages_fetched"] == 2
    assert calls == 2
    repeated_warnings = [
        r for r in caplog.records
        if "repeated cursor" in r.getMessage().lower()
    ]
    assert len(repeated_warnings) == 1


def test_empty_page_stops_pagination(monkeypatch: Any, caplog: Any) -> None:
    """If a page returns zero records (but a cursor), pagination stops."""
    client = _client()
    responses: list[dict[str, Any]] = [
        {"markets": [{"ticker": "A"}], "cursor": "c2"},
        {"markets": [], "cursor": "c3"},
        {"markets": [{"ticker": "B"}]},
    ]
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(client, "_request_json", _fake_request)
    caplog.set_level(logging.WARNING)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item["ticker"] for item in payload["markets"]] == ["A"]
    assert payload["pagination"]["pages_fetched"] == 2
    assert calls == 2
    empty_warnings = [
        r for r in caplog.records
        if "empty page" in r.getMessage().lower()
    ]
    assert len(empty_warnings) == 1


def test_single_page_no_cursor_no_pagination(monkeypatch: Any) -> None:
    """When first page has no cursor, no pagination occurs."""
    client = _client()
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, params, json_body
        calls += 1
        return {"markets": [{"ticker": "A"}, {"ticker": "B"}]}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert payload["pagination"]["pages_fetched"] == 1
    assert payload["pagination"]["had_more_pages"] is False
    assert calls == 1


def test_list_payload_skips_pagination(monkeypatch: Any) -> None:
    """Legacy list payloads bypass pagination entirely."""
    client = _client()

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del method, endpoint, params, json_body
        return [{"ticker": "A"}, {"ticker": "B"}]

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, list)
    assert len(payload) == 2


def test_dedupe_key_ticker_case_insensitive() -> None:
    """Ticker dedupe is case-normalized (upper)."""
    assert KalshiClient._market_dedupe_key(
        {"ticker": "ABC"}
    ) == KalshiClient._market_dedupe_key({"ticker": "abc"})


def test_dedupe_key_missing_fields_returns_none() -> None:
    """Records without ticker or id get None key (no dedup)."""
    assert KalshiClient._market_dedupe_key({"title": "foo"}) is None


def test_dedupe_key_prefers_ticker_over_id() -> None:
    """Ticker is preferred over id for dedup key."""
    key = KalshiClient._market_dedupe_key({"ticker": "XYZ", "id": "123"})
    assert key is not None
    assert key.startswith("ticker:")


def test_extract_pagination_cursor_all_variants() -> None:
    """cursor, next_cursor, next are all valid cursor keys."""
    assert KalshiClient._extract_pagination_cursor(
        {"cursor": "abc"}
    ) == "abc"
    assert KalshiClient._extract_pagination_cursor(
        {"next_cursor": "def"}
    ) == "def"
    assert KalshiClient._extract_pagination_cursor(
        {"next": "ghi"}
    ) == "ghi"
    assert KalshiClient._extract_pagination_cursor({}) is None
    assert KalshiClient._extract_pagination_cursor(
        {"cursor": "  "}
    ) is None


def test_pagination_summary_in_payload(monkeypatch: Any) -> None:
    """Verify pagination summary dict is injected into the returned payload."""
    client = _client()

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del method, endpoint, params, json_body
        return {"markets": [{"ticker": "A"}]}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=10)
    finally:
        client.close()

    assert isinstance(payload, dict)
    pagination = payload["pagination"]
    assert "pagination_enabled" in pagination
    assert "pages_fetched" in pagination
    assert "total_markets_fetched" in pagination
    assert "had_more_pages" in pagination
    assert "deduped_count" in pagination
    assert "max_pages_per_fetch" in pagination
    assert "max_markets_fetch" in pagination


def test_day5_diagnostics_extraction_from_pagination_payload() -> None:
    """_extract_market_fetch_diagnostics reads pagination dict correctly."""
    from kalshi_weather_bot.day5_cli import _extract_market_fetch_diagnostics

    payload = {
        "markets": [{"ticker": "A"}, {"ticker": "B"}],
        "pagination": {
            "pages_fetched": 3,
            "total_markets_fetched": 200,
            "had_more_pages": True,
            "deduped_count": 5,
        },
    }
    diag = _extract_market_fetch_diagnostics(payload, record_count=2)
    assert diag["pages_fetched"] == 3
    assert diag["total_markets_fetched"] == 200
    assert diag["had_more_pages"] is True
    assert diag["deduped_count"] == 5


def test_day5_diagnostics_fallback_without_pagination() -> None:
    """Without pagination dict, diagnostics falls back to cursor/count."""
    from kalshi_weather_bot.day5_cli import _extract_market_fetch_diagnostics

    payload = {
        "markets": [{"ticker": "A"}],
        "cursor": "c2",
    }
    diag = _extract_market_fetch_diagnostics(payload, record_count=1)
    assert diag["pages_fetched"] == 1
    assert diag["total_markets_fetched"] == 1
    assert diag["had_more_pages"] is True


def test_day5_diagnostics_fallback_no_cursor() -> None:
    """Without cursor or pagination dict, had_more_pages is False."""
    from kalshi_weather_bot.day5_cli import _extract_market_fetch_diagnostics

    payload = {"markets": [{"ticker": "A"}]}
    diag = _extract_market_fetch_diagnostics(payload, record_count=1)
    assert diag["had_more_pages"] is False


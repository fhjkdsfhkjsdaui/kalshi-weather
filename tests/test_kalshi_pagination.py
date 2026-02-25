"""Kalshi market pagination tests for Day 7 market discovery path."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from kalshi_weather_bot.exceptions import KalshiAPIError, KalshiRequestError
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
        "kalshi_page_fetch_max_retries": 2,
        "kalshi_page_retry_base_ms": 0,
        "kalshi_page_retry_jitter_ms": 0,
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
    assert payload["pagination"]["stop_reason"] == "page_cap_reached"
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


def test_page_retry_on_timeout_then_success(monkeypatch: Any) -> None:
    client = _client()
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, json_body
        calls += 1
        if params and params.get("cursor") == "c2" and calls == 2:
            raise KalshiRequestError("timeout", category="network", status_code=None)
        if params and params.get("cursor") == "c2":
            return {"markets": [{"ticker": "B"}]}
        return {"markets": [{"ticker": "A"}], "cursor": "c2"}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item["ticker"] for item in payload["markets"]] == ["A", "B"]
    assert payload["pagination"]["pages_fetched"] == 2
    assert calls == 3


def test_page_retry_repeated_timeout_then_success(monkeypatch: Any) -> None:
    client = _client(kalshi_page_fetch_max_retries=3)
    calls = 0
    cursor_failures = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls, cursor_failures
        del method, endpoint, json_body
        calls += 1
        if params and params.get("cursor") == "c2":
            if cursor_failures < 2:
                cursor_failures += 1
                raise KalshiRequestError("read timed out", category="network", status_code=None)
            return {"markets": [{"ticker": "B"}]}
        return {"markets": [{"ticker": "A"}], "cursor": "c2"}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert [item["ticker"] for item in payload["markets"]] == ["A", "B"]
    assert calls == 4


def test_page_retry_hard_failure_includes_progress_context(monkeypatch: Any) -> None:
    client = _client(kalshi_page_fetch_max_retries=1)
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, json_body
        calls += 1
        if params and params.get("cursor") == "c2":
            raise KalshiRequestError("timeout", category="network", status_code=None)
        return {"markets": [{"ticker": "A"}], "cursor": "c2"}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        with pytest.raises(KalshiAPIError) as exc_info:
            client.fetch_markets_raw(limit=100)
        message = str(exc_info.value)
    finally:
        client.close()

    assert "page_index=2" in message
    assert "cursor_present=True" in message
    assert "pages_fetched_so_far=1" in message
    assert "total_markets_fetched_so_far=1" in message
    assert calls == 3


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
            "candidate_matches_found": 7,
            "candidate_target": 100,
            "stopped_on_candidate_target": False,
            "stop_reason": "page_cap_reached",
        },
    }
    diag = _extract_market_fetch_diagnostics(payload, record_count=2)
    assert diag["pages_fetched"] == 3
    assert diag["total_markets_fetched"] == 200
    assert diag["had_more_pages"] is True
    assert diag["deduped_count"] == 5
    assert diag["candidate_matches_found"] == 7
    assert diag["candidate_target"] == 100
    assert diag["stopped_on_candidate_target"] is False
    assert diag["stop_reason"] == "page_cap_reached"


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


def test_page_retry_auth_error_does_not_retry(monkeypatch: Any) -> None:
    """Auth errors (non-retryable) should not be retried at the page level."""
    client = _client(kalshi_page_fetch_max_retries=3)
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, json_body
        calls += 1
        if params and params.get("cursor") == "c2":
            raise KalshiRequestError(
                "Authentication failed", category="auth", status_code=401
            )
        return {"markets": [{"ticker": "A"}], "cursor": "c2"}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        with pytest.raises(KalshiAPIError):
            client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    # First page OK (call 1), second page auth fail (call 2), NO retries.
    assert calls == 2


def test_page_retry_validation_error_does_not_retry(monkeypatch: Any) -> None:
    """Validation errors (non-retryable) should not be retried at the page level."""
    client = _client(kalshi_page_fetch_max_retries=3)
    calls = 0

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del method, endpoint, json_body
        calls += 1
        if params and params.get("cursor") == "c2":
            raise KalshiRequestError(
                "Bad request", category="validation", status_code=400
            )
        return {"markets": [{"ticker": "A"}], "cursor": "c2"}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        with pytest.raises(KalshiAPIError):
            client.fetch_markets_raw(limit=100)
    finally:
        client.close()

    assert calls == 2


def test_fetch_markets_stops_when_weather_candidate_target_met(monkeypatch: Any) -> None:
    client = _client(
        kalshi_max_pages_per_fetch=10,
        kalshi_max_markets_fetch=2000,
    )
    responses: list[dict[str, Any]] = [
        {
            "markets": [
                {"event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-1", "ticker": "SPORT-1"},
                {"event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-2", "ticker": "SPORT-2"},
            ],
            "cursor": "c2",
        },
        {
            "markets": [
                {"event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-3", "ticker": "SPORT-3"},
                {"event_ticker": "KXHIGHTEMP-NYC", "ticker": "KXHIGHTEMP-NYC-90"},
            ],
            "cursor": "c3",
        },
        {"markets": [{"event_ticker": "KXHIGHTEMP-SEA", "ticker": "KXHIGHTEMP-SEA-85"}]},
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
        payload = client.fetch_markets_raw(
            limit=100,
            candidate_target=1,
            candidate_predicate=lambda record: str(record.get("event_ticker", "")).startswith(
                "KXHIGHTEMP"
            ),
        )
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert calls == 2
    assert payload["pagination"]["pages_fetched"] == 2
    assert payload["pagination"]["candidate_matches_found"] == 1
    assert payload["pagination"]["candidate_target"] == 1
    assert payload["pagination"]["candidate_target_met"] is True
    assert payload["pagination"]["stopped_on_candidate_target"] is True
    assert payload["pagination"]["stop_reason"] == "target_reached"


def test_fetch_markets_candidate_target_pagination_ignores_market_cap(monkeypatch: Any) -> None:
    """Candidate-target scans should keep paging (up to page cap) beyond max_markets."""
    client = _client(
        kalshi_max_pages_per_fetch=5,
        kalshi_max_markets_fetch=2,
    )
    responses: list[dict[str, Any]] = [
        {
            "markets": [
                {"event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-1", "ticker": "SPORT-1"},
            ],
            "cursor": "c2",
        },
        {
            "markets": [
                {"event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-2", "ticker": "SPORT-2"},
            ],
            "cursor": "c3",
        },
        {
            "markets": [
                {"event_ticker": "KXHIGHTEMP-NYC", "ticker": "KXHIGHTEMP-NYC-90"},
            ],
        },
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
        payload = client.fetch_markets_raw(
            limit=100,
            candidate_target=1,
            candidate_predicate=lambda record: str(record.get("event_ticker", "")).startswith(
                "KXHIGHTEMP"
            ),
        )
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert calls == 3
    assert payload["pagination"]["pages_fetched"] == 3
    assert payload["pagination"]["candidate_matches_found"] == 1
    assert payload["pagination"]["candidate_target_met"] is True
    assert payload["pagination"]["stop_reason"] == "target_reached"
    assert payload["pagination"]["effective_max_markets"] == 500


def test_fetch_markets_candidate_target_ignored_without_predicate(monkeypatch: Any) -> None:
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
        if calls == 1:
            return {"markets": [{"ticker": "A"}], "cursor": "c2"}
        return {"markets": [{"ticker": "B"}]}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(limit=50, candidate_target=1, candidate_predicate=None)
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert calls == 2
    assert payload["pagination"]["candidate_target"] is None
    assert payload["pagination"]["candidate_matches_found"] == 0
    assert payload["pagination"]["candidate_target_met"] is False
    assert payload["pagination"]["stopped_on_candidate_target"] is False


# ---------------------------------------------------------------------------
# extra_params passthrough
# ---------------------------------------------------------------------------


def test_fetch_markets_extra_params_passed_to_all_pages(monkeypatch: Any) -> None:
    """extra_params should appear in both first and paginated page requests."""
    client = _client()
    call_params: list[dict[str, Any] | None] = []

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del method, endpoint, json_body
        call_params.append(params)
        if len(call_params) == 1:
            return {"markets": [{"ticker": "W1"}], "cursor": "c2"}
        return {"markets": [{"ticker": "W2"}]}

    monkeypatch.setattr(client, "_request_json", _fake_request)
    try:
        payload = client.fetch_markets_raw(
            limit=50,
            extra_params={"series_ticker": "KXHIGHNY", "status": "open"},
        )
    finally:
        client.close()

    assert isinstance(payload, dict)
    assert len(call_params) == 2
    # First page
    assert call_params[0]["series_ticker"] == "KXHIGHNY"
    assert call_params[0]["status"] == "open"
    assert call_params[0]["limit"] == 50
    # Second page
    assert call_params[1]["series_ticker"] == "KXHIGHNY"
    assert call_params[1]["status"] == "open"
    assert call_params[1]["cursor"] == "c2"


# ---------------------------------------------------------------------------
# Weather series fan-out
# ---------------------------------------------------------------------------


def test_parse_weather_series_tickers_splits_and_dedupes() -> None:
    """Comma-separated series tickers are parsed and deduped."""
    from kalshi_weather_bot.day5_cli import _parse_weather_series_tickers

    settings = _settings(kalshi_weather_series_tickers="KXHIGHNY, KXLOWNY,KXHIGHNY ,KXHIGHCHI")
    result = _parse_weather_series_tickers(settings)
    assert result == ["KXHIGHNY", "KXLOWNY", "KXHIGHCHI"]


def test_parse_weather_series_tickers_empty_string_returns_empty() -> None:
    """Empty config value disables fan-out."""
    from kalshi_weather_bot.day5_cli import _parse_weather_series_tickers

    settings = _settings(kalshi_weather_series_tickers="")
    assert _parse_weather_series_tickers(settings) == []


def test_fetch_weather_series_fanout_merges_and_dedupes(monkeypatch: Any) -> None:
    """Fan-out across two series tickers should merge records and dedupe by ticker."""
    from kalshi_weather_bot.day5_cli import _fetch_weather_series_fanout

    client = _client()
    calls: list[dict[str, Any]] = []

    def _fake_request(
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del method, endpoint, json_body
        calls.append(dict(params or {}))
        series = (params or {}).get("series_ticker", "")
        if series == "KXHIGHNY":
            return {
                "markets": [
                    {"ticker": "KXHIGHNY-T39", "event_ticker": "KXHIGHNY-26FEB25"},
                    {"ticker": "KXHIGHNY-T40", "event_ticker": "KXHIGHNY-26FEB25"},
                ],
            }
        if series == "KXLOWNY":
            return {
                "markets": [
                    {"ticker": "KXLOWNY-T10", "event_ticker": "KXLOWNY-26FEB25"},
                    # Duplicate from first series — should be deduped
                    {"ticker": "KXHIGHNY-T39", "event_ticker": "KXHIGHNY-26FEB25"},
                ],
            }
        return {"markets": []}

    monkeypatch.setattr(client, "_request_json", _fake_request)

    def _always_true(record: dict[str, Any]) -> bool:
        return True

    try:
        records, diagnostics = _fetch_weather_series_fanout(
            client=client,
            series_tickers=["KXHIGHNY", "KXLOWNY"],
            limit=200,
            candidate_predicate=_always_true,
            candidate_target=10,
            logger=logging.getLogger("test"),
        )
    finally:
        client.close()

    # 2 from KXHIGHNY + 1 new from KXLOWNY (1 was a dupe)
    assert len(records) == 3
    tickers = [r["ticker"] for r in records]
    assert tickers == ["KXHIGHNY-T39", "KXHIGHNY-T40", "KXLOWNY-T10"]

    assert diagnostics["fetch_strategy"] == "weather_series_fanout"
    assert diagnostics["series_count"] == 2
    assert diagnostics["total_markets_fetched"] == 3
    assert diagnostics["deduped_count"] == 1
    # candidate_matches_found is the sum of per-series counts (pre-dedup)
    assert diagnostics["candidate_matches_found"] == 4

    # Verify series_ticker was passed as query param
    assert calls[0]["series_ticker"] == "KXHIGHNY"
    assert calls[0]["status"] == "open"
    assert calls[1]["series_ticker"] == "KXLOWNY"
    assert calls[1]["status"] == "open"


def test_fanout_disabled_when_series_tickers_empty(monkeypatch: Any) -> None:
    """When KALSHI_WEATHER_SERIES_TICKERS is empty, broad scan is used instead."""
    from kalshi_weather_bot.day5_cli import _parse_weather_series_tickers

    settings = _settings(kalshi_weather_series_tickers="  ")
    assert _parse_weather_series_tickers(settings) == []


def test_resolve_market_records_with_diagnostics_falls_back_when_fanout_disabled(
    monkeypatch: Any,
) -> None:
    """Empty weather series config should keep the existing broad scan path."""
    from kalshi_weather_bot import day5_cli

    settings = _settings(
        kalshi_weather_series_tickers="",
        signal_max_markets_to_scan=200,
    )
    args = SimpleNamespace(
        input_markets_file=None,
        max_markets_to_scan=200,
    )

    class _FakeJournal:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def write_raw_snapshot(self, tag: str, payload: Any) -> str:
            del payload
            return f"/tmp/{tag}.json"

        def write_event(
            self,
            event_type: str,
            payload: dict[str, Any],
            metadata: dict[str, Any] | None = None,
        ) -> None:
            del metadata
            self.events.append({"event_type": event_type, "payload": payload})

    class _FakeClient:
        def __init__(self, *, settings: Any, logger: Any) -> None:
            del logger
            self.settings = settings

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
            del exc_type, exc, exc_tb

        def validate_connection(self) -> dict[str, Any]:
            return {"ok": True}

        def fetch_markets_raw(
            self,
            limit: int | None = None,
            *,
            candidate_predicate: Any = None,
            candidate_target: int | None = None,
            extra_params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del candidate_predicate, candidate_target
            calls.append({"limit": limit, "extra_params": extra_params})
            return {
                "markets": [{"ticker": "KXHIGHNY-26FEB25-T39"}],
                "pagination": {
                    "pages_fetched": 1,
                    "total_markets_fetched": 1,
                    "had_more_pages": False,
                    "deduped_count": 0,
                    "candidate_matches_found": 1,
                },
            }

    calls: list[dict[str, Any]] = []
    journal = _FakeJournal()
    monkeypatch.setattr(day5_cli, "KalshiClient", _FakeClient)

    records, diagnostics = day5_cli._resolve_market_records_with_diagnostics(
        args=args,
        settings=settings,
        logger=logging.getLogger("test"),
        journal=journal,  # type: ignore[arg-type]
        weather_candidate_target=50,
        weather_candidate_predicate=lambda _: True,
    )

    assert len(records) == 1
    assert calls == [{"limit": 200, "extra_params": None}]
    assert diagnostics["fetch_strategy"] == "broad_scan"
    signal_loaded = next(
        event
        for event in journal.events
        if event["event_type"] == "signal_markets_input_loaded"
    )
    assert signal_loaded["payload"]["source"] == "api"

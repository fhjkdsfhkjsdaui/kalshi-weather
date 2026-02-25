"""Day 1 assumption tests — auth headers, response extraction, weather/open
filters, and journal serialization.

These tests exercise the highest-risk code paths where Kalshi schema or auth
assumptions can silently break the pipeline.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa as rsa_module

from kalshi_weather_bot.config import Settings
from kalshi_weather_bot.exceptions import JournalError, KalshiAPIError
from kalshi_weather_bot.journal import JournalWriter, _json_default
from kalshi_weather_bot.kalshi_client import KalshiClient, _load_rsa_private_key, _sign_pss_text
from kalshi_weather_bot.models import MarketSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Generate a small test RSA key once for the module (2048-bit, fast enough for tests).
_TEST_RSA_KEY = rsa_module.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_RSA_PEM = _TEST_RSA_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")


def _make_settings(**overrides: Any) -> Any:
    """Build a minimal Settings-like object for KalshiClient without .env."""
    from unittest.mock import MagicMock

    defaults = {
        "kalshi_api_base_url": "https://api.example.com",
        "kalshi_auth_mode": "bearer",
        "kalshi_bearer_token": "test-token-abc",
        "kalshi_api_key_id": None,
        "kalshi_api_key_secret": None,
        "kalshi_markets_endpoint": "/trade-api/v2/markets",
        "kalshi_timeout_seconds": 5.0,
        "kalshi_default_limit": 100,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _make_client(**settings_overrides: Any) -> KalshiClient:
    """Build a KalshiClient with mocked settings (no network)."""
    settings = _make_settings(**settings_overrides)
    logger = logging.getLogger("test")
    return KalshiClient(settings=settings, logger=logger)


def _market(
    *,
    title: str = "Test market",
    status: str | None = "open",
    ticker: str | None = None,
    category: str | None = None,
    raw_extra: dict[str, Any] | None = None,
) -> MarketSummary:
    raw: dict[str, Any] = {"status": status or ""}
    if ticker:
        raw["ticker"] = ticker
    if category:
        raw["category"] = category
    if raw_extra:
        raw.update(raw_extra)
    return MarketSummary(
        market_id=ticker or "MKT-1",
        ticker=ticker,
        title=title,
        status=status,
        category=category,
        raw=raw,
    )


# ===========================================================================
# 1. Auth header construction
# ===========================================================================

class TestAuthHeaders:
    """Verify _build_auth_headers produces correct headers for each mode."""

    def test_bearer_header_format(self) -> None:
        client = _make_client(kalshi_auth_mode="bearer", kalshi_bearer_token="tok123")
        headers = client._build_auth_headers()
        assert headers == {"Authorization": "Bearer tok123"}

    def test_bearer_missing_token_raises(self) -> None:
        client = _make_client(kalshi_auth_mode="bearer", kalshi_bearer_token=None)
        with pytest.raises(KalshiAPIError, match="Missing bearer token"):
            client._build_auth_headers()

    def test_bearer_empty_token_raises(self) -> None:
        client = _make_client(kalshi_auth_mode="bearer", kalshi_bearer_token="")
        with pytest.raises(KalshiAPIError, match="Missing bearer token"):
            client._build_auth_headers()

    def test_key_secret_returns_signed_headers(self) -> None:
        """RSA-PSS mode should produce KALSHI-ACCESS-KEY, TIMESTAMP, SIGNATURE."""
        client = _make_client(
            kalshi_auth_mode="key_secret",
            kalshi_api_key_id="mykey",
            kalshi_api_key_secret=_TEST_RSA_PEM,
        )
        headers = client._build_auth_headers(method="GET", path="/trade-api/v2/markets")
        assert headers["KALSHI-ACCESS-KEY"] == "mykey"
        assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()
        # Signature should be valid base64.
        sig_bytes = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        assert len(sig_bytes) > 0

    def test_key_secret_missing_key_id_raises(self) -> None:
        client = _make_client(
            kalshi_auth_mode="key_secret",
            kalshi_api_key_id=None,
            kalshi_api_key_secret=_TEST_RSA_PEM,
        )
        with pytest.raises(KalshiAPIError, match="KALSHI_API_KEY_ID"):
            client._build_auth_headers()

    def test_key_secret_missing_rsa_key_raises(self) -> None:
        client = _make_client(
            kalshi_auth_mode="key_secret",
            kalshi_api_key_id="mykey",
            kalshi_api_key_secret=None,
        )
        with pytest.raises(KalshiAPIError, match="RSA private key not loaded"):
            client._build_auth_headers()


# ===========================================================================
# 2. Response extraction — extract_market_records
# ===========================================================================

class TestExtractMarketRecords:
    """Verify tolerant parsing of various response container shapes."""

    def test_direct_list_passthrough(self) -> None:
        client = _make_client()
        payload: list[Any] = [{"ticker": "A"}, {"ticker": "B"}]
        assert client.extract_market_records(payload) == payload

    def test_markets_key(self) -> None:
        client = _make_client()
        payload = {"markets": [{"ticker": "A"}], "cursor": "abc"}
        assert client.extract_market_records(payload) == [{"ticker": "A"}]

    def test_data_key(self) -> None:
        client = _make_client()
        payload = {"data": [{"ticker": "X"}]}
        assert client.extract_market_records(payload) == [{"ticker": "X"}]

    def test_results_key(self) -> None:
        client = _make_client()
        payload = {"results": [{"ticker": "Y"}]}
        assert client.extract_market_records(payload) == [{"ticker": "Y"}]

    def test_items_key(self) -> None:
        client = _make_client()
        payload = {"items": [{"ticker": "Z"}]}
        assert client.extract_market_records(payload) == [{"ticker": "Z"}]

    def test_unknown_key_fallback_works(self) -> None:
        """P1 regression: unknown container key should still extract via fallback."""
        client = _make_client()
        payload = {"contracts": [{"ticker": "FALLBACK"}]}
        result = client.extract_market_records(payload)
        assert result == [{"ticker": "FALLBACK"}]

    def test_no_list_in_dict_raises(self) -> None:
        client = _make_client()
        payload = {"count": 5, "status": "ok"}
        with pytest.raises(KalshiAPIError, match="Unable to find market records"):
            client.extract_market_records(payload)

    def test_non_dict_non_list_raises(self) -> None:
        client = _make_client()
        with pytest.raises(KalshiAPIError, match="Unexpected markets response type"):
            client.extract_market_records("not a dict or list")  # type: ignore[arg-type]

    def test_empty_markets_list_returns_empty(self) -> None:
        client = _make_client()
        payload = {"markets": []}
        assert client.extract_market_records(payload) == []


# ===========================================================================
# 2b. Request param handling — endpoint query + params merge
# ===========================================================================

class TestRequestParamMerging:
    """Verify endpoint query params are preserved when request params are provided."""

    def test_request_json_merges_endpoint_query_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _make_client()
        captured: dict[str, Any] = {}

        def _fake_auth_headers(method: str = "GET", path: str = "/") -> dict[str, str]:
            captured["signed_path"] = path
            return {}

        def _fake_request(
            method: str,
            url: str,
            params: dict[str, Any] | None = None,
            json: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> httpx.Response:
            captured["url"] = url
            captured["params"] = params
            request = httpx.Request(method, f"https://api.example.com{url}")
            return httpx.Response(200, request=request, json={"markets": []})

        monkeypatch.setattr(client, "_build_auth_headers", _fake_auth_headers)
        monkeypatch.setattr(client._client, "request", _fake_request)

        payload = client._request_json(
            "GET",
            "/trade-api/v2/markets?series_ticker=KXHIGHNY&status=open",
            params={"limit": 1000, "cursor": "abc123"},
        )

        assert payload == {"markets": []}
        assert captured["url"] == "/trade-api/v2/markets"
        assert captured["signed_path"] == "/trade-api/v2/markets"
        assert captured["params"] == {
            "series_ticker": "KXHIGHNY",
            "status": "open",
            "limit": 1000,
            "cursor": "abc123",
        }

    def test_request_params_override_endpoint_query_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _make_client()
        captured: dict[str, Any] = {}

        def _fake_auth_headers(method: str = "GET", path: str = "/") -> dict[str, str]:
            return {}

        def _fake_request(
            method: str,
            url: str,
            params: dict[str, Any] | None = None,
            json: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> httpx.Response:
            captured["params"] = params
            request = httpx.Request(method, f"https://api.example.com{url}")
            return httpx.Response(200, request=request, json={"markets": []})

        monkeypatch.setattr(client, "_build_auth_headers", _fake_auth_headers)
        monkeypatch.setattr(client._client, "request", _fake_request)

        client._request_json(
            "GET",
            "/trade-api/v2/markets?limit=5&status=closed",
            params={"limit": 25, "status": "open"},
        )

        assert captured["params"] == {"limit": 25, "status": "open"}


# ===========================================================================
# 3. Weather and open-status filters
# ===========================================================================

class TestWeatherFilter:
    """Verify _looks_weather with word-boundary matching."""

    @pytest.mark.parametrize(
        "title",
        [
            "Will it rain in NYC tomorrow?",
            "NYC rainfall above 2 inches?",
            "DC temperature above 90F?",
            "Hurricane to make landfall?",
            "Denver snowfall over 6 inches?",
            "Chicago heatwave this week?",
            "Wind speed above 50mph in Miami?",
        ],
    )
    def test_weather_titles_match(self, title: str) -> None:
        m = _market(title=title)
        assert KalshiClient._looks_weather(m) is True

    @pytest.mark.parametrize(
        "title",
        [
            "Will Ukraine training exercises continue?",
            "Winding down of trade negotiations?",
            "Electoral brainstorm session outcome?",
            "Fed funds rate above 5%?",
            "Gold price above $2000?",
        ],
    )
    def test_non_weather_titles_no_match(self, title: str) -> None:
        m = _market(title=title)
        assert KalshiClient._looks_weather(m) is False

    def test_weather_in_category(self) -> None:
        m = _market(title="Above 80?", category="weather")
        assert KalshiClient._looks_weather(m) is True

    def test_weather_in_tags(self) -> None:
        m = _market(title="Above 80?", raw_extra={"tags": ["weather", "us"]})
        assert KalshiClient._looks_weather(m) is True

    def test_weather_in_event_title(self) -> None:
        m = _market(title="Above 80?", raw_extra={"event_title": "NYC Temperature Markets"})
        assert KalshiClient._looks_weather(m) is True

    def test_weather_in_description(self) -> None:
        m = _market(title="Above 80?", raw_extra={"description": "Based on precipitation data"})
        assert KalshiClient._looks_weather(m) is True


class TestOpenFilter:
    """Verify _looks_open status detection."""

    @pytest.mark.parametrize("status", ["open", "active", "trading", "listed", "live"])
    def test_open_statuses(self, status: str) -> None:
        m = _market(status=status)
        assert KalshiClient._looks_open(m) is True

    @pytest.mark.parametrize("status", ["closed", "settled", "inactive", "expired"])
    def test_closed_statuses(self, status: str) -> None:
        m = _market(status=status)
        assert KalshiClient._looks_open(m) is False

    def test_case_insensitive(self) -> None:
        m = _market(status="Open")
        assert KalshiClient._looks_open(m) is True

    def test_fallback_bool_is_open(self) -> None:
        m = _market(status=None, raw_extra={"is_open": True})
        assert KalshiClient._looks_open(m) is True

    def test_fallback_bool_false(self) -> None:
        m = _market(status=None, raw_extra={"is_open": False})
        assert KalshiClient._looks_open(m) is False

    def test_unknown_status_no_bool_returns_false(self) -> None:
        m = _market(status="pending")
        assert KalshiClient._looks_open(m) is False


class TestFilterOpenWeather:
    """Integration test for the combined filter."""

    def test_filters_correctly(self) -> None:
        client = _make_client()
        markets = [
            _market(title="Rain in NYC?", status="open"),
            _market(title="Rain in LA?", status="closed"),
            _market(title="Fed rate above 5%?", status="open"),
            _market(title="Snow in Denver?", status="active"),
        ]
        result = client.filter_open_weather_markets(markets)
        titles = [m.title for m in result]
        assert "Rain in NYC?" in titles
        assert "Snow in Denver?" in titles
        assert "Rain in LA?" not in titles
        assert "Fed rate above 5%?" not in titles


# ===========================================================================
# 4.5. Strict validation guard in auth/check flow
# ===========================================================================

def test_validate_connection_rejects_non_market_payload_shape() -> None:
    """
    Auth/check should fail when response has list-like data but no market fields.

    This prevents false-positive auth_success journaling on malformed payloads.
    """
    client = _make_client()
    client.fetch_markets_raw = lambda limit=1: {"errors": [{"code": "BAD_REQUEST"}]}  # type: ignore[assignment]
    with pytest.raises(KalshiAPIError, match="payload shape invalid during auth validation"):
        client.validate_connection()


# ===========================================================================
# 5. Journal serialization
# ===========================================================================

class TestJsonDefault:
    """Verify _json_default handles expected types and rejects unknowns."""

    def test_naive_datetime(self) -> None:
        dt = datetime(2025, 1, 15, 12, 0, 0)
        result = _json_default(dt)
        assert result == "2025-01-15T12:00:00+00:00"

    def test_aware_datetime(self) -> None:
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = _json_default(dt)
        assert "2025-01-15T12:00:00" in result

    def test_path(self) -> None:
        p = Path("/data/journal/test.jsonl")
        result = _json_default(p)
        assert result == "/data/journal/test.jsonl"

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())


class TestJournalWriter:
    """Verify journal write operations and error handling."""

    def test_write_event_creates_valid_jsonl(self, tmp_path: Path) -> None:
        jw = JournalWriter(
            journal_dir=tmp_path / "journal",
            raw_payload_dir=tmp_path / "raw",
            session_id="test123",
        )
        jw.write_event("test_event", {"key": "value"}, {"meta": "data"})

        lines = jw.events_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "test_event"
        assert record["session_id"] == "test123"
        assert record["payload"]["key"] == "value"
        assert record["metadata"]["meta"] == "data"
        assert "ts" in record

    def test_write_event_appends(self, tmp_path: Path) -> None:
        jw = JournalWriter(tmp_path / "j", tmp_path / "r", "s1")
        jw.write_event("a", {"n": 1})
        jw.write_event("b", {"n": 2})
        lines = jw.events_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_event_with_datetime_in_payload(self, tmp_path: Path) -> None:
        jw = JournalWriter(tmp_path / "j", tmp_path / "r", "s1")
        jw.write_event("dt_test", {"when": datetime(2025, 6, 1, tzinfo=UTC)})
        record = json.loads(jw.events_path.read_text().strip())
        assert "2025-06-01" in record["payload"]["when"]

    def test_write_event_non_serializable_raises_journal_error(self, tmp_path: Path) -> None:
        """P2b regression: TypeError from _json_default must become JournalError."""
        jw = JournalWriter(tmp_path / "j", tmp_path / "r", "s1")
        with pytest.raises(JournalError, match="Failed writing event journal"):
            jw.write_event("bad", {"obj": object()})

    def test_write_raw_snapshot_creates_valid_json(self, tmp_path: Path) -> None:
        jw = JournalWriter(tmp_path / "j", tmp_path / "r", "s1")
        path = jw.write_raw_snapshot("markets", {"markets": [{"ticker": "A"}]})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["markets"][0]["ticker"] == "A"

    def test_write_raw_snapshot_non_serializable_raises_journal_error(
        self, tmp_path: Path
    ) -> None:
        jw = JournalWriter(tmp_path / "j", tmp_path / "r", "s1")
        with pytest.raises(JournalError, match="Failed writing raw payload snapshot"):
            jw.write_raw_snapshot("bad", {"obj": object()})


# ===========================================================================
# 6. Datetime parsing edge cases
# ===========================================================================

class TestParseDatetime:
    """Verify _parse_datetime handles formats Kalshi might use."""

    def test_iso_with_z(self) -> None:
        result = KalshiClient._parse_datetime("2025-03-15T18:00:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_iso_with_offset(self) -> None:
        result = KalshiClient._parse_datetime("2025-03-15T18:00:00+00:00")
        assert result is not None

    def test_unix_timestamp(self) -> None:
        result = KalshiClient._parse_datetime(1710525600)
        assert result is not None
        assert result.year == 2024

    def test_none_returns_none(self) -> None:
        assert KalshiClient._parse_datetime(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert KalshiClient._parse_datetime("") is None

    def test_garbage_string_returns_none(self) -> None:
        assert KalshiClient._parse_datetime("not-a-date") is None

    def test_naive_iso(self) -> None:
        result = KalshiClient._parse_datetime("2025-03-15T18:00:00")
        assert result is not None
        assert result.tzinfo == UTC


# ===========================================================================
# 7. Config — key_secret file loading
# ===========================================================================


class TestKeySecretFileLoading:
    """Verify config loads RSA key from file when KALSHI_API_KEY_SECRET_FILE is set."""

    def _base_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set minimal required env vars for key_secret mode."""
        monkeypatch.setenv("APP_ENV", "dev")
        monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("KALSHI_AUTH_MODE", "key_secret")
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
        monkeypatch.setenv("NWS_USER_AGENT", "test-agent")

    def test_loads_key_from_pem_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._base_env(monkeypatch)
        pem = tmp_path / "test.pem"
        pem.write_text("-----BEGIN RSA PRIVATE KEY-----\nFAKEKEY\n-----END RSA PRIVATE KEY-----\n")
        monkeypatch.setenv("KALSHI_API_KEY_SECRET_FILE", str(pem))

        settings = Settings(_env_file=None)
        assert settings.kalshi_api_key_secret is not None
        assert "FAKEKEY" in settings.kalshi_api_key_secret

    def test_missing_pem_file_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._base_env(monkeypatch)
        monkeypatch.setenv("KALSHI_API_KEY_SECRET_FILE", str(tmp_path / "nonexistent.pem"))

        with pytest.raises(Exception, match="does not exist"):
            Settings(_env_file=None)

    def test_inline_secret_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backward compat: inline KALSHI_API_KEY_SECRET without file path."""
        self._base_env(monkeypatch)
        monkeypatch.setenv("KALSHI_API_KEY_SECRET", "inline-secret-value")

        settings = Settings(_env_file=None)
        assert settings.kalshi_api_key_secret == "inline-secret-value"

    def test_file_takes_precedence_over_inline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When both file and inline are set, file wins."""
        self._base_env(monkeypatch)
        monkeypatch.setenv("KALSHI_API_KEY_SECRET", "inline-value")
        pem = tmp_path / "test.pem"
        pem.write_text("file-value-key")
        monkeypatch.setenv("KALSHI_API_KEY_SECRET_FILE", str(pem))

        settings = Settings(_env_file=None)
        assert settings.kalshi_api_key_secret == "file-value-key"

    def test_no_secret_and_no_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """key_secret mode with neither inline secret nor file should raise."""
        self._base_env(monkeypatch)
        # Don't set KALSHI_API_KEY_SECRET or KALSHI_API_KEY_SECRET_FILE

        with pytest.raises(Exception, match="KALSHI_API_KEY_SECRET"):
            Settings(_env_file=None)


# ===========================================================================
# 8. Config — signal defaults
# ===========================================================================


def test_signal_max_markets_to_scan_default_is_1000(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("KALSHI_AUTH_MODE", "bearer")
    monkeypatch.setenv("KALSHI_BEARER_TOKEN", "test-token")
    monkeypatch.setenv("NWS_USER_AGENT", "test-agent")
    monkeypatch.setenv("DRY_RUN_MODE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")
    monkeypatch.delenv("SIGNAL_MAX_MARKETS_TO_SCAN", raising=False)

    settings = Settings(_env_file=None)
    assert settings.signal_max_markets_to_scan == 1000


# ===========================================================================
# 9. RSA-PSS signing
# ===========================================================================


class TestRSAPSSSigning:
    """Verify RSA-PSS signing and key loading utilities."""

    def test_sign_pss_text_produces_valid_base64(self) -> None:
        sig = _sign_pss_text(_TEST_RSA_KEY, "1234567890GET/trade-api/v2/markets")
        decoded = base64.b64decode(sig)
        # RSA 2048-bit key produces 256-byte signatures.
        assert len(decoded) == 256

    def test_sign_pss_text_deterministic_structure(self) -> None:
        """Same input should produce different signatures (PSS is randomized)."""
        msg = "1234567890GET/trade-api/v2/markets"
        sig1 = _sign_pss_text(_TEST_RSA_KEY, msg)
        sig2 = _sign_pss_text(_TEST_RSA_KEY, msg)
        # PSS uses random salt, so signatures should differ.
        assert sig1 != sig2

    def test_load_rsa_private_key_from_pem(self) -> None:
        key = _load_rsa_private_key(_TEST_RSA_PEM)
        assert key.key_size == 2048

    def test_load_rsa_private_key_invalid_pem_raises(self) -> None:
        with pytest.raises(KalshiAPIError, match="Failed to load RSA private key"):
            _load_rsa_private_key("not-a-pem-key")

    def test_signature_verifies_with_public_key(self) -> None:
        """End-to-end: sign with private key, verify with public key."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        message = "1709000000GET/trade-api/v2/markets"
        sig_b64 = _sign_pss_text(_TEST_RSA_KEY, message)
        sig_bytes = base64.b64decode(sig_b64)

        public_key = _TEST_RSA_KEY.public_key()
        # Should not raise — valid signature.
        public_key.verify(
            sig_bytes,
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

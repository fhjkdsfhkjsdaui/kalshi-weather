"""Day 3 tests for Kalshi weather contract parser and audit CLI."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from kalshi_weather_bot.contracts.parser import KalshiWeatherContractParser, summarize_parse_results
from kalshi_weather_bot.contracts_cli import (
    _extract_market_records,
)
from kalshi_weather_bot.contracts_cli import (
    main as contracts_cli_main,
)
from kalshi_weather_bot.exceptions import ContractParserError
from kalshi_weather_bot.journal import JournalWriter
from kalshi_weather_bot.log_setup import JsonConsoleFormatter
from kalshi_weather_bot.redaction import sanitize_text


def _market(title: str, **overrides: Any) -> dict[str, Any]:
    market: dict[str, Any] = {
        "ticker": "KXWEATHER-TEST",
        "event_ticker": "KXWEATHER",
        "title": title,
        "subtitle": "",
        "rules_primary": "",
        "rules_secondary": "",
        "status": "active",
        "open_time": "2026-02-24T10:00:00Z",
        "close_time": "2026-02-24T18:00:00Z",
        "expiration_time": "2026-02-24T19:00:00Z",
        "yes_sub_title": "Condition true",
        "no_sub_title": "Condition false",
    }
    market.update(overrides)
    return market


def test_parse_temperature_threshold_high_confidence() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(
        _market("Will the high temperature in New York, NY be above 90F on July 4?")
    )
    assert parsed.weather_candidate is True
    assert parsed.parse_status == "parsed"
    assert parsed.weather_dimension == "temperature"
    assert parsed.metric_subtype == "high_temp"
    assert parsed.threshold_operator == ">"
    assert parsed.threshold_value == 90
    assert parsed.threshold_unit == "F"
    assert parsed.location_normalized is not None
    assert parsed.location_normalized.city == "New York"
    assert parsed.location_normalized.state == "NY"
    assert parsed.parse_confidence >= 0.75


def test_parse_precip_between_range() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(
        _market("Will rainfall in Seattle, WA be between 1 and 2 inches?")
    )
    assert parsed.parse_status == "parsed"
    assert parsed.weather_dimension == "precipitation"
    assert parsed.threshold_operator == "between"
    assert parsed.threshold_low == 1
    assert parsed.threshold_high == 2
    assert parsed.threshold_unit == "in"


def test_parse_snowfall_at_least() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(
        _market("Will snowfall in Denver, CO be at least 6 inches?")
    )
    assert parsed.parse_status == "parsed"
    assert parsed.weather_dimension == "snowfall"
    assert parsed.threshold_operator == ">="
    assert parsed.threshold_value == 6
    assert parsed.threshold_unit == "in"


def test_ambiguous_weather_title_rejects_safely() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(_market("Will the weather in Phoenix, AZ be notable tomorrow?"))
    assert parsed.weather_candidate is True
    assert parsed.parse_status == "rejected"
    assert "missing_threshold_operator" in parsed.rejection_reasons
    assert "missing_threshold_value" in parsed.rejection_reasons


def test_non_weather_title_marked_unsupported() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(_market("Will the Fed funds rate be above 5.5%?"))
    assert parsed.weather_candidate is False
    assert parsed.parse_status == "unsupported"
    assert parsed.rejection_reasons == ["non_weather_market"]


def test_operator_and_unit_parsing_for_wind() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(_market("Will wind gust in Miami, FL be at most 35 mph?"))
    assert parsed.parse_status == "parsed"
    assert parsed.weather_dimension == "wind"
    assert parsed.metric_subtype == "wind_gust"
    assert parsed.threshold_operator == "<="
    assert parsed.threshold_value == 35
    assert parsed.threshold_unit == "mph"


def test_datetime_extraction_from_market_fields() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(
        _market(
            "Will rainfall in Austin, TX be above 0.5 inches?",
            open_time=1767225600,
            close_time="2026-12-31T23:00:00Z",
            settlement_time="2027-01-01T01:00:00Z",
        )
    )
    assert parsed.contract_start_time is not None
    assert parsed.contract_start_time.tzinfo is not None
    assert parsed.contract_end_time is not None
    assert parsed.contract_end_time.tzinfo is not None
    assert parsed.resolution_time is not None
    assert parsed.resolution_time.tzinfo is not None


def test_confidence_boundary_between_parsed_and_rejected() -> None:
    parser = KalshiWeatherContractParser()
    parsed = parser.parse_market(_market("Will snowfall in Boston, MA be above 4 inches?"))
    rejected = parser.parse_market(_market("Will the weather in Boston, MA change tomorrow?"))
    assert parsed.parse_status == "parsed"
    assert rejected.parse_status == "rejected"
    assert parsed.parse_confidence > rejected.parse_confidence


def test_contract_parser_cli_smoke(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    payload = {
        "markets": [
            _market("Will high temperature in Dallas, TX be above 95F?"),
            _market("Will the Fed funds rate be above 5.5%?", ticker="KXNONWEATHER"),
        ]
    }
    input_file = tmp_path / "markets.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("KALSHI_AUTH_MODE", "bearer")
    monkeypatch.setenv("KALSHI_BEARER_TOKEN", "test-bearer-token")
    monkeypatch.setenv("NWS_USER_AGENT", "test-agent")
    monkeypatch.setenv("JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("RAW_PAYLOAD_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("WEATHER_RAW_PAYLOAD_DIR", str(tmp_path / "raw_weather"))

    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day3", "--input-file", str(input_file), "--max-print", "1"],
    )
    exit_code = contracts_cli_main()
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Scanned 2 markets" in output
    assert "weather candidates=1" in output

    journal_dir = tmp_path / "journal"
    journal_files = list(journal_dir.glob("*.jsonl"))
    assert journal_files
    lines = journal_files[0].read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event_type"] for line in lines]
    assert "contract_parse_summary" in events


def test_journal_redacts_secrets_in_error_payload(tmp_path: Path) -> None:
    writer = JournalWriter(
        journal_dir=tmp_path / "journal",
        raw_payload_dir=tmp_path / "raw",
        session_id="redacttest",
    )
    writer.write_event(
        "failure",
        payload={
            "error": "Authorization=Bearer abc123 signature=XYZ",
            "headers": {"KALSHI-ACCESS-SIGNATURE": "abc", "Authorization": "Bearer test"},
            "private_key": "-----BEGIN PRIVATE KEY-----\nTOPSECRET\n-----END PRIVATE KEY-----",
        },
    )
    content = writer.events_path.read_text(encoding="utf-8")
    assert "abc123" not in content
    assert "TOPSECRET" not in content
    assert "[REDACTED]" in content


def test_log_formatter_redacts_sensitive_message_content() -> None:
    formatter = JsonConsoleFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="auth failed Authorization=Bearer abc123 KALSHI-ACCESS-SIGNATURE=xyz",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert "abc123" not in output
    assert "xyz" not in output
    assert "[REDACTED]" in output


# ===========================================================================
# Additional parser correctness tests
# ===========================================================================


class TestParserOverconfidencePrevention:
    """Verify vague titles are rejected, not overconfidently parsed."""

    def test_bare_weather_keyword_gets_rejected(self) -> None:
        """'weather' alone should not get a dimension; must be rejected."""
        parser = KalshiWeatherContractParser()
        parsed = parser.parse_market(
            _market("Will the weather in Chicago, IL be extreme tomorrow?")
        )
        assert parsed.weather_candidate is True
        assert parsed.weather_dimension is None
        assert parsed.parse_status == "rejected"
        assert "missing_dimension" in parsed.rejection_reasons

    def test_bare_weather_with_threshold_still_rejected(self) -> None:
        """Even with a parseable threshold, bare 'weather' should be rejected."""
        parser = KalshiWeatherContractParser()
        parsed = parser.parse_market(
            _market("Will the weather in LA, CA be above 5 on the severity scale?")
        )
        assert parsed.weather_candidate is True
        assert parsed.weather_dimension is None
        assert parsed.parse_status == "rejected"


class TestNationwideLocationCoverage:
    """Verify parser handles many U.S. locations, not just NYC."""

    @pytest.mark.parametrize(
        "title,expected_city,expected_state",
        [
            ("Will high temp in Chicago, IL be above 90F?", "Chicago", "IL"),
            ("Will rainfall in Houston, TX be above 2 inches?", "Houston", "TX"),
            ("Will snowfall in Anchorage, AK be at least 12 inches?", "Anchorage", "AK"),
            ("Will high temp in Los Angeles, CA be above 100F?", "Los Angeles", "CA"),
            ("Will wind speed in Oklahoma City, OK be above 40 mph?", "Oklahoma City", "OK"),
            ("Will rainfall in Portland, OR be between 0.5 and 1 inches?", "Portland", "OR"),
        ],
    )
    def test_parses_various_us_cities(
        self, title: str, expected_city: str, expected_state: str
    ) -> None:
        parser = KalshiWeatherContractParser()
        parsed = parser.parse_market(_market(title))
        assert parsed.parse_status == "parsed"
        assert parsed.location_normalized is not None
        assert parsed.location_normalized.city == expected_city
        assert parsed.location_normalized.state == expected_state

    def test_city_only_without_state_still_parses(self) -> None:
        parser = KalshiWeatherContractParser()
        parsed = parser.parse_market(
            _market("Will high temperature in Seattle be above 80F?")
        )
        assert parsed.parse_status == "parsed"
        assert parsed.location_normalized is not None
        assert parsed.location_normalized.city == "Seattle"
        assert parsed.location_normalized.state is None


class TestPayloadFieldDefensiveness:
    """Verify parser reads from alternate field names and missing fields."""

    def test_missing_subtitle_and_rules_still_parses(self) -> None:
        """A minimal market dict with only a title should work."""
        parser = KalshiWeatherContractParser()
        market = {"ticker": "TEST", "title": "Will rainfall in Miami, FL be above 1 inch?"}
        parsed = parser.parse_market(market)
        assert parsed.parse_status == "parsed"
        assert parsed.weather_dimension == "precipitation"

    def test_market_title_alternate_key(self) -> None:
        """Title from 'market_title' key should be found."""
        parser = KalshiWeatherContractParser()
        market = {"id": "TEST", "market_title": "Will high temp in Denver, CO be above 95F?"}
        parsed = parser.parse_market(market)
        assert parsed.parse_status == "parsed"
        assert parsed.weather_dimension == "temperature"

    def test_weather_tag_triggers_candidate(self) -> None:
        """A 'weather' tag should trigger weather_candidate even without keywords."""
        parser = KalshiWeatherContractParser()
        market = {
            "ticker": "X",
            "title": "Will metric X in Denver, CO be above 50?",
            "tags": ["weather"],
        }
        parsed = parser.parse_market(market)
        assert parsed.weather_candidate is True


class TestCLIEmptyPayloadHandling:
    """Verify CLI extraction handles edge cases cleanly."""

    def test_empty_markets_list_returns_empty(self) -> None:
        result = _extract_market_records({"markets": []})
        assert result == []

    def test_no_list_field_in_dict_raises(self) -> None:
        with pytest.raises(ContractParserError, match="No market records found"):
            _extract_market_records({"count": 5, "status": "ok"})

    def test_list_of_non_dicts_raises(self) -> None:
        """A non-empty list with no dict objects is malformed payload shape."""
        with pytest.raises(ContractParserError, match="contained no object records"):
            _extract_market_records(["not", "a", "dict"])


class TestSummarizeParsResults:
    """Verify summarize_parse_results aggregation."""

    def test_summary_counts_correct(self) -> None:
        parser = KalshiWeatherContractParser()
        results = parser.parse_markets([
            _market("Will high temp in NYC, NY be above 90F?"),
            _market("Will the weather in LA, CA be nice?"),
            _market("Will the stock market crash?"),
        ])
        summary = summarize_parse_results(results)
        assert summary.total_markets_scanned == 3
        assert summary.parsed == 1
        assert summary.rejected == 1
        assert summary.unsupported == 1


class TestRedactionCoverage:
    """Verify redaction handles mixed-case bearer tokens."""

    def test_mixed_case_bearer_token_redacted(self) -> None:
        text = "Bearer AbCdEf123456.token_value"
        result = sanitize_text(text)
        assert "AbCdEf123456" not in result
        assert "[REDACTED]" in result

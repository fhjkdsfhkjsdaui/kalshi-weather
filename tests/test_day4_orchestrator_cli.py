"""Day 4 offline orchestrator CLI smoke tests."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kalshi_weather_bot.day4_cli import main as day4_main


def _set_required_env(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("KALSHI_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("KALSHI_AUTH_MODE", "bearer")
    monkeypatch.setenv("KALSHI_BEARER_TOKEN", "test-token")
    monkeypatch.setenv("NWS_USER_AGENT", "test-agent")
    monkeypatch.setenv("JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("RAW_PAYLOAD_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("WEATHER_RAW_PAYLOAD_DIR", str(tmp_path / "raw_weather"))
    monkeypatch.setenv("DRY_RUN_MODE", "true")


def test_day4_cli_fixture_smoke(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    fixture = {
        "intents": [
            {
                "market_id": "KX-WEATHER-CLI-1",
                "side": "yes",
                "price_cents": 30,
                "quantity": 1,
                "parser_confidence": 0.9,
                "parsed_contract_ref": "parsed-1",
                "weather_snapshot_ref": "weather-1",
                "weather_snapshot_retrieved_at": (now - timedelta(seconds=120)).isoformat(),
                "timestamp": now.isoformat(),
            },
            {
                "market_id": "KX-WEATHER-CLI-2",
                "side": "no",
                "price_cents": 35,
                "quantity": 1,
                "parser_confidence": 0.2,
                "parsed_contract_ref": "parsed-2",
                "weather_snapshot_ref": "weather-1",
                "weather_snapshot_retrieved_at": (now - timedelta(seconds=120)).isoformat(),
                "timestamp": now.isoformat(),
            },
        ]
    }
    fixture_path = tmp_path / "day4_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--input-json", str(fixture_path), "--max-print", "5"],
    )

    exit_code = day4_main()
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Processed 2 actions" in output
    assert "accepted=1" in output
    assert "rejected=1" in output

    journal_files = list((tmp_path / "journal").glob("*.jsonl"))
    assert journal_files
    lines = journal_files[0].read_text(encoding="utf-8").strip().splitlines()
    event_types = [json.loads(line)["event_type"] for line in lines]
    assert "risk_evaluation" in event_types
    assert "dry_run_execution_summary" in event_types


def test_day4_cli_default_demo_intents(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    """Run with no --input-json to exercise built-in demo intents."""
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["kalshi-weather-day4", "--max-print", "5"],
    )
    exit_code = day4_main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Processed 2 actions" in output
    # Demo intent 1 (confidence 0.88) should be accepted,
    # Demo intent 2 (confidence 0.40) should be rejected
    assert "accepted=1" in output
    assert "rejected=1" in output


def test_day4_cli_kill_switch_rejects_all(
    monkeypatch: Any, tmp_path: Path, capsys: Any,
) -> None:
    """All intents should be rejected when --kill-switch is active."""
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["kalshi-weather-day4", "--kill-switch", "--max-print", "5"],
    )
    exit_code = day4_main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "accepted=0" in output
    assert "rejected=2" in output


def test_day4_cli_cancel_first_accepted(
    monkeypatch: Any, tmp_path: Path, capsys: Any,
) -> None:
    """--cancel-first-accepted should produce a cancellation action."""
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--cancel-first-accepted", "--max-print", "10"],
    )
    exit_code = day4_main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "cancelled=1" in output


def test_day4_cli_fixture_kill_switch(
    monkeypatch: Any, tmp_path: Path, capsys: Any,
) -> None:
    """Fixture with kill_switch_active should reject all intents."""
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    fixture = {
        "kill_switch_active": True,
        "intents": [
            {
                "market_id": "KX-KILL-TEST",
                "side": "yes",
                "price_cents": 25,
                "quantity": 1,
                "parser_confidence": 0.95,
                "parsed_contract_ref": "p1",
                "weather_snapshot_ref": "w1",
                "weather_snapshot_retrieved_at": (now - timedelta(seconds=60)).isoformat(),
                "timestamp": now.isoformat(),
            },
        ],
    }
    fixture_path = tmp_path / "kill_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--input-json", str(fixture_path), "--max-print", "5"],
    )
    exit_code = day4_main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "accepted=0" in output
    assert "rejected=1" in output


def test_day4_cli_malformed_fixture_returns_error(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    """Malformed fixture should produce exit code 4."""
    fixture_path = tmp_path / "bad.json"
    fixture_path.write_text("not valid json!", encoding="utf-8")

    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--input-json", str(fixture_path)],
    )
    exit_code = day4_main()
    assert exit_code == 4


def test_day4_cli_missing_fixture_returns_error(
    monkeypatch: Any, tmp_path: Path,
) -> None:
    """Non-existent fixture file should produce exit code 4."""
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--input-json", str(tmp_path / "nope.json")],
    )
    exit_code = day4_main()
    assert exit_code == 4


def test_day4_cli_nationwide_markets(
    monkeypatch: Any, tmp_path: Path, capsys: Any,
) -> None:
    """Fixture with markets from various US locations all process correctly."""
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    fixture = {
        "intents": [
            {
                "market_id": f"KX-WEATHER-{city}",
                "side": "yes",
                "price_cents": 30,
                "quantity": 1,
                "parser_confidence": 0.9,
                "parsed_contract_ref": f"parsed-{city}",
                "weather_snapshot_ref": f"weather-{city}",
                "weather_snapshot_retrieved_at": (now - timedelta(seconds=60)).isoformat(),
                "timestamp": now.isoformat(),
            }
            for city in ["CHI", "HOU", "ANC", "LAX", "OKC", "PDX"]
        ],
    }
    fixture_path = tmp_path / "nationwide.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    _set_required_env(monkeypatch, tmp_path)
    # Raise concurrent order limit so all 6 markets can be accepted
    monkeypatch.setenv("RISK_MAX_CONCURRENT_OPEN_ORDERS", "10")
    monkeypatch.setenv("RISK_MAX_TOTAL_EXPOSURE_CENTS", "10000")
    monkeypatch.setenv("RISK_MAX_EXPOSURE_PER_MARKET_CENTS", "10000")
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--input-json", str(fixture_path), "--max-print", "10"],
    )
    exit_code = day4_main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Processed 6 actions" in output
    # All should be accepted with good confidence and fresh weather
    assert "accepted=6" in output


def test_day4_cli_fixture_bare_list(
    monkeypatch: Any, tmp_path: Path, capsys: Any,
) -> None:
    """Fixture as bare list (no wrapper object) should work."""
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    fixture = [
        {
            "market_id": "KX-BARE-LIST",
            "side": "yes",
            "price_cents": 40,
            "quantity": 1,
            "parser_confidence": 0.85,
            "parsed_contract_ref": "p1",
            "weather_snapshot_ref": "w1",
            "weather_snapshot_retrieved_at": (now - timedelta(seconds=60)).isoformat(),
            "timestamp": now.isoformat(),
        },
    ]
    fixture_path = tmp_path / "bare_list.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day4", "--input-json", str(fixture_path), "--max-print", "5"],
    )
    exit_code = day4_main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "accepted=1" in output


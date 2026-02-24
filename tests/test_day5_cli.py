"""Day 5 CLI offline smoke tests."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kalshi_weather_bot.day5_cli import main as day5_main


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


def _write_weather_fixture(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "day5_weather_snapshot.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    payload["retrieval_timestamp"] = (now - timedelta(seconds=90)).isoformat()
    payload["generated_timestamp"] = (now - timedelta(minutes=10)).isoformat()
    payload["updated_timestamp"] = (now - timedelta(minutes=3)).isoformat()
    target = tmp_path / "weather_snapshot.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_day5_cli_offline_smoke(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SIGNAL_MIN_MODEL_CONFIDENCE", "0.5")
    monkeypatch.setenv("RISK_MAX_STAKE_PER_TRADE_CENTS", "50")

    markets_file = Path(__file__).parent / "fixtures" / "day5_markets.json"
    weather_file = _write_weather_fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day5",
            "--input-markets-file",
            str(markets_file),
            "--input-weather-file",
            str(weather_file),
            "--max-candidates",
            "3",
            "--print-rejections",
            "5",
        ],
    )

    exit_code = day5_main()
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "scanned=3" in output
    assert "weather_candidates=3" in output
    assert "dry_run_submitted=1" in output
    assert "risk_rejected=1" in output
    assert "Day 5 Dry-Run Actions" in output
    assert "Day 5 Signal Rejections" in output

    journal_files = list((tmp_path / "journal").glob("*.jsonl"))
    assert journal_files
    event_types: list[str] = []
    for line in journal_files[0].read_text(encoding="utf-8").strip().splitlines():
        event_types.append(json.loads(line)["event_type"])
    assert "signal_scan_start" in event_types
    assert "signal_batch_summary" in event_types
    assert "signal_scan_shutdown" in event_types
    assert "signal_candidate_selected" in event_types
    assert "signal_candidate_rejected" in event_types
    assert "dry_run_order_submitted" in event_types
    assert "risk_rejection" in event_types


def test_day5_cli_invalid_max_markets_to_scan(monkeypatch: Any, tmp_path: Path) -> None:
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kalshi-weather-day5", "--max-markets-to-scan", "0"],
    )
    exit_code = day5_main()
    assert exit_code == 2

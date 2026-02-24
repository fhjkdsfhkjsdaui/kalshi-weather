"""Tests for terminal dashboard rendering and plain-mode integration path."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console

from kalshi_weather_bot import day7_cli
from kalshi_weather_bot.ui.terminal_dashboard import TerminalDashboard


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
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")
    monkeypatch.setenv("ALLOW_LIVE_API", "true")
    monkeypatch.setenv("ALLOW_LIVE_FILLS", "true")
    monkeypatch.setenv("MICRO_MODE_ENABLED", "true")


def test_dashboard_summary_lines_include_key_metrics() -> None:
    console = Console(record=True, width=120)
    dashboard = TerminalDashboard(console=console, session_id="sess123")
    counts: Counter[str] = Counter(
        {
            "cycles_processed": 2,
            "total_markets_fetched": 400,
            "weather_markets_after_filter": 25,
            "candidates_generated": 3,
            "orders_submitted": 1,
            "fills": 1,
            "partial_fills": 0,
            "cancels": 0,
            "rejects": 0,
            "unresolved": 0,
        }
    )
    lines = dashboard.build_end_summary_lines(
        mode="live_micro",
        effective_dry_run=False,
        supervised=True,
        total_counts=counts,
        final_summary={
            "open_positions_count": 1,
            "realized_pnl_cents": 45,
            "daily_gross_exposure_cents": 110,
            "gross_exposure_utilization": 0.22,
            "halted_early": False,
            "halt_reason": None,
        },
        exit_code=0,
    )
    rendered = "\n".join(lines)
    assert "mode=live_micro" in rendered
    assert "orders_submitted=1" in rendered
    assert "realized_pnl=45c ($0.45)" in rendered
    assert "exit_code=0" in rendered


def test_dashboard_narrow_console_renders_sections() -> None:
    console = Console(record=True, width=90)
    dashboard = TerminalDashboard(console=console, session_id="sess-narrow")
    dashboard.record_event(
        severity="WARN",
        message="Pagination cursor seen; additional pages remain.",
        dedupe_key="warn:pagination",
    )
    counts = {
        "pages_fetched": 3,
        "total_markets_fetched": 500,
        "weather_markets_after_filter": 15,
        "candidates_generated": 2,
        "had_more_pages": 1,
    }
    totals: Counter[str] = Counter(
        {
            "cycles_processed": 1,
            "signal_scanned": 200,
            "weather_markets_after_filter": 15,
            "candidates_generated": 2,
            "orders_submitted": 1,
            "fills": 0,
            "partial_fills": 0,
            "cancels": 1,
            "rejects": 0,
            "unresolved": 0,
        }
    )
    dashboard.render_cycle(
        cycle_index=1,
        max_cycles=3,
        mode="live_micro",
        effective_dry_run=False,
        supervised=True,
        counts=counts,
        total_counts=totals,
        final_summary={
            "open_positions_count": 0,
            "realized_pnl_cents": 0,
            "daily_gross_exposure_cents": 40,
            "gross_exposure_utilization": 0.08,
            "halted_early": False,
            "halt_reason": None,
        },
        skip_reasons=Counter(),
        attempts=[],
    )
    text = console.export_text()
    assert "Runtime Status" in text
    assert "Key Metrics" in text
    assert "Health / Connectivity" in text
    assert "Event Feed" in text


def test_day7_cli_explicit_plain_mode_path(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    _set_required_env(monkeypatch, tmp_path)

    def _fake_eval(
        *, args: Any, settings: Any, logger: Any, journal: Any
    ) -> tuple[Any, Any, Any]:
        del args, settings, logger, journal
        return [], [], {
            "scanned": 0,
            "weather_candidates": 0,
            "matched": 0,
            "estimated": 0,
            "filtered": 0,
            "selected": 0,
            "pages_fetched": 1,
            "total_markets_fetched": 0,
            "weather_markets_after_filter": 0,
            "candidates_generated": 0,
            "had_more_pages": 0,
            "deduped_count": 0,
        }

    monkeypatch.setattr(day7_cli, "_evaluate_signal_candidates", _fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kalshi-weather-day7",
            "--dry-run",
            "--ui-mode",
            "plain",
            "--max-cycles",
            "1",
        ],
    )

    exit_code = day7_cli.main()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "mode=dry_run" in output

    journal_files = sorted((tmp_path / "journal").glob("*.jsonl"))
    assert journal_files
    records = [
        json.loads(line)
        for line in journal_files[0].read_text().splitlines()
        if line.strip()
    ]
    start_event = next(item for item in records if item["event_type"] == "micro_session_start")
    assert start_event["payload"]["ui_mode"] == "plain"


def test_initialize_dashboard_plain_mode_returns_none() -> None:
    logger = logging.getLogger("test.day7.ui")
    logger.handlers = [logging.StreamHandler()]
    logger.propagate = False
    console = Console(record=True)
    dashboard = day7_cli._initialize_dashboard(
        ui_mode="plain",
        console=console,
        logger=logger,
        session_id="test-session",
    )
    assert dashboard is None
    assert len(logger.handlers) == 1

"""Professional operator-facing terminal dashboard for supervised runs."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..redaction import sanitize_text
from .event_buffer import EventBuffer
from .models import Severity


def _severity_from_level(level_no: int) -> Severity:
    if level_no >= logging.CRITICAL:
        return "CRITICAL"
    if level_no >= logging.ERROR:
        return "ERROR"
    if level_no >= logging.WARNING:
        return "WARN"
    return "INFO"


class _DashboardLogHandler(logging.Handler):
    """Route logger output into dashboard event feed instead of JSON spam."""

    def __init__(self, dashboard: TerminalDashboard) -> None:
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.dashboard.record_log(
                severity=_severity_from_level(record.levelno),
                message=sanitize_text(record.getMessage()),
            )
        except Exception:
            self.handleError(record)


class TerminalDashboard:
    """Rich-rendered dashboard for live supervision with bounded event feed."""

    def __init__(
        self,
        *,
        console: Console,
        session_id: str,
        max_events: int = 80,
        dedupe_window_seconds: int = 45,
    ) -> None:
        self.console = console
        self.session_id = session_id
        self.events = EventBuffer(
            max_events=max_events,
            dedupe_window_seconds=dedupe_window_seconds,
        )
        self._logger: logging.Logger | None = None
        self._original_handlers: list[logging.Handler] = []
        self._log_handler: logging.Handler | None = None
        self.health: dict[str, Any] = {
            "kalshi_status": "unknown",
            "weather_status": "unknown",
            "last_kalshi_fetch_ts": None,
            "last_weather_fetch_ts": None,
            "pages_fetched": 0,
            "cursor_seen": False,
        }

    def attach_logger(self, logger: logging.Logger) -> None:
        """Replace CLI JSON handler with dashboard feed handler."""
        self._logger = logger
        self._original_handlers = list(logger.handlers)
        self._log_handler = _DashboardLogHandler(self)
        logger.handlers = [self._log_handler]

    def detach_logger(self) -> None:
        """Restore original logger handlers."""
        if self._logger is None:
            return
        self._logger.handlers = self._original_handlers
        self._logger = None
        self._original_handlers = []
        self._log_handler = None

    def record_log(self, *, severity: Severity, message: str) -> None:
        """Append one log entry into deduped feed."""
        dedupe_key: str | None = None
        if severity in {"WARN", "ERROR"}:
            dedupe_key = f"{severity}:{message}"
        self.events.add(severity=severity, message=message, dedupe_key=dedupe_key)

    def record_event(
        self,
        *,
        severity: Severity,
        message: str,
        dedupe_key: str | None = None,
    ) -> None:
        self.events.add(severity=severity, message=message, dedupe_key=dedupe_key)

    def render_startup(
        self,
        *,
        mode: str,
        effective_dry_run: bool,
        supervised: bool,
        config_mode: str,
        cli_mode: str | None,
    ) -> None:
        self.record_event(
            severity="INFO",
            message=(
                f"Session start: mode={mode} dry_run={effective_dry_run} "
                f"supervised={supervised} config_mode={config_mode} cli_mode={cli_mode}"
            ),
        )

    def render_cycle(
        self,
        *,
        cycle_index: int,
        max_cycles: int,
        mode: str,
        effective_dry_run: bool,
        supervised: bool,
        counts: dict[str, int],
        total_counts: Counter[str],
        final_summary: dict[str, Any],
        skip_reasons: Counter[str],
        attempts: list[Any],
    ) -> None:
        now = datetime.now(UTC)
        self.health["kalshi_status"] = "ok"
        self.health["weather_status"] = "ok"
        self.health["last_kalshi_fetch_ts"] = now
        self.health["last_weather_fetch_ts"] = now
        self.health["pages_fetched"] = counts.get("pages_fetched", 1)
        self.health["cursor_seen"] = bool(counts.get("had_more_pages", 0))

        self.record_event(
            severity="INFO",
            message=(
                f"Cycle {cycle_index}/{max_cycles}: "
                f"markets={counts.get('total_markets_fetched', 0)} "
                f"weather={counts.get('weather_markets_after_filter', 0)} "
                f"candidates={counts.get('candidates_generated', 0)} "
                f"submitted={total_counts['orders_submitted']}"
            ),
        )
        if counts.get("weather_markets_after_filter", 0) == 0:
            self.record_event(
                severity="WARN",
                message="No weather markets found after filtering in this cycle.",
                dedupe_key="warn:no_weather_markets_after_filter",
            )
        elif counts.get("candidates_generated", 0) == 0:
            self.record_event(
                severity="WARN",
                message="Weather markets found but no candidates cleared signal filters.",
                dedupe_key="warn:weather_markets_but_no_candidates",
            )

        header = self._build_header_panel(
            mode=mode,
            effective_dry_run=effective_dry_run,
            supervised=supervised,
        )
        metrics = self._build_metrics_panel(
            cycle_index=cycle_index,
            max_cycles=max_cycles,
            total_counts=total_counts,
            final_summary=final_summary,
        )
        health = self._build_health_panel()
        positions = self._build_positions_panel(
            attempts=attempts,
            final_summary=final_summary,
        )
        events = self._build_events_panel()
        top_row = Columns([metrics, health], equal=True, expand=True)
        if self.console.width < 110:
            body = Group(header, metrics, health, positions, events)
        else:
            body = Group(header, top_row, positions, events)
        self.console.print(body)

        if skip_reasons:
            rendered = ", ".join(
                f"{reason}:{count}" for reason, count in skip_reasons.most_common(5)
            )
            self.console.print(
                Panel(
                    rendered,
                    title="Top Skip Reasons",
                    border_style="yellow",
                )
            )

    def render_end(
        self,
        *,
        mode: str,
        effective_dry_run: bool,
        supervised: bool,
        total_counts: Counter[str],
        final_summary: dict[str, Any],
        exit_code: int,
    ) -> None:
        lines = self.build_end_summary_lines(
            mode=mode,
            effective_dry_run=effective_dry_run,
            supervised=supervised,
            total_counts=total_counts,
            final_summary=final_summary,
            exit_code=exit_code,
        )
        border = "green" if exit_code == 0 else "red"
        self.console.print(
            Panel(
                "\n".join(lines),
                title="Session Summary",
                border_style=border,
            )
        )

    def build_end_summary_lines(
        self,
        *,
        mode: str,
        effective_dry_run: bool,
        supervised: bool,
        total_counts: Counter[str],
        final_summary: dict[str, Any],
        exit_code: int,
    ) -> list[str]:
        pnl_cents = int(final_summary.get("realized_pnl_cents", 0))
        pnl_dollars = pnl_cents / 100.0
        return [
            (
                f"mode={mode} dry_run={effective_dry_run} supervised={supervised} "
                f"session_id={self.session_id}"
            ),
            (
                f"cycles={total_counts['cycles_processed']} "
                f"markets={total_counts['total_markets_fetched']} "
                f"weather_markets={total_counts['weather_markets_after_filter']} "
                f"candidates={total_counts['candidates_generated']}"
            ),
            (
                f"orders_submitted={total_counts['orders_submitted']} "
                f"fills={total_counts['fills']} "
                f"partial_fills={total_counts['partial_fills']} "
                f"cancels={total_counts['cancels']} "
                f"rejects={total_counts['rejects']} unresolved={total_counts['unresolved']}"
            ),
            (
                f"open_positions={final_summary.get('open_positions_count', 0)} "
                f"realized_pnl={pnl_cents}c (${pnl_dollars:.2f}) "
                f"gross_exposure={final_summary.get('daily_gross_exposure_cents', 0)}c "
                f"gross_util={final_summary.get('gross_exposure_utilization', 0.0):.2f}"
            ),
            (
                f"halted_early={final_summary.get('halted_early', False)} "
                f"halt_reason={final_summary.get('halt_reason') or '-'} exit_code={exit_code}"
            ),
        ]

    def _build_header_panel(
        self,
        *,
        mode: str,
        effective_dry_run: bool,
        supervised: bool,
    ) -> Panel:
        now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
        if mode == "dry_run":
            mode_style = "cyan"
        elif mode == "live_cancel_only":
            mode_style = "magenta"
        else:
            mode_style = "green"
        dry_style = "cyan" if effective_dry_run else "yellow"
        supervised_style = "green" if supervised else "red"
        text = Text()
        text.append("Kalshi Weather Ops", style="bold white")
        text.append("  |  ")
        text.append(f"mode={mode}", style=f"bold {mode_style}")
        text.append("  ")
        text.append(f"dry_run={effective_dry_run}", style=dry_style)
        text.append("  ")
        text.append(f"supervised={supervised}", style=supervised_style)
        text.append("  ")
        text.append(f"session={self.session_id}", style="bold blue")
        text.append("  ")
        text.append(now_utc, style="dim")
        return Panel(text, border_style="blue", title="Runtime Status")

    def _build_metrics_panel(
        self,
        *,
        cycle_index: int,
        max_cycles: int,
        total_counts: Counter[str],
        final_summary: dict[str, Any],
    ) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("Cycle", f"{cycle_index}/{max_cycles}")
        table.add_row("Markets Scanned", str(total_counts["signal_scanned"]))
        table.add_row("Weather Markets", str(total_counts["weather_markets_after_filter"]))
        table.add_row("Candidates", str(total_counts["candidates_generated"]))
        table.add_row("Orders Submitted", str(total_counts["orders_submitted"]))
        table.add_row(
            "Fills / Partial",
            f"{total_counts['fills']} / {total_counts['partial_fills']}",
        )
        table.add_row(
            "Cancels / Rejects",
            f"{total_counts['cancels']} / {total_counts['rejects']}",
        )
        table.add_row("Unresolved", str(total_counts["unresolved"]))
        table.add_row("Open Positions", str(final_summary.get("open_positions_count", 0)))
        pnl_cents = int(final_summary.get("realized_pnl_cents", 0))
        table.add_row("Realized PnL", f"{pnl_cents}c (${pnl_cents / 100.0:.2f})")
        gross = int(final_summary.get("daily_gross_exposure_cents", 0))
        gross_util = float(final_summary.get("gross_exposure_utilization", 0.0))
        table.add_row("Gross Exposure", f"{gross}c ({gross_util:.2f})")
        table.add_row(
            "Halted",
            f"{final_summary.get('halted_early', False)}",
        )
        table.add_row("Halt Reason", str(final_summary.get("halt_reason") or "-"))
        return Panel(table, title="Key Metrics", border_style="cyan")

    def _build_health_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Kalshi API", str(self.health.get("kalshi_status", "unknown")))
        table.add_row("Weather Provider", str(self.health.get("weather_status", "unknown")))
        table.add_row(
            "Last Kalshi Fetch",
            self._format_ts(self.health.get("last_kalshi_fetch_ts")),
        )
        table.add_row(
            "Last Weather Fetch",
            self._format_ts(self.health.get("last_weather_fetch_ts")),
        )
        table.add_row("Pages Fetched", str(self.health.get("pages_fetched", 0)))
        table.add_row("Cursor Seen", str(self.health.get("cursor_seen", False)))
        table.add_row(
            "Pagination Warnings",
            str(self.events.count_matching(severity="WARN", contains="pagination")),
        )
        table.add_row("Warn Count", str(self.events.count_matching(severity="WARN")))
        table.add_row("Error Count", str(self.events.count_matching(severity="ERROR")))
        table.add_row(
            "Critical Count",
            str(self.events.count_matching(severity="CRITICAL")),
        )
        return Panel(table, title="Health / Connectivity", border_style="magenta")

    def _build_positions_panel(
        self,
        *,
        attempts: list[Any],
        final_summary: dict[str, Any],
    ) -> Panel:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Market", overflow="fold")
        table.add_column("Side", width=4)
        table.add_column("Qty", justify="right", width=4)
        table.add_column("Price", justify="right", width=6)
        table.add_column("Outcome", width=12)
        shown = 0
        for attempt in attempts[-8:]:
            market = attempt.candidate.ticker or attempt.candidate.market_id
            table.add_row(
                market[:28],
                attempt.candidate.side.upper(),
                str(attempt.candidate.quantity),
                f"{attempt.candidate.price_cents}c",
                attempt.outcome,
            )
            shown += 1
        if shown == 0:
            table.add_row(
                "No recent attempts",
                "-",
                "-",
                "-",
                f"open_positions={final_summary.get('open_positions_count', 0)}",
            )
        return Panel(table, title="Active / Recent Positions", border_style="green")

    def _build_events_panel(self) -> Panel:
        table = Table(show_header=True, header_style="bold")
        table.add_column("UTC", width=9)
        table.add_column("Severity", width=9)
        table.add_column("Message", overflow="fold")
        for event in self.events.snapshot(newest_first=False)[-12:]:
            sev = event.severity
            sev_style = {
                "INFO": "white",
                "WARN": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold red",
            }[sev]
            msg = event.message
            if event.count > 1 and event.last_seen is not None:
                msg = (
                    f"{msg} (x{event.count}, "
                    f"last {event.last_seen.strftime('%H:%M:%SZ')})"
                )
            table.add_row(
                event.ts.strftime("%H:%M:%S"),
                f"[{sev_style}]{sev}[/{sev_style}]",
                msg,
            )
        if not self.events.snapshot():
            table.add_row("-", "INFO", "No events yet")
        return Panel(table, title="Event Feed", border_style="white")

    @staticmethod
    def _format_ts(value: Any) -> str:
        if not isinstance(value, datetime):
            return "-"
        ts = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return ts.strftime("%H:%M:%SZ")

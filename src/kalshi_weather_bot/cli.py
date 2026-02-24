"""Day 1 CLI: load config, authenticate, fetch markets, journal results."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import load_settings
from .exceptions import ConfigError, JournalError, KalshiAPIError
from .journal import JournalWriter
from .kalshi_client import KalshiClient
from .log_setup import setup_logger
from .models import MarketSummary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="List open weather markets from Kalshi (Day 1).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override market fetch limit from config.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=None,
        help="Override max number of weather markets shown.",
    )
    return parser.parse_args()


def _print_summary(
    console: Console,
    all_markets: list[MarketSummary],
    weather_markets: list[MarketSummary],
    max_print: int,
) -> None:
    console.print(
        f"Fetched {len(all_markets)} total markets | "
        f"Open weather markets: {len(weather_markets)}"
    )

    if not weather_markets:
        console.print("No open weather markets found.")
        console.print("Diagnostic hints:")
        console.print("- Verify KALSHI_MARKETS_ENDPOINT in .env against your existing bot.")
        console.print("- Confirm weather category/tags field names in raw payload snapshot.")
        console.print(
            "- Confirm auth mode/header format in "
            "src/kalshi_weather_bot/kalshi_client.py."
        )
        return

    table = Table(title="Open Weather Markets (Day 1)")
    table.add_column("Market ID/Ticker", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("Status")
    table.add_column("Close (UTC)")
    table.add_column("Settle (UTC)")

    for market in weather_markets[:max_print]:
        table.add_row(
            market.ticker or market.market_id,
            market.title,
            market.status or "unknown",
            market.close_time.astimezone(UTC).isoformat() if market.close_time else "-",
            market.settlement_time.astimezone(UTC).isoformat()
            if market.settlement_time
            else "-",
        )
    console.print(table)


def main() -> int:
    """Run the Day 1 workflow."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("Configuration failure: %s", exc)
        return 2

    try:
        journal = JournalWriter(
            journal_dir=settings.journal_dir,
            raw_payload_dir=settings.raw_payload_dir,
            session_id=session_id,
        )
        journal.write_event(
            event_type="startup",
            payload=settings.safe_summary(),
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize journal: %s", exc)
        return 3

    exit_code = 0
    try:
        with KalshiClient(settings=settings, logger=logger) as client:
            try:
                auth_info = client.validate_connection()
                journal.write_event(
                    "auth_success",
                    payload=auth_info,
                    metadata={"session_id": session_id},
                )
                logger.info("Kalshi authentication/connectivity validated.")
            except KalshiAPIError as exc:
                journal.write_event(
                    "auth_failure",
                    payload={"error": str(exc)},
                    metadata={"session_id": session_id},
                )
                logger.error("Authentication failure: %s", exc)
                exit_code = 4
                return exit_code

            raw_payload = client.fetch_markets_raw(limit=args.limit)
            raw_snapshot_path = journal.write_raw_snapshot("markets", raw_payload)
            journal.write_event(
                "raw_payload_snapshot",
                payload={"path": str(raw_snapshot_path)},
                metadata={"session_id": session_id},
            )

            all_markets = client.normalize_markets(raw_payload)
            open_weather = client.filter_open_weather_markets(all_markets)

            max_print = args.max_print or settings.kalshi_max_print
            _print_summary(console, all_markets, open_weather, max_print=max_print)

            summary_payload: dict[str, Any] = {
                "total_markets_fetched": len(all_markets),
                "open_weather_markets": len(open_weather),
                "printed_markets": min(len(open_weather), max_print),
            }
            journal.write_event(
                "summary",
                payload=summary_payload,
                metadata={"session_id": session_id},
            )
    except (KalshiAPIError, JournalError) as exc:
        exit_code = 5
        logger.error("Run failed: %s", exc)
        try:
            journal.write_event(
                "run_failure",
                payload={"error": str(exc)},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write run_failure event to journal.")
    except Exception as exc:  # pragma: no cover - defensive catch for Day 1 CLI
        exit_code = 99
        logger.exception("Unexpected failure: %s", exc)
        try:
            journal.write_event(
                "run_failure_unhandled",
                payload={"error": str(exc), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write run_failure_unhandled event to journal.")
    finally:
        if journal is not None:
            try:
                journal.write_event(
                    "shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write shutdown event to journal.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

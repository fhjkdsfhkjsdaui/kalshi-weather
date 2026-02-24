"""Day 6 CLI: live cancel-only order lifecycle validation runner."""

from __future__ import annotations

import argparse
import sys
import uuid

from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .exceptions import ConfigError, JournalError, KalshiAPIError
from .execution.cancel_only import CancelOnlyRunner
from .execution.live_adapter import KalshiLiveOrderAdapter, OrderAdapterError
from .execution.live_models import CancelOnlyBatchResult, CancelOnlyOrderIntent
from .journal import JournalWriter
from .kalshi_client import KalshiClient
from .log_setup import setup_logger
from .redaction import sanitize_text


def parse_args() -> argparse.Namespace:
    """Parse Day 6 operator CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Day 6 live cancel-only order lifecycle validation.",
    )
    parser.add_argument("--mode", choices=["live_cancel_only"], default="live_cancel_only")
    parser.add_argument("--market", required=True, help="Target market/ticker identifier.")
    parser.add_argument("--side", choices=["yes", "no"], required=True)
    parser.add_argument("--price-cents", type=int, required=True)
    parser.add_argument("--qty", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--poll-timeout-seconds", type=float, default=None)
    parser.add_argument("--poll-interval-seconds", type=float, default=None)
    parser.add_argument("--cancel-delay-ms", type=int, default=None)
    parser.add_argument("--max-print", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Day 6 CLI is for live_cancel_only validation; use Day 4 for dry-run flows.",
    )
    return parser.parse_args()


def _print_attempts(console: Console, result: CancelOnlyBatchResult, max_rows: int) -> None:
    table = Table(title="Day 6 Cancel-Only Attempts")
    table.add_column("Attempt")
    table.add_column("Local Terminal")
    table.add_column("Remote Terminal")
    table.add_column("Matched")
    table.add_column("Filled Qty")
    table.add_column("Halt", overflow="fold")
    for item in result.attempts[:max_rows]:
        reconciliation = item.reconciliation
        remote_terminal = (
            reconciliation.remote_terminal_state if reconciliation is not None else "-"
        )
        matched = (
            str(reconciliation.matched) if reconciliation is not None else "-"
        )
        filled = (
            str(reconciliation.filled_quantity)
            if reconciliation is not None
            else str(item.local_record.filled_quantity)
        )
        table.add_row(
            str(item.attempt_index),
            item.local_record.current_state,
            str(remote_terminal),
            matched,
            filled,
            item.halt_reason or "-",
        )
    console.print(table)


def _build_runner(
    *,
    settings: Settings,
    logger: object,
    journal: JournalWriter,
) -> tuple[CancelOnlyRunner, KalshiClient]:
    client = KalshiClient(settings=settings, logger=logger)
    adapter = KalshiLiveOrderAdapter(client=client)
    runner = CancelOnlyRunner(
        settings=settings,
        adapter=adapter,
        logger=logger,
        journal=journal,
    )
    return runner, client


def main() -> int:
    """Run Day 6 cancel-only validation."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    if args.max_print <= 0:
        logger.error("--max-print must be > 0.")
        return 2
    if args.dry_run:
        logger.error("Day 6 CLI does not support --dry-run. Use Day 4 CLI for dry-run mode.")
        return 2
    if not (1 <= args.price_cents <= 99):
        logger.error("--price-cents must be between 1 and 99 (Kalshi binary contract range).")
        return 2
    if args.qty <= 0:
        logger.error("--qty must be > 0.")
        return 2
    if args.attempts <= 0:
        logger.error("--attempts must be > 0.")
        return 2

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("Configuration failure: %s", sanitize_text(str(exc)))
        return 2

    if settings.execution_mode != "live_cancel_only":
        logger.error(
            "Unsafe startup: EXECUTION_MODE must be 'live_cancel_only' for Day 6 CLI."
        )
        return 2
    if not settings.allow_live_api or not settings.cancel_only_enabled:
        logger.error(
            "Unsafe startup: ALLOW_LIVE_API=true and CANCEL_ONLY_ENABLED=true are required."
        )
        return 2

    poll_timeout = (
        args.poll_timeout_seconds
        if args.poll_timeout_seconds is not None
        else settings.cancel_only_poll_timeout_seconds
    )
    poll_interval = (
        args.poll_interval_seconds
        if args.poll_interval_seconds is not None
        else settings.cancel_only_poll_interval_seconds
    )
    cancel_delay_ms = (
        args.cancel_delay_ms
        if args.cancel_delay_ms is not None
        else settings.cancel_only_cancel_delay_ms
    )

    try:
        journal = JournalWriter(
            journal_dir=settings.journal_dir,
            raw_payload_dir=settings.raw_payload_dir,
            session_id=session_id,
        )
        journal.write_event(
            "day6_cancel_only_startup",
            payload={
                "mode": args.mode,
                "market": args.market,
                "side": args.side,
                "price_cents": args.price_cents,
                "qty": args.qty,
                "attempts": args.attempts,
                "poll_timeout_seconds": poll_timeout,
                "poll_interval_seconds": poll_interval,
                "cancel_delay_ms": cancel_delay_ms,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize day6 journal: %s", sanitize_text(str(exc)))
        return 3

    exit_code = 0
    client: KalshiClient | None = None
    try:
        intent = CancelOnlyOrderIntent(
            market_id=args.market,
            side=args.side,
            price_cents=args.price_cents,
            quantity=args.qty,
        )
        runner, client = _build_runner(settings=settings, logger=logger, journal=journal)
        result = runner.run_batch(
            intent=intent,
            attempts=args.attempts,
            poll_timeout_seconds=poll_timeout,
            poll_interval_seconds=poll_interval,
            cancel_delay_ms=cancel_delay_ms,
        )
        summary = result.summary
        console.print(
            "attempts_requested="
            f"{summary.attempts_requested} attempts_executed={summary.attempts_executed} "
            f"submit_ack={summary.submit_ack_count} cancel_success={summary.cancel_success_count} "
            f"rejected={summary.rejected_count} unresolved={summary.unresolved_count} "
            f"mismatch={summary.reconciliation_mismatch_count} "
            f"unexpected_fills={summary.unexpected_fill_count} "
            f"halted_early={summary.halted_early}"
        )
        _print_attempts(console, result, max_rows=args.max_print)

        if summary.unexpected_fill_count > 0:
            exit_code = 5
        elif summary.unresolved_count > 0 or summary.reconciliation_mismatch_count > 0:
            exit_code = 4

    except (KalshiAPIError, OrderAdapterError, ValueError, JournalError) as exc:
        exit_code = 4
        logger.error("Day 6 cancel-only flow failed: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "day6_cancel_only_failure",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write day6_cancel_only_failure event.")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        exit_code = 99
        logger.exception("Unexpected Day 6 failure: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "day6_cancel_only_failure_unhandled",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write day6_cancel_only_failure_unhandled event.")
    finally:
        if client is not None:
            client.close()
        if journal is not None:
            try:
                journal.write_event(
                    "day6_cancel_only_shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write day6_cancel_only_shutdown event.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

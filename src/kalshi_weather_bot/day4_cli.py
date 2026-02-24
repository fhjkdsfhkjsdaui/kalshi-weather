"""Day 4 CLI: risk evaluation + dry-run execution plumbing demo."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .exceptions import ConfigError, JournalError
from .execution.dry_run import DryRunExecutor
from .execution.models import ExecutionResult, TradeIntent
from .journal import JournalWriter
from .log_setup import setup_logger
from .redaction import sanitize_text
from .risk.manager import RiskManager


def parse_args() -> argparse.Namespace:
    """Parse Day 4 orchestration CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Day 4 risk manager + dry-run execution smoke flow."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional offline fixture JSON for intents.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=10,
        help="Maximum results to print in the summary table.",
    )
    parser.add_argument(
        "--kill-switch",
        action="store_true",
        help="Force kill-switch active for this run.",
    )
    parser.add_argument(
        "--cancel-first-accepted",
        action="store_true",
        help="Cancel the first accepted dry-run order after submission.",
    )
    return parser.parse_args()


def _default_demo_intents(now: datetime) -> list[TradeIntent]:
    # Keep demo intents fresh by default so strict stale-age settings don't
    # cause surprising rejections during smoke validation.
    return [
        TradeIntent(
            market_id="KXWEATHER-DEMO-001",
            side="yes",
            price_cents=41,
            quantity=1,
            parser_confidence=0.88,
            parsed_contract_ref="demo_parsed_contract_001",
            weather_snapshot_ref="demo_weather_snapshot_001",
            weather_snapshot_retrieved_at=now,
            timestamp=now,
            strategy_tag="day4_demo",
            market_liquidity=500,
            market_spread_cents=2,
        ),
        TradeIntent(
            market_id="KXWEATHER-DEMO-002",
            side="no",
            price_cents=35,
            quantity=1,
            parser_confidence=0.40,
            parsed_contract_ref="demo_parsed_contract_002",
            weather_snapshot_ref="demo_weather_snapshot_001",
            weather_snapshot_retrieved_at=now,
            timestamp=now,
            strategy_tag="day4_demo",
            market_liquidity=500,
            market_spread_cents=2,
        ),
    ]


def _load_intents_from_json(path: Path, now: datetime) -> tuple[list[TradeIntent], bool]:
    if not path.exists():
        raise ValueError(f"Input fixture file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed reading {path}: {exc}") from exc

    kill_switch_active = False
    raw_intents: list[Any]
    if isinstance(payload, dict):
        raw_intents_value = payload.get("intents")
        if not isinstance(raw_intents_value, list):
            raise ValueError("Fixture JSON object must include an 'intents' list.")
        raw_intents = raw_intents_value
        kill_switch_active = bool(payload.get("kill_switch_active", False))
    elif isinstance(payload, list):
        raw_intents = payload
    else:
        raise ValueError("Fixture JSON must be a list of intents or an object with 'intents'.")

    intents: list[TradeIntent] = []
    for idx, raw in enumerate(raw_intents):
        if not isinstance(raw, dict):
            raise ValueError(f"Intent at index {idx} must be an object.")
        default_payload: dict[str, Any] = {
            "parsed_contract_ref": f"fixture_parsed_contract_{idx}",
            "weather_snapshot_ref": "fixture_weather_snapshot",
            "weather_snapshot_retrieved_at": now,
            "timestamp": now,
            "parser_confidence": 0.85,
            "side": "yes",
            "price_cents": 50,
            "quantity": 1,
        }
        merged = {**default_payload, **raw}
        intents.append(TradeIntent.model_validate(merged))
    return intents, kill_switch_active


def _print_results(console: Console, results: list[ExecutionResult], max_print: int) -> None:
    table = Table(title="Day 4 Dry-Run Execution Results")
    table.add_column("Intent ID")
    table.add_column("Market")
    table.add_column("Side")
    table.add_column("Px")
    table.add_column("Qty")
    table.add_column("Status")
    table.add_column("Decision")
    table.add_column("Reasons", overflow="fold")

    for result in results[:max_print]:
        intent_id = result.intent.intent_id if result.intent else "-"
        market_id = (
            result.intent.market_id
            if result.intent
            else (result.order.market_id if result.order else "-")
        )
        side = result.intent.side if result.intent else (result.order.side if result.order else "-")
        px = str(result.intent.price_cents) if result.intent else "-"
        qty = str(result.intent.quantity) if result.intent else "-"
        reasons = ", ".join(result.reasons) if result.reasons else "-"
        table.add_row(
            intent_id or "-",
            market_id,
            side,
            px,
            qty,
            result.status,
            result.decision_code,
            reasons,
        )
    console.print(table)


def main() -> int:
    """Run Day 4 orchestration demo flow."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    if args.max_print <= 0:
        logger.error("--max-print must be > 0.")
        return 2

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
            event_type="day4_startup",
            payload={
                "input_json": str(args.input_json) if args.input_json else None,
                "max_print": args.max_print,
                "risk_enabled": settings.risk_enabled,
                "dry_run_mode": settings.dry_run_mode,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize day4 journal: %s", exc)
        return 3

    exit_code = 0
    try:
        now = datetime.now(UTC)
        intents: list[TradeIntent]
        input_kill_switch = False
        if args.input_json:
            intents, input_kill_switch = _load_intents_from_json(args.input_json, now=now)
        else:
            intents = _default_demo_intents(now=now)

        risk_manager = RiskManager(settings=settings, logger=logger)
        executor = DryRunExecutor(
            settings=settings,
            risk_manager=risk_manager,
            logger=logger,
            journal=journal,
        )
        if args.kill_switch or input_kill_switch:
            executor.set_kill_switch(True)

        results: list[ExecutionResult] = []
        accepted_orders: list[str] = []
        for intent in intents:
            result = executor.submit_intent(intent)
            results.append(result)
            if result.order is not None:
                accepted_orders.append(result.order.order_id)

        if args.cancel_first_accepted and accepted_orders:
            results.append(executor.cancel_dry_run_order(accepted_orders[0]))

        accepted = sum(1 for result in results if result.status == "accepted")
        rejected = sum(1 for result in results if result.status == "rejected")
        cancelled = sum(1 for result in results if result.status == "cancelled")
        open_orders = executor.list_open_dry_run_orders()

        console.print(
            f"Processed {len(results)} actions | accepted={accepted} "
            f"rejected={rejected} cancelled={cancelled} open_orders={len(open_orders)}"
        )
        _print_results(console, results=results, max_print=args.max_print)

        journal.write_event(
            "dry_run_execution_summary",
            payload={
                "actions_processed": len(results),
                "accepted": accepted,
                "rejected": rejected,
                "cancelled": cancelled,
                "open_orders": len(open_orders),
                "open_order_ids": [order.order_id for order in open_orders],
            },
            metadata={"session_id": session_id},
        )
    except (ValueError, ValidationError, JournalError) as exc:
        exit_code = 4
        logger.error("Day 4 dry-run flow failed: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "day4_failure",
                payload={"error": sanitize_text(str(exc))},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write day4_failure event.")
    except Exception as exc:  # pragma: no cover - runtime safety guard
        exit_code = 99
        logger.exception("Unexpected Day 4 failure: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "day4_failure_unhandled",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write day4_failure_unhandled event.")
    finally:
        if journal is not None:
            try:
                journal.write_event(
                    "day4_shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write day4_shutdown event.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

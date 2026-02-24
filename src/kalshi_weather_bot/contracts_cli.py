"""Day 3 CLI: audit Kalshi contract parsing quality with confidence scoring."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import load_settings
from .contracts import (
    KalshiWeatherContractParser,
    ParsedWeatherContract,
    summarize_parse_results,
)
from .exceptions import ConfigError, ContractParserError, JournalError, KalshiAPIError
from .journal import JournalWriter
from .kalshi_client import KalshiClient
from .log_setup import setup_logger
from .redaction import sanitize_text


def parse_args() -> argparse.Namespace:
    """Parse Day 3 contract parser audit CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Audit Kalshi weather contract parsing from file or API (Day 3)."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to a raw Kalshi markets JSON snapshot. If omitted, fetch from API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Markets fetch limit when using API mode.",
    )
    parser.add_argument(
        "--only-weather",
        action="store_true",
        help="Only print weather-candidate contracts in examples.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum parse confidence to include in printed examples.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=5,
        help="Maximum examples to print per status bucket.",
    )
    return parser.parse_args()


def _load_payload_from_file(input_file: Path) -> dict[str, Any] | list[Any]:
    if not input_file.exists():
        raise ContractParserError(f"Input file does not exist: {input_file}")
    try:
        with input_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractParserError(f"Failed reading input file {input_file}: {exc}") from exc
    if not isinstance(payload, (dict, list)):
        raise ContractParserError(
            f"Input file payload must be dict/list, got {type(payload).__name__}."
        )
    return payload


def _extract_market_records(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    else:
        found = False
        records = []
        for key in ("markets", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                found = True
                break
        if not found:
            # Fallback: first list value under an unexpected key.
            for value in payload.values():
                if isinstance(value, list):
                    records = value
                    found = True
                    break
        if not found:
            raise ContractParserError(
                "No market records found in payload. Expected top-level list or a list field."
            )
    # Filter to dict records; an empty source list is valid (zero markets),
    # but a non-empty list with no objects indicates malformed payload shape.
    dict_records = [record for record in records if isinstance(record, dict)]
    if records and not dict_records:
        raise ContractParserError(
            "Market records list existed but contained no object records."
        )
    return dict_records


def _format_threshold(record: ParsedWeatherContract) -> str:
    if (
        record.threshold_operator in {"between", "range"}
        and record.threshold_low is not None
        and record.threshold_high is not None
    ):
        unit = f" {record.threshold_unit}" if record.threshold_unit else ""
        return (
            f"{record.threshold_operator} "
            f"{record.threshold_low:g}-{record.threshold_high:g}{unit}"
        )
    if record.threshold_value is not None and record.threshold_operator:
        unit = f" {record.threshold_unit}" if record.threshold_unit else ""
        return f"{record.threshold_operator} {record.threshold_value:g}{unit}"
    return "-"


def _print_summary(console: Console, summary_payload: dict[str, Any]) -> None:
    console.print(
        "Scanned "
        f"{summary_payload['total_markets_scanned']} markets | "
        f"weather candidates={summary_payload['weather_candidates']} | "
        f"parsed={summary_payload['parsed']} | "
        f"ambiguous={summary_payload['ambiguous']} | "
        f"unsupported/rejected={summary_payload['unsupported'] + summary_payload['rejected']}"
    )
    top_reasons = summary_payload.get("top_rejection_reasons") or {}
    if top_reasons:
        reason_text = ", ".join(f"{reason}:{count}" for reason, count in top_reasons.items())
        console.print(f"Top rejection reasons: {reason_text}")


def _print_examples(
    console: Console,
    records: list[ParsedWeatherContract],
    *,
    status: str,
    max_print: int,
) -> None:
    matches = [record for record in records if record.parse_status == status][:max_print]
    if not matches:
        return

    table = Table(title=f"Examples: {status}")
    table.add_column("Ticker/ID", overflow="fold")
    table.add_column("Confidence")
    table.add_column("Dimension")
    table.add_column("Threshold")
    table.add_column("Location", overflow="fold")
    table.add_column("Title", overflow="fold")

    for record in matches:
        location = record.location_raw or "-"
        table.add_row(
            record.ticker or record.provider_market_id,
            f"{record.parse_confidence:.2f}",
            record.weather_dimension or "-",
            _format_threshold(record),
            location,
            record.raw_title,
        )
    console.print(table)


def main() -> int:
    """Run Day 3 contract parser audit flow."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    if args.max_print <= 0:
        logger.error("--max-print must be > 0.")
        return 2
    if args.min_confidence < 0 or args.min_confidence > 1:
        logger.error("--min-confidence must be between 0 and 1.")
        return 2
    if args.limit is not None and args.limit <= 0:
        logger.error("--limit must be > 0 when provided.")
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
            "contract_parse_start",
            payload={
                "source_mode": "file" if args.input_file else "api",
                "input_file": str(args.input_file) if args.input_file else None,
                "only_weather": args.only_weather,
                "min_confidence": args.min_confidence,
                "max_print": args.max_print,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize journal: %s", exc)
        return 3

    exit_code = 0
    try:
        raw_payload: dict[str, Any] | list[Any]
        if args.input_file:
            raw_payload = _load_payload_from_file(args.input_file)
            journal.write_event(
                "contract_parse_input_file_loaded",
                payload={"path": str(args.input_file)},
                metadata={"session_id": session_id},
            )
        else:
            with KalshiClient(settings=settings, logger=logger) as client:
                client.validate_connection()
                raw_payload = client.fetch_markets_raw(limit=args.limit)
            raw_path = journal.write_raw_snapshot("markets_for_parse", raw_payload)
            journal.write_event(
                "contract_parse_input_snapshot",
                payload={"path": str(raw_path)},
                metadata={"session_id": session_id},
            )

        records = _extract_market_records(raw_payload)
        parser = KalshiWeatherContractParser(logger=logger)
        parsed = parser.parse_markets(records)
        summary = summarize_parse_results(parsed)
        summary_payload = summary.model_dump(mode="json")
        _print_summary(console, summary_payload)

        visible = [
            record
            for record in parsed
            if (not args.only_weather or record.weather_candidate)
            and record.parse_confidence >= args.min_confidence
        ]
        for status in ("parsed", "ambiguous", "rejected", "unsupported"):
            if args.only_weather and status == "unsupported":
                continue
            _print_examples(console, visible, status=status, max_print=args.max_print)

        rejection_samples = [
            {
                "ticker": record.ticker,
                "provider_market_id": record.provider_market_id,
                "parse_status": record.parse_status,
                "confidence": record.parse_confidence,
                "rejection_reasons": record.rejection_reasons,
                "title": record.raw_title,
            }
            for record in parsed
            if record.parse_status in {"rejected", "ambiguous"}
        ][:20]

        journal.write_event(
            "contract_parse_summary",
            payload={
                **summary_payload,
                "visible_after_filters": len(visible),
                "only_weather": args.only_weather,
                "min_confidence": args.min_confidence,
            },
            metadata={"session_id": session_id},
        )
        if rejection_samples:
            journal.write_event(
                "contract_parse_rejection_sample",
                payload={"samples": rejection_samples},
                metadata={"session_id": session_id},
            )
    except (KalshiAPIError, ContractParserError, JournalError) as exc:
        exit_code = 4
        logger.error("Contract parse audit failure: %s", exc)
        try:
            journal.write_event(
                "contract_parse_failure",
                payload={"error": sanitize_text(str(exc))},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to journal contract_parse_failure.")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        exit_code = 99
        logger.exception("Unexpected contract parse audit failure: %s", exc)
        try:
            journal.write_event(
                "contract_parse_failure_unhandled",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to journal contract_parse_failure_unhandled.")
    finally:
        if journal is not None:
            try:
                journal.write_event(
                    "contract_parse_shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write contract_parse_shutdown event.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

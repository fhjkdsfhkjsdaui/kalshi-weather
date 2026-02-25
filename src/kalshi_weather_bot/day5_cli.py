"""Day 5 CLI: signal evaluation pipeline to ranked dry-run trade candidates."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .contracts.parser import KalshiWeatherContractParser
from .exceptions import (
    ConfigError,
    JournalError,
    KalshiAPIError,
    SignalProcessingError,
    WeatherProviderError,
)
from .execution.dry_run import DryRunExecutor
from .execution.models import ExecutionResult, TradeIntent
from .journal import JournalWriter
from .kalshi_client import KalshiClient
from .log_setup import setup_logger
from .redaction import sanitize_text
from .risk.manager import RiskManager
from .signal.edge import EdgeCalculator
from .signal.estimator import WeatherProbabilityEstimator
from .signal.matcher import WeatherMarketMatcher
from .signal.models import SignalEvaluation, SignalRejection
from .signal.pricing import parse_cents_field, parse_dollars_field
from .signal.selector import CandidateSelector
from .weather.models import WeatherSnapshot
from .weather.nws import NWSWeatherProvider

_MAX_DETAIL_EVENTS = 150


def parse_args() -> argparse.Namespace:
    """Parse Day 5 CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Day 5 signal evaluation + candidate selection + dry-run submission."
    )
    parser.add_argument("--input-markets-file", type=Path, default=None)
    parser.add_argument("--input-weather-file", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--min-edge", type=float, default=None)
    parser.add_argument("--min-model-confidence", type=float, default=None)
    parser.add_argument("--max-markets-to-scan", type=int, default=None)
    parser.add_argument("--print-rejections", type=int, default=10)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--weather-url", type=str, default=None)
    parser.add_argument(
        "--forecast-type",
        choices=["daily", "hourly"],
        default="hourly",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run through Day 4 dry-run executor (must remain true for Day 5).",
    )
    return parser.parse_args()


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        raise SignalProcessingError(f"File not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SignalProcessingError(f"Failed reading JSON file {path}: {exc}") from exc


def _extract_market_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        found: list[Any] | None = None
        for key in ("markets", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                found = value
                break
        if found is None:
            for value in payload.values():
                if isinstance(value, list):
                    found = value
                    break
        if found is None:
            raise SignalProcessingError("Unable to find market records list in input payload.")
        records = found
    else:
        raise SignalProcessingError(
            f"Unsupported market payload type: {type(payload).__name__} (expected dict/list)."
        )

    dict_records = [record for record in records if isinstance(record, dict)]
    if records and not dict_records:
        raise SignalProcessingError("Market records list contained no object records.")
    return dict_records


def _load_weather_snapshot_from_file(path: Path) -> WeatherSnapshot:
    payload = _load_json_file(path)
    snapshot_payload: Any
    if isinstance(payload, dict) and "snapshot" in payload:
        snapshot_payload = payload["snapshot"]
    else:
        snapshot_payload = payload
    try:
        return WeatherSnapshot.model_validate(snapshot_payload)
    except ValidationError as exc:
        raise SignalProcessingError(f"Invalid weather snapshot payload in {path}: {exc}") from exc


def _resolve_weather_snapshot(
    *,
    args: argparse.Namespace,
    settings: Settings,
    logger: Any,
    journal: JournalWriter,
) -> WeatherSnapshot:
    if args.input_weather_file:
        snapshot = _load_weather_snapshot_from_file(args.input_weather_file)
        journal.write_event(
            "signal_weather_input_loaded",
            payload={"source": "file", "path": str(args.input_weather_file)},
        )
        return snapshot

    lat = args.lat if args.lat is not None else settings.weather_default_lat
    lon = args.lon if args.lon is not None else settings.weather_default_lon
    if not args.weather_url and (lat is None or lon is None):
        raise SignalProcessingError(
            "Weather input missing. Provide --input-weather-file or --weather-url or --lat/--lon "
            "(or WEATHER_DEFAULT_LAT/LON)."
        )

    with NWSWeatherProvider(settings=settings, logger=logger) as provider:
        fetch_result = provider.fetch_forecast(
            lat=lat,
            lon=lon,
            forecast_url=args.weather_url,
            forecast_type=args.forecast_type,
        )
        if settings.weather_journal_raw_payloads:
            points_path: str | None = None
            if fetch_result.raw_points_payload is not None:
                points_path = str(
                    journal.write_raw_snapshot("day5_nws_points", fetch_result.raw_points_payload)
                )
            forecast_path = str(
                journal.write_raw_snapshot("day5_nws_forecast", fetch_result.raw_forecast_payload)
            )
            journal.write_event(
                "signal_weather_raw_snapshot",
                payload={"points_path": points_path, "forecast_path": forecast_path},
            )
        return fetch_result.snapshot


def _resolve_market_records(
    *,
    args: argparse.Namespace,
    settings: Settings,
    logger: Any,
    journal: JournalWriter,
) -> list[dict[str, Any]]:
    records, _diagnostics = _resolve_market_records_with_diagnostics(
        args=args,
        settings=settings,
        logger=logger,
        journal=journal,
    )
    return records


def _resolve_market_records_with_diagnostics(
    *,
    args: argparse.Namespace,
    settings: Settings,
    logger: Any,
    journal: JournalWriter,
    weather_candidate_target: int | None = None,
    weather_candidate_predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.input_markets_file:
        payload = _load_json_file(args.input_markets_file)
        records = _extract_market_records(payload)
        diagnostics = _extract_market_fetch_diagnostics(payload, record_count=len(records))
        journal.write_event(
            "signal_markets_input_loaded",
            payload={
                "source": "file",
                "path": str(args.input_markets_file),
                "count": len(records),
                **diagnostics,
            },
        )
        return records, diagnostics

    limit = args.max_markets_to_scan or settings.signal_max_markets_to_scan
    with KalshiClient(settings=settings, logger=logger) as client:
        client.validate_connection()
        payload = client.fetch_markets_raw(
            limit=limit,
            candidate_target=weather_candidate_target,
            candidate_predicate=weather_candidate_predicate,
        )
    records = _extract_market_records(payload)
    diagnostics = _extract_market_fetch_diagnostics(payload, record_count=len(records))
    raw_path = journal.write_raw_snapshot("day5_markets", payload)
    journal.write_event(
        "signal_markets_input_loaded",
        payload={
            "source": "api",
            "limit": limit,
            "raw_snapshot_path": str(raw_path),
            **diagnostics,
        },
    )
    return records, diagnostics


def _extract_market_fetch_diagnostics(
    payload: Any,
    *,
    record_count: int,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "pages_fetched": 1,
        "total_markets_fetched": record_count,
        "had_more_pages": False,
        "deduped_count": 0,
    }
    if not isinstance(payload, dict):
        return diagnostics

    cursor_present = bool(
        payload.get("cursor") or payload.get("next_cursor") or payload.get("next")
    )
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        pages_fetched = pagination.get("pages_fetched")
        if isinstance(pages_fetched, int) and pages_fetched > 0:
            diagnostics["pages_fetched"] = pages_fetched
        total_markets = pagination.get("total_markets_fetched")
        if isinstance(total_markets, int) and total_markets >= 0:
            diagnostics["total_markets_fetched"] = total_markets
        had_more_pages = pagination.get("had_more_pages")
        if isinstance(had_more_pages, bool):
            diagnostics["had_more_pages"] = had_more_pages
        deduped_count = pagination.get("deduped_count")
        if isinstance(deduped_count, int) and deduped_count >= 0:
            diagnostics["deduped_count"] = deduped_count
        return diagnostics

    diagnostics["had_more_pages"] = cursor_present
    return diagnostics


def _market_spread_cents(market: dict[str, Any]) -> int | None:
    ask = parse_cents_field(market.get("yes_ask"))
    bid = parse_cents_field(market.get("yes_bid"))
    if ask is None:
        ask = parse_dollars_field(market.get("yes_ask_dollars"))
    if bid is None:
        bid = parse_dollars_field(market.get("yes_bid_dollars"))
    if ask is None or bid is None:
        return None
    return max(0, ask - bid)


def _market_liquidity(market: dict[str, Any]) -> int | None:
    value = market.get("liquidity")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _print_candidate_table(console: Console, results: list[ExecutionResult], max_rows: int) -> None:
    table = Table(title="Day 5 Dry-Run Actions")
    table.add_column("Market", overflow="fold")
    table.add_column("Side")
    table.add_column("Price")
    table.add_column("Status")
    table.add_column("Decision")
    table.add_column("Reasons", overflow="fold")
    for result in results[:max_rows]:
        market = result.intent.market_id if result.intent else "-"
        side = result.intent.side if result.intent else "-"
        price = str(result.intent.price_cents) if result.intent else "-"
        reasons = ", ".join(result.reasons) if result.reasons else "-"
        table.add_row(market, side, price, result.status, result.decision_code, reasons)
    console.print(table)


def _print_rejection_table(
    console: Console,
    rejections: list[SignalRejection],
    max_rows: int,
) -> None:
    table = Table(title="Day 5 Signal Rejections")
    table.add_column("Market", overflow="fold")
    table.add_column("Stage")
    table.add_column("Reason")
    table.add_column("Details", overflow="fold")
    for rejection in rejections[:max_rows]:
        details = ", ".join(rejection.reasons[:3]) if rejection.reasons else "-"
        table.add_row(
            rejection.market_id,
            rejection.stage,
            rejection.reason_code,
            details,
        )
    console.print(table)


def main() -> int:
    """Run Day 5 signal pipeline end-to-end in dry-run mode."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    if args.max_candidates is not None and args.max_candidates <= 0:
        logger.error("--max-candidates must be > 0.")
        return 2
    if args.print_rejections <= 0:
        logger.error("--print-rejections must be > 0.")
        return 2
    if args.max_markets_to_scan is not None and args.max_markets_to_scan <= 0:
        logger.error("--max-markets-to-scan must be > 0.")
        return 2
    if not args.dry_run:
        logger.error("Day 5 supports dry-run mode only. Use --dry-run.")
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
            "signal_scan_start",
            payload={
                "input_markets_file": (
                    str(args.input_markets_file) if args.input_markets_file else None
                ),
                "input_weather_file": (
                    str(args.input_weather_file) if args.input_weather_file else None
                ),
                "max_candidates": args.max_candidates or settings.signal_max_candidates,
                "max_markets_to_scan": (
                    args.max_markets_to_scan or settings.signal_max_markets_to_scan
                ),
                "min_edge": (
                    args.min_edge
                    if args.min_edge is not None
                    else settings.signal_min_edge
                ),
                "min_model_confidence": (
                    args.min_model_confidence
                    if args.min_model_confidence is not None
                    else settings.signal_min_model_confidence
                ),
                "dry_run": args.dry_run,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize day5 journal: %s", exc)
        return 3

    exit_code = 0
    try:
        now = datetime.now(UTC)
        scan_limit = args.max_markets_to_scan or settings.signal_max_markets_to_scan
        market_records = _resolve_market_records(
            args=args,
            settings=settings,
            logger=logger,
            journal=journal,
        )
        weather_snapshot = _resolve_weather_snapshot(
            args=args,
            settings=settings,
            logger=logger,
            journal=journal,
        )
        weather_ref = weather_snapshot.raw_payload_path or weather_snapshot.source_url

        parser = KalshiWeatherContractParser(logger=logger)
        matcher = WeatherMarketMatcher(logger=logger)
        estimator = WeatherProbabilityEstimator(logger=logger)
        edge_calculator = EdgeCalculator(settings=settings, logger=logger)
        selector = CandidateSelector(settings=settings, logger=logger)

        evaluations: list[SignalEvaluation] = []
        eval_by_market_id: dict[str, SignalEvaluation] = {}
        detail_event_budget = _MAX_DETAIL_EVENTS

        for raw_market in market_records[:scan_limit]:
            parsed = parser.parse_market(raw_market)
            market_id = parsed.provider_market_id
            title = parsed.raw_title

            match_result = matcher.match_contract(parsed, weather_snapshot)
            estimate_result = estimator.estimate(parsed, match_result)
            edge_result = edge_calculator.compute(raw_market, estimate_result)
            weather_age_seconds = (now - weather_snapshot.retrieval_timestamp).total_seconds()
            evaluation = SignalEvaluation(
                market_id=market_id,
                ticker=parsed.ticker,
                title=title,
                market_raw=raw_market,
                parsed_contract=parsed,
                weather_snapshot_ref=weather_ref,
                weather_snapshot_retrieved_at=weather_snapshot.retrieval_timestamp,
                weather_age_seconds=weather_age_seconds,
                match_result=match_result,
                estimate_result=estimate_result,
                edge_result=edge_result,
            )
            evaluations.append(evaluation)
            eval_by_market_id[market_id] = evaluation

            if detail_event_budget > 0:
                journal.write_event(
                    "signal_match_result",
                    payload={
                        "market_id": market_id,
                        "matched": match_result.matched,
                        "decision_code": match_result.decision_code,
                        "matcher_confidence": match_result.matcher_confidence,
                        "selected_period_count": match_result.selected_period_count,
                        "reasons": match_result.reasons[:3],
                    },
                )
                journal.write_event(
                    "signal_estimate_result",
                    payload={
                        "market_id": market_id,
                        "available": estimate_result.available,
                        "decision_code": estimate_result.decision_code,
                        "estimate_method": estimate_result.estimate_method,
                        "estimated_probability": estimate_result.estimated_probability,
                        "model_confidence": estimate_result.model_confidence,
                        "missing_data_flags": estimate_result.missing_data_flags[:3],
                    },
                )
                journal.write_event(
                    "signal_edge_result",
                    payload={
                        "market_id": market_id,
                        "valid": edge_result.valid,
                        "decision_code": edge_result.decision_code,
                        "recommended_side": edge_result.recommended_side,
                        "edge_after_buffers": edge_result.edge_after_buffers,
                        "reasons": edge_result.reasons[:3],
                    },
                )
                detail_event_budget -= 3

        selection = selector.select(
            evaluations,
            max_candidates=args.max_candidates or settings.signal_max_candidates,
            min_edge_override=args.min_edge,
            min_model_confidence_override=args.min_model_confidence,
        )

        rejection_samples = selection.rejected[: args.print_rejections]
        for rejection in rejection_samples:
            journal.write_event(
                "signal_candidate_rejected",
                payload=rejection.model_dump(mode="json"),
            )
        for candidate in selection.selected:
            journal.write_event(
                "signal_candidate_selected",
                payload={
                    "market_id": candidate.market_id,
                    "side": candidate.side,
                    "price_cents": candidate.price_cents,
                    "score": candidate.score,
                    "edge_after_buffers": candidate.edge_result.edge_after_buffers,
                    "model_confidence": candidate.estimate_result.model_confidence,
                },
            )

        risk_manager = RiskManager(settings=settings, logger=logger)
        executor = DryRunExecutor(
            settings=settings,
            risk_manager=risk_manager,
            logger=logger,
            journal=journal,
        )

        action_results: list[ExecutionResult] = []
        risk_rejected = 0
        dry_run_submitted = 0
        for candidate in selection.selected:
            evaluation = eval_by_market_id[candidate.market_id]
            intent = TradeIntent(
                market_id=candidate.market_id,
                side=candidate.side,
                price_cents=candidate.price_cents,
                quantity=candidate.quantity,
                parser_confidence=candidate.parsed_contract.parse_confidence,
                parsed_contract_ref=candidate.parsed_contract.provider_market_id,
                weather_snapshot_ref=evaluation.weather_snapshot_ref,
                weather_snapshot_retrieved_at=evaluation.weather_snapshot_retrieved_at,
                timestamp=now,
                strategy_tag="day5_signal_loop",
                market_liquidity=_market_liquidity(evaluation.market_raw),
                market_spread_cents=_market_spread_cents(evaluation.market_raw),
                metadata={"signal_score": candidate.score},
            )
            result = executor.submit_intent(intent)
            action_results.append(result)
            if result.status == "accepted":
                dry_run_submitted += 1
            elif result.status == "rejected":
                risk_rejected += 1

        weather_candidates = sum(
            1 for item in evaluations if item.parsed_contract.weather_candidate
        )
        matched_count = sum(1 for item in evaluations if item.match_result.matched)
        estimated_count = sum(1 for item in evaluations if item.estimate_result.available)
        filtered_out = len(selection.rejected)

        console.print(
            f"scanned={len(evaluations)} weather_candidates={weather_candidates} "
            f"matched={matched_count} estimated={estimated_count} filtered={filtered_out} "
            f"risk_rejected={risk_rejected} dry_run_submitted={dry_run_submitted}"
        )
        _print_candidate_table(console, results=action_results, max_rows=10)
        if rejection_samples:
            _print_rejection_table(
                console,
                rejections=rejection_samples,
                max_rows=args.print_rejections,
            )

        journal.write_event(
            "signal_batch_summary",
            payload={
                "markets_scanned": len(evaluations),
                "weather_candidates": weather_candidates,
                "matched": matched_count,
                "estimated": estimated_count,
                "filtered_out": filtered_out,
                "selected_candidates": len(selection.selected),
                "risk_rejected": risk_rejected,
                "dry_run_submitted": dry_run_submitted,
                "rejection_samples_count": len(rejection_samples),
            },
            metadata={"session_id": session_id},
        )
    except (
        SignalProcessingError,
        WeatherProviderError,
        KalshiAPIError,
        JournalError,
        ValidationError,
    ) as exc:
        exit_code = 4
        logger.error("Day 5 signal flow failed: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "signal_failure",
                payload={"error": sanitize_text(str(exc))},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write signal_failure event.")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        exit_code = 99
        logger.exception("Unexpected Day 5 failure: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "signal_failure_unhandled",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write signal_failure_unhandled event.")
    finally:
        if journal is not None:
            try:
                journal.write_event(
                    "signal_scan_shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write signal_scan_shutdown event.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

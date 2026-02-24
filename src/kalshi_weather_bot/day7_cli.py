"""Day 7 CLI: supervised micro-live execution over Day 5 signal candidates."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .contracts.parser import KalshiWeatherContractParser
from .day5_cli import (
    _market_liquidity,
    _market_spread_cents,
    _resolve_market_records,
    _resolve_weather_snapshot,
)
from .exceptions import (
    ConfigError,
    JournalError,
    KalshiAPIError,
    SignalProcessingError,
    WeatherProviderError,
)
from .execution.dry_run import DryRunExecutor
from .execution.live_adapter import KalshiLiveOrderAdapter, OrderAdapterError
from .execution.live_micro import LiveMicroRunner
from .execution.micro_models import (
    MicroOrderAttempt,
    MicroSessionResult,
    MicroTradeCandidate,
)
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
from .signal.selector import CandidateSelector


def parse_args() -> argparse.Namespace:
    """Parse Day 7 operator CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Day 7 supervised micro-live session using Day 5 candidate pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "live_cancel_only", "live_micro"],
        default=None,
    )
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-trades-this-run", type=int, default=None)
    parser.add_argument("--market-scope", type=str, default=None)
    parser.add_argument("--city-whitelist", type=str, default=None)
    parser.add_argument("--supervised", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--forecast-type", choices=["daily", "hourly"], default="hourly")
    parser.add_argument("--poll-timeout-seconds", type=float, default=None)
    parser.add_argument("--poll-interval-seconds", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force Day 7 runner into dry_run mode for offline verification.",
    )
    return parser.parse_args()


def _combine_scope_filters(settings: Settings, args: argparse.Namespace) -> str | None:
    tokens: list[str] = []
    for source in [settings.micro_market_scope_whitelist, args.market_scope, args.city_whitelist]:
        if not source:
            continue
        tokens.extend(part.strip() for part in source.split(",") if part.strip())
    if not tokens:
        return None

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return ",".join(deduped)


def _evaluate_signal_candidates(
    *,
    args: argparse.Namespace,
    settings: Settings,
    logger: Any,
    journal: JournalWriter,
) -> tuple[list[MicroTradeCandidate], list[SignalRejection], dict[str, int]]:
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

    for raw_market in market_records[:scan_limit]:
        parsed = parser.parse_market(raw_market)
        market_id = parsed.provider_market_id
        match_result = matcher.match_contract(parsed, weather_snapshot)
        estimate_result = estimator.estimate(parsed, match_result)
        edge_result = edge_calculator.compute(raw_market, estimate_result)
        weather_age_seconds = (now - weather_snapshot.retrieval_timestamp).total_seconds()
        evaluation = SignalEvaluation(
            market_id=market_id,
            ticker=parsed.ticker,
            title=parsed.raw_title,
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

    selection = selector.select(
        evaluations,
        max_candidates=args.max_candidates or settings.signal_max_candidates,
        min_edge_override=args.min_edge,
        min_model_confidence_override=args.min_model_confidence,
    )

    candidates: list[MicroTradeCandidate] = []
    for selected in selection.selected:
        evaluation = eval_by_market_id[selected.market_id]
        location_hint = selected.parsed_contract.location_raw
        if not location_hint and selected.parsed_contract.location_normalized:
            parts = [
                selected.parsed_contract.location_normalized.city,
                selected.parsed_contract.location_normalized.state,
            ]
            location_hint = ", ".join(part for part in parts if part)

        candidates.append(
            MicroTradeCandidate(
                market_id=selected.market_id,
                ticker=selected.ticker,
                title=selected.title,
                side=selected.side,
                price_cents=selected.price_cents,
                quantity=selected.quantity,
                parser_confidence=selected.parsed_contract.parse_confidence,
                edge_after_buffers=selected.edge_result.edge_after_buffers or 0.0,
                weather_age_seconds=evaluation.weather_age_seconds,
                parsed_contract_ref=selected.parsed_contract.provider_market_id,
                weather_snapshot_ref=evaluation.weather_snapshot_ref,
                market_liquidity=_market_liquidity(evaluation.market_raw),
                market_spread_cents=_market_spread_cents(evaluation.market_raw),
                strategy_tag="day7_micro_session",
                location_hint=location_hint,
            )
        )

    counts = {
        "scanned": len(evaluations),
        "weather_candidates": sum(
            1 for item in evaluations if item.parsed_contract.weather_candidate
        ),
        "matched": sum(1 for item in evaluations if item.match_result.matched),
        "estimated": sum(1 for item in evaluations if item.estimate_result.available),
        "filtered": len(selection.rejected),
        "selected": len(selection.selected),
    }
    return candidates, selection.rejected, counts


def _build_live_micro_runner(
    *,
    settings: Settings,
    logger: Any,
    journal: JournalWriter,
) -> tuple[LiveMicroRunner, KalshiClient]:
    client = KalshiClient(settings=settings, logger=logger)
    adapter = KalshiLiveOrderAdapter(client=client)
    runner = LiveMicroRunner(
        settings=settings,
        adapter=adapter,
        logger=logger,
        journal=journal,
    )
    return runner, client


def _run_dry_run_mode(
    *,
    settings: Settings,
    candidates: list[MicroTradeCandidate],
    max_trades_this_run: int,
    now: datetime,
    logger: Any,
    journal: JournalWriter,
) -> tuple[list[ExecutionResult], int]:
    risk_manager = RiskManager(settings=settings, logger=logger)
    executor = DryRunExecutor(
        settings=settings,
        risk_manager=risk_manager,
        logger=logger,
        journal=journal,
    )

    results: list[ExecutionResult] = []
    for candidate in candidates[:max_trades_this_run]:
        intent = TradeIntent(
            market_id=candidate.market_id,
            side=candidate.side,
            price_cents=candidate.price_cents,
            quantity=candidate.quantity,
            parser_confidence=candidate.parser_confidence,
            parsed_contract_ref=candidate.parsed_contract_ref,
            weather_snapshot_ref=candidate.weather_snapshot_ref,
            weather_snapshot_retrieved_at=now,
            timestamp=now,
            strategy_tag=candidate.strategy_tag,
            market_liquidity=candidate.market_liquidity,
            market_spread_cents=candidate.market_spread_cents,
            metadata={"edge_after_buffers": candidate.edge_after_buffers},
        )
        results.append(executor.submit_intent(intent))

    accepted = sum(1 for item in results if item.status == "accepted")
    return results, accepted


def _print_micro_attempts(
    console: Console,
    attempts: list[MicroOrderAttempt],
    max_rows: int,
) -> None:
    table = Table(title="Day 7 Micro Attempts")
    table.add_column("Market", overflow="fold")
    table.add_column("Side")
    table.add_column("Price")
    table.add_column("Outcome")
    table.add_column("Order")
    table.add_column("Reasons", overflow="fold")
    for attempt in attempts[:max_rows]:
        reasons = ", ".join(attempt.reasons[:3]) if attempt.reasons else "-"
        table.add_row(
            attempt.candidate.market_id,
            attempt.candidate.side,
            str(attempt.candidate.price_cents),
            attempt.outcome,
            attempt.order_id or "-",
            reasons,
        )
    console.print(table)


def main() -> int:
    """Run Day 7 supervised signal->execution session."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    if args.max_cycles <= 0:
        logger.error("--max-cycles must be > 0.")
        return 2
    if args.max_candidates is not None and args.max_candidates <= 0:
        logger.error("--max-candidates must be > 0.")
        return 2
    if args.max_markets_to_scan is not None and args.max_markets_to_scan <= 0:
        logger.error("--max-markets-to-scan must be > 0.")
        return 2
    if args.print_rejections <= 0:
        logger.error("--print-rejections must be > 0.")
        return 2
    if args.max_trades_this_run is not None and args.max_trades_this_run <= 0:
        logger.error("--max-trades-this-run must be > 0.")
        return 2

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("Configuration failure: %s", sanitize_text(str(exc)))
        return 2

    scope = _combine_scope_filters(settings, args)
    if scope != settings.micro_market_scope_whitelist:
        settings = settings.model_copy(update={"micro_market_scope_whitelist": scope})

    mode = args.mode or settings.execution_mode
    if args.dry_run is True:
        mode = "dry_run"
    if args.dry_run is False and mode == "dry_run":
        logger.error("--no-dry-run requires --mode live_micro.")
        return 2
    if mode == "live_cancel_only":
        logger.error("Day 7 CLI does not run cancel-only mode; use kalshi-weather-day6.")
        return 2

    max_trades_this_run = args.max_trades_this_run or settings.micro_max_trades_per_run

    try:
        journal = JournalWriter(
            journal_dir=settings.journal_dir,
            raw_payload_dir=settings.raw_payload_dir,
            session_id=session_id,
        )
        journal.write_event(
            "micro_session_start",
            payload={
                "mode": mode,
                "max_cycles": args.max_cycles,
                "max_trades_this_run": max_trades_this_run,
                "supervised": args.supervised,
                "market_scope": scope,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize day7 journal: %s", sanitize_text(str(exc)))
        return 3

    exit_code = 0
    client: KalshiClient | None = None
    micro_runner: LiveMicroRunner | None = None
    attempts_all: list[MicroOrderAttempt] = []
    total_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    final_summary: dict[str, Any] = {}

    try:
        if mode == "live_micro" and settings.micro_require_supervised_mode and not args.supervised:
            logger.error(
                "Unsafe startup: --supervised is required by "
                "MICRO_REQUIRE_SUPERVISED_MODE."
            )
            return 2

        if mode == "live_micro":
            micro_runner, client = _build_live_micro_runner(
                settings=settings,
                logger=logger,
                journal=journal,
            )

        for _cycle_index in range(1, args.max_cycles + 1):
            candidates, rejections, counts = _evaluate_signal_candidates(
                args=args,
                settings=settings,
                logger=logger,
                journal=journal,
            )
            total_counts.update({
                "cycles_processed": 1,
                "candidates_seen": len(candidates),
                "signal_scanned": counts["scanned"],
                "signal_filtered": counts["filtered"],
            })

            for rejection in rejections[: args.print_rejections]:
                journal.write_event(
                    "signal_candidate_rejected",
                    payload=rejection.model_dump(mode="json"),
                )

            if mode == "dry_run":
                now = datetime.now(UTC)
                dry_results, accepted = _run_dry_run_mode(
                    settings=settings,
                    candidates=candidates,
                    max_trades_this_run=max_trades_this_run,
                    now=now,
                    logger=logger,
                    journal=journal,
                )
                total_counts.update(
                    {
                        "orders_submitted": accepted,
                        "trades_allowed": accepted,
                        "fills": 0,
                        "partial_fills": 0,
                        "cancels": 0,
                        "rejects": sum(1 for item in dry_results if item.status == "rejected"),
                        "unresolved": 0,
                    }
                )
                final_summary = {
                    "mode": "dry_run",
                    "open_positions_count": 0,
                    "realized_pnl_cents": 0,
                    "daily_gross_exposure_cents": 0,
                    "halted_early": False,
                    "halt_reason": None,
                    "gross_exposure_utilization": 0.0,
                    "realized_loss_utilization": 0.0,
                    "trades_per_run_utilization": accepted / max_trades_this_run,
                    "trades_per_day_utilization": 0.0,
                }
                continue

            if micro_runner is None:  # pragma: no cover - defensive
                raise RuntimeError("live_micro runner not initialized.")
            micro_result: MicroSessionResult = micro_runner.run_candidates(
                candidates=candidates,
                max_trades_this_run=max_trades_this_run,
                poll_timeout_seconds=args.poll_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                supervised=args.supervised,
                cycles_processed=1,
            )
            attempts_all.extend(micro_result.attempts)
            summary = micro_result.summary
            total_counts.update(
                {
                    "trades_allowed": summary.trades_allowed,
                    "orders_submitted": summary.orders_submitted,
                    "fills": summary.fills,
                    "partial_fills": summary.partial_fills,
                    "cancels": summary.cancels,
                    "rejects": summary.rejects,
                    "unresolved": summary.unresolved,
                    "trades_skipped": summary.trades_skipped,
                }
            )
            skip_reasons.update(summary.skip_reasons)
            final_summary = summary.model_dump(mode="json")

            if summary.halted_early:
                exit_code = 4
                break

        if mode == "live_micro" and total_counts["unresolved"] > 0:
            exit_code = 4

        summary_line = (
            f"mode={mode} cycles={total_counts['cycles_processed']} "
            f"signal_scanned={total_counts['signal_scanned']} "
            f"candidates_seen={total_counts['candidates_seen']} "
            f"trades_allowed={total_counts['trades_allowed']} "
            f"trades_skipped={total_counts['trades_skipped']} "
            f"orders_submitted={total_counts['orders_submitted']} "
            f"fills={total_counts['fills']} "
            f"partial_fills={total_counts['partial_fills']} "
            f"cancels={total_counts['cancels']} "
            f"rejects={total_counts['rejects']} "
            f"unresolved={total_counts['unresolved']}"
        )
        console.print(summary_line)
        if attempts_all:
            _print_micro_attempts(console, attempts_all, max_rows=10)

        if skip_reasons:
            top_reasons = ", ".join(
                f"{reason}:{count}" for reason, count in skip_reasons.most_common(5)
            )
            console.print(f"skip_reasons={top_reasons}")

        if final_summary:
            console.print(
                f"open_positions={final_summary.get('open_positions_count', 0)} "
                f"realized_pnl_cents={final_summary.get('realized_pnl_cents', 0)} "
                f"daily_gross_exposure_cents={final_summary.get('daily_gross_exposure_cents', 0)} "
                f"gross_util={final_summary.get('gross_exposure_utilization', 0.0)} "
                f"loss_util={final_summary.get('realized_loss_utilization', 0.0)} "
                f"halted_early={final_summary.get('halted_early', False)} "
                f"halt_reason={final_summary.get('halt_reason') or '-'}"
            )

        journal.write_event(
            "micro_session_summary",
            payload={
                "mode": mode,
                "cycles_processed": total_counts["cycles_processed"],
                "signal_scanned": total_counts["signal_scanned"],
                "candidates_seen": total_counts["candidates_seen"],
                "trades_allowed": total_counts["trades_allowed"],
                "trades_skipped": total_counts["trades_skipped"],
                "orders_submitted": total_counts["orders_submitted"],
                "fills": total_counts["fills"],
                "partial_fills": total_counts["partial_fills"],
                "cancels": total_counts["cancels"],
                "rejects": total_counts["rejects"],
                "unresolved": total_counts["unresolved"],
                "skip_reasons": dict(skip_reasons),
                "final_summary": final_summary,
            },
            metadata={"session_id": session_id},
        )
    except (
        SignalProcessingError,
        WeatherProviderError,
        KalshiAPIError,
        OrderAdapterError,
        ValueError,
        JournalError,
    ) as exc:
        exit_code = 4
        logger.error("Day 7 session failed: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "micro_session_failure",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write micro_session_failure event.")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        exit_code = 99
        logger.exception("Unexpected Day 7 failure: %s", sanitize_text(str(exc)))
        try:
            journal.write_event(
                "micro_session_failure_unhandled",
                payload={"error": sanitize_text(str(exc)), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write micro_session_failure_unhandled event.")
    finally:
        if client is not None:
            client.close()
        if journal is not None:
            try:
                journal.write_event(
                    "micro_session_shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write micro_session_shutdown event.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

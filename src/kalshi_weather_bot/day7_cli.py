"""Day 7 CLI: supervised micro-live execution over Day 5 signal candidates."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .contracts.models import ParsedWeatherContract
from .contracts.parser import KalshiWeatherContractParser
from .day5_cli import (
    _market_liquidity,
    _market_spread_cents,
    _resolve_market_records_with_diagnostics,
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
from .ui.terminal_dashboard import TerminalDashboard

_MAX_WEATHER_DIAGNOSTIC_SAMPLES = 5


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
        "--ui-mode",
        choices=["plain", "rich"],
        default="plain",
        help="Terminal output mode; rich enables dashboard panels.",
    )
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


def _resolve_runtime_mode(
    *,
    settings: Settings,
    args: argparse.Namespace,
    logger: Any,
) -> tuple[Settings, str, bool]:
    """Resolve effective execution mode for this CLI run with safe, explicit precedence."""
    config_mode = settings.execution_mode
    cli_mode = args.mode
    effective_mode = cli_mode if cli_mode is not None else config_mode

    if args.dry_run is True:
        effective_dry_run = True
    elif args.dry_run is False:
        effective_dry_run = False
    else:
        effective_dry_run = effective_mode == "dry_run"

    if cli_mode is not None and cli_mode != config_mode:
        logger.warning(
            "Execution mode mismatch: config EXECUTION_MODE=%s, CLI --mode=%s. "
            "Using CLI mode for this run. To keep consistent, set EXECUTION_MODE=%s in .env.",
            config_mode,
            cli_mode,
            cli_mode,
        )

    if cli_mode == "live_micro" and effective_dry_run:
        raise ValueError(
            "Conflicting runtime flags: config EXECUTION_MODE="
            f"{config_mode}, CLI --mode=live_micro, effective dry_run=true. "
            "Use --no-dry-run for live micro execution."
        )

    if effective_dry_run:
        effective_mode = "dry_run"

    if effective_mode == "live_cancel_only":
        raise ValueError("Day 7 CLI does not run cancel-only mode; use kalshi-weather-day6.")

    if not effective_dry_run and effective_mode != "live_micro":
        raise ValueError(
            "Invalid runtime mode: dry_run=false requires live_micro mode. "
            f"config EXECUTION_MODE={config_mode}, CLI --mode={cli_mode!r}."
        )

    runtime_settings = settings.model_copy(update={"execution_mode": effective_mode})
    return runtime_settings, effective_mode, effective_dry_run


def _first_market_string(raw_market: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = raw_market.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _market_search_text(raw_market: dict[str, Any]) -> str:
    """Build parser search text from known title/subtitle/rules fields."""
    title = _first_market_string(raw_market, ("title", "market_title", "name"))
    subtitle = _first_market_string(raw_market, ("subtitle", "market_subtitle", "event_subtitle"))
    rules_primary = _first_market_string(raw_market, ("rules_primary", "rules", "rulesPrimary"))
    rules_secondary = _first_market_string(raw_market, ("rules_secondary", "rulesSecondary"))
    return " ".join(part for part in (title, subtitle, rules_primary, rules_secondary) if part)


def _normalized_weather_age_seconds(
    *,
    now: datetime,
    retrieval_timestamp: datetime,
    logger: Any,
) -> float:
    """Compute weather snapshot age with a defensive clamp for small clock skew."""
    age_seconds = (now - retrieval_timestamp).total_seconds()
    if age_seconds < 0:
        logger.warning(
            "Weather snapshot retrieval timestamp is in the future; clamping age to 0. "
            "age_seconds=%.3f retrieval_timestamp=%s now=%s",
            age_seconds,
            retrieval_timestamp.isoformat(),
            now.isoformat(),
        )
        return 0.0
    return age_seconds


def _weather_filter_rejection_reasons(
    *,
    raw_market: dict[str, Any],
    parsed: ParsedWeatherContract,
) -> list[str]:
    """Classify why a market was excluded from weather-filtered candidates."""
    reasons: list[str] = []

    category = _first_market_string(raw_market, ("category", "series", "event_category", "group"))
    tags = raw_market.get("tags")
    weather_tag_present = isinstance(tags, list) and any(
        isinstance(tag, str) and tag.strip().lower() == "weather"
        for tag in tags
    )
    if not category and not weather_tag_present:
        reasons.append("missing_category_field")
    if category and "weather" not in category.lower() and not weather_tag_present:
        reasons.append("category_mismatch")

    title = _first_market_string(raw_market, ("title", "market_title", "name"))
    ticker = _first_market_string(raw_market, ("ticker", "symbol"))
    if not title or not ticker:
        reasons.append("missing_title_ticker")

    status = _first_market_string(raw_market, ("status", "market_status")) or ""
    if status.lower() in {"closed", "inactive", "settled", "expired", "finalized"}:
        reasons.append("closed_inactive")

    event_type = _first_market_string(raw_market, ("event_type", "type", "market_type"))
    if event_type and event_type.lower() not in {"binary", "event", "yes_no", "yes-no"}:
        reasons.append("unsupported_event_type")

    if parsed.weather_candidate and parsed.parse_status != "parsed":
        reasons.append("parse_failure")

    if not parsed.weather_candidate and not reasons:
        reasons.append("unsupported_event_type")

    # We only report reasons for excluded contracts.
    if parsed.weather_candidate and parsed.parse_status == "parsed":
        return []
    return sorted(set(reasons))


def _weather_near_miss_sample(
    *,
    raw_market: dict[str, Any],
    parsed: ParsedWeatherContract,
    rejection_reasons: list[str],
) -> dict[str, Any]:
    tags = raw_market.get("tags")
    normalized_tags: list[str] = []
    if isinstance(tags, list):
        normalized_tags = [str(item) for item in tags[:5]]
    title = _first_market_string(raw_market, ("title", "market_title", "name")) or ""
    return {
        "ticker": _first_market_string(raw_market, ("ticker", "symbol", "market_id", "id")),
        "title": title[:120],
        "category": _first_market_string(
            raw_market,
            ("category", "series", "event_category", "group"),
        ),
        "tags": normalized_tags,
        "status": _first_market_string(raw_market, ("status", "market_status")),
        "parse_status": parsed.parse_status,
        "rejection_reasons": rejection_reasons,
    }


def _evaluate_signal_candidates(
    *,
    args: argparse.Namespace,
    settings: Settings,
    logger: Any,
    journal: JournalWriter,
) -> tuple[list[MicroTradeCandidate], list[SignalRejection], dict[str, int]]:
    scan_limit = args.max_markets_to_scan or settings.signal_max_markets_to_scan
    parser = KalshiWeatherContractParser(logger=logger)

    def _weather_prefilter(record: dict[str, Any]) -> bool:
        return parser._is_weather_candidate(_market_search_text(record), record)

    market_records, market_fetch = _resolve_market_records_with_diagnostics(
        args=args,
        settings=settings,
        logger=logger,
        journal=journal,
        weather_candidate_target=scan_limit,
        weather_candidate_predicate=_weather_prefilter,
    )
    weather_snapshot = _resolve_weather_snapshot(
        args=args,
        settings=settings,
        logger=logger,
        journal=journal,
    )
    # Compute "now" after weather retrieval to avoid negative age under normal latency.
    now = datetime.now(UTC)
    weather_ref = weather_snapshot.raw_payload_path or weather_snapshot.source_url
    weather_age_seconds = _normalized_weather_age_seconds(
        now=now,
        retrieval_timestamp=weather_snapshot.retrieval_timestamp,
        logger=logger,
    )

    matcher = WeatherMarketMatcher(logger=logger)
    estimator = WeatherProbabilityEstimator(logger=logger)
    edge_calculator = EdgeCalculator(settings=settings, logger=logger)
    selector = CandidateSelector(settings=settings, logger=logger)

    evaluations: list[SignalEvaluation] = []
    eval_by_market_id: dict[str, SignalEvaluation] = {}
    weather_rejection_reasons: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    near_miss_samples: list[dict[str, Any]] = []
    weather_candidates_scanned = 0

    for raw_market in market_records:
        parsed = parser.parse_market(raw_market)
        market_id = parsed.provider_market_id
        if parsed.weather_candidate:
            weather_candidates_scanned += 1
        match_result = matcher.match_contract(parsed, weather_snapshot)
        estimate_result = estimator.estimate(parsed, match_result)
        edge_result = edge_calculator.compute(raw_market, estimate_result)
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
        parse_status_counts[parsed.parse_status] += 1
        rejection_reasons = _weather_filter_rejection_reasons(
            raw_market=raw_market,
            parsed=parsed,
        )
        if rejection_reasons:
            weather_rejection_reasons.update(rejection_reasons)
            if len(near_miss_samples) < _MAX_WEATHER_DIAGNOSTIC_SAMPLES:
                near_miss_samples.append(
                    _weather_near_miss_sample(
                        raw_market=raw_market,
                        parsed=parsed,
                        rejection_reasons=rejection_reasons,
                    )
                )
        if weather_candidates_scanned >= scan_limit:
            break

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
        "total_markets_fetched": int(
            market_fetch.get("total_markets_fetched", len(market_records))
        ),
        "pages_fetched": int(market_fetch.get("pages_fetched", 1)),
        "weather_markets_after_filter": sum(
            1 for item in evaluations if item.parsed_contract.weather_candidate
        ),
        "candidates_generated": len(candidates),
        "had_more_pages": int(bool(market_fetch.get("had_more_pages", False))),
        "deduped_count": int(market_fetch.get("deduped_count", 0)),
        "prefilter_weather_matches_found": int(market_fetch.get("candidate_matches_found", 0)),
        "prefilter_stopped_on_target": int(
            bool(market_fetch.get("stopped_on_candidate_target", False))
        ),
    }
    weather_candidates = counts["weather_candidates"]
    excluded_count = max(0, len(evaluations) - weather_candidates)
    reason_counts = dict(weather_rejection_reasons.most_common())
    logger.info(
        "Weather filter diagnostics: scanned=%d weather_markets_after_filter=%d "
        "excluded=%d reasons=%s",
        len(evaluations),
        weather_candidates,
        excluded_count,
        reason_counts,
    )
    if near_miss_samples:
        logger.info(
            "Weather filter near-miss samples (max=%d): %s",
            _MAX_WEATHER_DIAGNOSTIC_SAMPLES,
            near_miss_samples,
        )
    journal.write_event(
        "micro_weather_filter_diagnostics",
        payload={
            "scanned": len(evaluations),
            "weather_markets_after_filter": weather_candidates,
            "excluded": excluded_count,
            "reason_counts": reason_counts,
            "parse_status_counts": dict(parse_status_counts),
            "near_miss_samples": near_miss_samples,
        },
    )
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


def _initialize_dashboard(
    *,
    ui_mode: str,
    console: Console,
    logger: Any,
    session_id: str,
) -> TerminalDashboard | None:
    """Best-effort rich dashboard setup with safe fallback to plain mode."""
    if ui_mode != "rich":
        return None
    try:
        dashboard = TerminalDashboard(console=console, session_id=session_id)
        dashboard.attach_logger(logger)
        return dashboard
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to initialize rich dashboard; falling back to plain output: %s",
            sanitize_text(str(exc)),
        )
        return None


def _safe_dashboard_action(
    dashboard: TerminalDashboard | None,
    *,
    logger: Any,
    action_name: str,
    action: Callable[[TerminalDashboard], None],
) -> TerminalDashboard | None:
    """Guard dashboard operations so UI failures never interrupt execution path."""
    if dashboard is None:
        return None
    try:
        action(dashboard)
        return dashboard
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Dashboard failure during %s; switching to plain output: %s",
            action_name,
            sanitize_text(str(exc)),
        )
        with suppress(Exception):
            dashboard.detach_logger()
        return None


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
    dashboard: TerminalDashboard | None = None

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

    config_mode = settings.execution_mode
    cli_mode = args.mode
    try:
        settings, mode, effective_dry_run = _resolve_runtime_mode(
            settings=settings,
            args=args,
            logger=logger,
        )
    except ValueError as exc:
        logger.error("Day 7 startup mode validation failed: %s", sanitize_text(str(exc)))
        return 2
    if mode == "live_micro":
        missing_flags: list[str] = []
        if not settings.allow_live_api:
            missing_flags.append("ALLOW_LIVE_API=true")
        if not settings.allow_live_fills:
            missing_flags.append("ALLOW_LIVE_FILLS=true")
        if not settings.micro_mode_enabled:
            missing_flags.append("MICRO_MODE_ENABLED=true")
        if missing_flags:
            logger.error(
                "Unsafe live_micro startup: missing required guard flags for this run (%s). "
                "Set these in .env or pass a safe mode (--dry-run).",
                ", ".join(missing_flags),
            )
            return 2
        if settings.micro_require_supervised_mode and not args.supervised:
            logger.error(
                "Unsafe startup: --supervised is required by MICRO_REQUIRE_SUPERVISED_MODE."
            )
            return 2

    dashboard = _initialize_dashboard(
        ui_mode=args.ui_mode,
        console=console,
        logger=logger,
        session_id=session_id,
    )
    dashboard = _safe_dashboard_action(
        dashboard,
        logger=logger,
        action_name="startup render",
        action=lambda d: d.render_startup(
            mode=mode,
            effective_dry_run=effective_dry_run,
            supervised=args.supervised,
            config_mode=config_mode,
            cli_mode=cli_mode,
        ),
    )
    logger.info(
        "Day 7 runtime state: config_mode=%s cli_mode=%s "
        "effective_mode=%s dry_run=%s supervised=%s",
        config_mode,
        cli_mode,
        mode,
        effective_dry_run,
        args.supervised,
    )

    max_trades_this_run = (
        args.max_trades_this_run
        if args.max_trades_this_run is not None
        else settings.micro_max_trades_per_run
    )

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
                "config_execution_mode": config_mode,
                "cli_mode": cli_mode,
                "effective_execution_mode": mode,
                "effective_dry_run": effective_dry_run,
                "ui_mode": args.ui_mode,
                "max_cycles": args.max_cycles,
                "max_trades_this_run": max_trades_this_run,
                "supervised": args.supervised,
                "market_scope": scope,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize day7 journal: %s", sanitize_text(str(exc)))
        if dashboard is not None:
            error_text = sanitize_text(str(exc))
            dashboard = _safe_dashboard_action(
                dashboard,
                logger=logger,
                action_name="journal init error event",
                action=lambda d, message=error_text: d.record_event(
                    severity="ERROR",
                    message=f"Journal initialization failed: {message}",
                ),
            )
            dashboard = _safe_dashboard_action(
                dashboard,
                logger=logger,
                action_name="journal init summary render",
                action=lambda d: d.render_end(
                    mode=mode,
                    effective_dry_run=effective_dry_run,
                    supervised=args.supervised,
                    total_counts=Counter(),
                    final_summary={
                        "open_positions_count": 0,
                        "realized_pnl_cents": 0,
                        "daily_gross_exposure_cents": 0,
                        "gross_exposure_utilization": 0.0,
                        "realized_loss_utilization": 0.0,
                        "halted_early": True,
                        "halt_reason": "journal_init_failed",
                    },
                    exit_code=3,
                ),
            )
            if dashboard is not None:
                dashboard.detach_logger()
        return 3

    exit_code = 0
    client: KalshiClient | None = None
    micro_runner: LiveMicroRunner | None = None
    attempts_all: list[MicroOrderAttempt] = []
    total_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    final_summary: dict[str, Any] = {}
    cycle_data_failures = 0

    try:
        if mode == "live_micro":
            micro_runner, client = _build_live_micro_runner(
                settings=settings,
                logger=logger,
                journal=journal,
            )

        for _cycle_index in range(1, args.max_cycles + 1):
            try:
                candidates, rejections, counts = _evaluate_signal_candidates(
                    args=args,
                    settings=settings,
                    logger=logger,
                    journal=journal,
                )
            except (KalshiAPIError, WeatherProviderError) as exc:
                cycle_data_failures += 1
                logger.error(
                    "Day 7 cycle %d/%d data fetch failed; continuing to next cycle: %s",
                    _cycle_index,
                    args.max_cycles,
                    sanitize_text(str(exc)),
                )
                journal.write_event(
                    "micro_cycle_data_failure",
                    payload={
                        "cycle_index": _cycle_index,
                        "error": sanitize_text(str(exc)),
                        "error_type": type(exc).__name__,
                        "continue_next_cycle": _cycle_index < args.max_cycles,
                    },
                    metadata={"session_id": session_id},
                )
                if dashboard is not None:
                    err_type = type(exc).__name__
                    error_msg = sanitize_text(str(exc))
                    def _record_cycle_failure(
                        d: TerminalDashboard,
                        idx: int = _cycle_index,
                        msg: str = error_msg,
                        err: str = err_type,
                    ) -> None:
                        d.record_event(
                            severity="ERROR",
                            message=f"Cycle {idx} data fetch failure: {msg}",
                            dedupe_key=f"cycle_data_failure:{err}:{msg}",
                        )
                    dashboard = _safe_dashboard_action(
                        dashboard,
                        logger=logger,
                        action_name="cycle data failure event",
                        action=_record_cycle_failure,
                    )
                continue
            total_counts.update({
                "cycles_processed": 1,
                "candidates_seen": len(candidates),
                "signal_scanned": counts["scanned"],
                "signal_filtered": counts["filtered"],
                "total_markets_fetched": counts.get("total_markets_fetched", counts["scanned"]),
                "weather_markets_after_filter": counts.get(
                    "weather_markets_after_filter",
                    counts.get("weather_candidates", 0),
                ),
                "pages_fetched": counts.get("pages_fetched", 1),
                "candidates_generated": counts.get("candidates_generated", len(candidates)),
            })
            if counts.get("had_more_pages", 0):
                total_counts.update({"pagination_cursor_seen_cycles": 1})
            if dashboard is not None and counts.get("had_more_pages", 0):
                dashboard = _safe_dashboard_action(
                    dashboard,
                    logger=logger,
                    action_name="pagination cursor warning",
                    action=lambda d: d.record_event(
                        severity="WARN",
                        message=(
                            "Pagination cursor still present after fetch caps; "
                            "additional pages remain."
                        ),
                        dedupe_key="warn:pagination_cursor_after_caps",
                    ),
                )
            if dashboard is not None and counts.get("deduped_count", 0) > 0:
                deduped_count = counts.get("deduped_count", 0)
                dashboard = _safe_dashboard_action(
                    dashboard,
                    logger=logger,
                    action_name="dedupe info event",
                    action=lambda d, deduped=deduped_count: d.record_event(
                        severity="INFO",
                        message=f"Market dedupe removed {deduped} duplicates.",
                    ),
                )
            logger.info(
                "Day 7 cycle diagnostics: pages_fetched=%d total_markets_fetched=%d "
                "weather_markets_after_filter=%d candidates_generated=%d "
                "prefilter_weather_matches_found=%d prefilter_stopped_on_target=%s",
                counts.get("pages_fetched", 1),
                counts.get("total_markets_fetched", counts["scanned"]),
                counts.get("weather_markets_after_filter", counts.get("weather_candidates", 0)),
                counts.get("candidates_generated", len(candidates)),
                counts.get("prefilter_weather_matches_found", 0),
                bool(counts.get("prefilter_stopped_on_target", 0)),
            )
            journal.write_event(
                "micro_cycle_signal_summary",
                payload={
                    "cycle_index": _cycle_index,
                    "pages_fetched": counts.get("pages_fetched", 1),
                    "total_markets_fetched": counts.get(
                        "total_markets_fetched",
                        counts["scanned"],
                    ),
                    "weather_markets_after_filter": counts.get(
                        "weather_markets_after_filter",
                        counts.get("weather_candidates", 0),
                    ),
                    "candidates_generated": counts.get("candidates_generated", len(candidates)),
                    "had_more_pages": bool(counts.get("had_more_pages", 0)),
                    "deduped_count": counts.get("deduped_count", 0),
                    "prefilter_weather_matches_found": counts.get(
                        "prefilter_weather_matches_found",
                        0,
                    ),
                    "prefilter_stopped_on_target": bool(
                        counts.get("prefilter_stopped_on_target", 0)
                    ),
                    "signal_scanned": counts["scanned"],
                    "signal_filtered": counts["filtered"],
                    "selected": counts.get("selected", len(candidates)),
                },
                metadata={"session_id": session_id},
            )

            for rejection in rejections[: args.print_rejections]:
                journal.write_event(
                    "signal_candidate_rejected",
                    payload=rejection.model_dump(mode="json"),
                )

            if effective_dry_run:
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
                    "effective_dry_run": True,
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
                if dashboard is not None:
                    cycle_index = _cycle_index
                    cycle_counts = dict(counts)
                    cycle_summary = dict(final_summary)
                    def _render_cycle(
                        d: TerminalDashboard,
                        idx: int = cycle_index,
                        c: dict[str, int] = cycle_counts,
                        s: dict[str, Any] = cycle_summary,
                    ) -> None:
                        d.render_cycle(
                            cycle_index=idx,
                            max_cycles=args.max_cycles,
                            mode=mode,
                            effective_dry_run=effective_dry_run,
                            supervised=args.supervised,
                            counts=c,
                            total_counts=total_counts,
                            final_summary=s,
                            skip_reasons=skip_reasons,
                            attempts=attempts_all,
                        )
                    dashboard = _safe_dashboard_action(
                        dashboard,
                        logger=logger,
                        action_name="cycle render",
                        action=_render_cycle,
                    )
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
            if dashboard is not None:
                cycle_index = _cycle_index
                cycle_counts = dict(counts)
                cycle_summary = dict(final_summary)
                def _render_cycle(
                    d: TerminalDashboard,
                    idx: int = cycle_index,
                    c: dict[str, int] = cycle_counts,
                    s: dict[str, Any] = cycle_summary,
                ) -> None:
                    d.render_cycle(
                        cycle_index=idx,
                        max_cycles=args.max_cycles,
                        mode=mode,
                        effective_dry_run=effective_dry_run,
                        supervised=args.supervised,
                        counts=c,
                        total_counts=total_counts,
                        final_summary=s,
                        skip_reasons=skip_reasons,
                        attempts=attempts_all,
                    )
                dashboard = _safe_dashboard_action(
                    dashboard,
                    logger=logger,
                    action_name="cycle render",
                    action=_render_cycle,
                )

            if summary.halted_early:
                exit_code = 4
                break

        if mode == "live_micro" and total_counts["unresolved"] > 0:
            exit_code = 4
        if cycle_data_failures > 0 and total_counts["cycles_processed"] == 0 and exit_code == 0:
            exit_code = 4

        if dashboard is None:
            summary_line = (
                f"mode={mode} cycles={total_counts['cycles_processed']} "
                f"cycle_data_failures={cycle_data_failures} "
                f"pages_fetched={total_counts['pages_fetched']} "
                f"total_markets_fetched={total_counts['total_markets_fetched']} "
                f"weather_markets_after_filter={total_counts['weather_markets_after_filter']} "
                f"signal_scanned={total_counts['signal_scanned']} "
                f"candidates_generated={total_counts['candidates_generated']} "
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
                    "daily_gross_exposure_cents="
                    f"{final_summary.get('daily_gross_exposure_cents', 0)} "
                    f"gross_util={final_summary.get('gross_exposure_utilization', 0.0)} "
                    f"loss_util={final_summary.get('realized_loss_utilization', 0.0)} "
                    f"halted_early={final_summary.get('halted_early', False)} "
                    f"halt_reason={final_summary.get('halt_reason') or '-'}"
                )

        journal.write_event(
            "micro_session_summary",
            payload={
                "mode": mode,
                "effective_dry_run": effective_dry_run,
                "cycles_processed": total_counts["cycles_processed"],
                "cycle_data_failures": cycle_data_failures,
                "pages_fetched": total_counts["pages_fetched"],
                "total_markets_fetched": total_counts["total_markets_fetched"],
                "weather_markets_after_filter": total_counts["weather_markets_after_filter"],
                "signal_scanned": total_counts["signal_scanned"],
                "candidates_generated": total_counts["candidates_generated"],
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
        error_text = sanitize_text(str(exc))
        error_type = type(exc).__name__
        logger.error("Day 7 session failed: %s", error_text)
        if dashboard is not None:
            dashboard = _safe_dashboard_action(
                dashboard,
                logger=logger,
                action_name="failure event",
                action=lambda d, message=error_text, err_type=error_type: d.record_event(
                    severity="ERROR",
                    message=f"Session failure: {message}",
                    dedupe_key=f"error:{err_type}:{message}",
                ),
            )
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
        error_text = sanitize_text(str(exc))
        logger.exception("Unexpected Day 7 failure: %s", error_text)
        if dashboard is not None:
            dashboard = _safe_dashboard_action(
                dashboard,
                logger=logger,
                action_name="unhandled failure event",
                action=lambda d, message=error_text: d.record_event(
                    severity="CRITICAL",
                    message=f"Unhandled failure: {message}",
                ),
            )
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
        if dashboard is not None:
            if not final_summary:
                final_summary = {
                    "open_positions_count": 0,
                    "realized_pnl_cents": 0,
                    "daily_gross_exposure_cents": 0,
                    "gross_exposure_utilization": 0.0,
                    "realized_loss_utilization": 0.0,
                    "halted_early": exit_code != 0,
                    "halt_reason": None,
                }
            dashboard = _safe_dashboard_action(
                dashboard,
                logger=logger,
                action_name="final summary render",
                action=lambda d: d.render_end(
                    mode=mode,
                    effective_dry_run=effective_dry_run,
                    supervised=args.supervised,
                    total_counts=total_counts,
                    final_summary=final_summary,
                    exit_code=exit_code,
                ),
            )
            if dashboard is not None:
                dashboard.detach_logger()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

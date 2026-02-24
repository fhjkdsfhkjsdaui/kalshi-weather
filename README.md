# Kalshi Weather Bot Foundation (Day 1 + Day 2 + Day 3 + Day 4 + Day 5 + Day 6 + Day 7)

This repository is a practical foundation for a Kalshi weather trading research system.

- Day 1: Kalshi read-only market discovery + journaling
- Day 2: NWS forecast ingestion + normalized weather snapshot journaling
- Day 3: Kalshi weather contract parser + mapping audit CLI
- Day 4: risk manager + dry-run execution skeleton (no live orders)
- Day 5: explainable signal evaluation + candidate selection into dry-run execution
- Day 6: live cancel-only order lifecycle validation + reconciliation (operator-controlled)
- Day 7: supervised `live_micro` mode with hard caps, policy gate, and fill-confirmed position tracking

Out of scope:
- strategy/signal logic
- general fill-seeking live trading
- replay/backtesting/dashboard/database systems

## Project layout (relevant parts)

```text
.
├── .env.example
├── pyproject.toml
├── src/kalshi_weather_bot
│   ├── cli.py
│   ├── config.py
│   ├── contracts_cli.py
│   ├── day4_cli.py
│   ├── day5_cli.py
│   ├── day6_cli.py
│   ├── day7_cli.py
│   ├── exceptions.py
│   ├── journal.py
│   ├── kalshi_client.py
│   ├── redaction.py
│   ├── weather_cli.py
│   ├── contracts
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── parser.py
│   ├── execution
│   │   ├── __init__.py
│   │   ├── cancel_only.py
│   │   ├── dry_run.py
│   │   ├── lifecycle.py
│   │   ├── live_adapter.py
│   │   ├── live_micro.py
│   │   ├── live_models.py
│   │   ├── micro_models.py
│   │   ├── micro_policy.py
│   │   ├── models.py
│   │   ├── position_tracker.py
│   │   └── reconciliation.py
│   └── risk
│       ├── __init__.py
│       ├── manager.py
│       └── models.py
│   └── signal
│       ├── __init__.py
│       ├── edge.py
│       ├── estimator.py
│       ├── matcher.py
│       ├── models.py
│       └── selector.py
│   └── weather
│       ├── __init__.py
│       ├── base.py
│       ├── models.py
│       └── nws.py
└── tests
    ├── test_day4_dry_run_executor.py
    ├── test_day4_orchestrator_cli.py
    ├── test_day4_risk_manager.py
    ├── test_day5_cli.py
    ├── test_day5_edge.py
    ├── test_day5_estimator.py
    ├── test_day5_matcher.py
    ├── test_day5_selector.py
    ├── test_day6_cancel_only.py
    ├── test_day7_live_micro.py
    ├── test_contract_parser_day3.py
    ├── test_day1_assumptions.py
    └── test_nws_provider_day2.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env`:
- set Kalshi credentials for Day 1
- set `NWS_USER_AGENT` for Day 2
- optionally set `WEATHER_DEFAULT_LAT` and `WEATHER_DEFAULT_LON`

## Day 1 command (existing)

```bash
kalshi-weather-day1
```

## Day 2 command (new)

### Option A: points lookup by coordinates

```bash
kalshi-weather-day2 --lat 40.7128 --lon -74.0060 --forecast-type hourly
```

### Option B: direct NWS forecast URL

```bash
kalshi-weather-day2 --url https://api.weather.gov/gridpoints/OKX/33,35/forecast
```

Optional:

```bash
kalshi-weather-day2 --lat 40.7128 --lon -74.0060 --forecast-type daily --max-print 6
```

## Day 3 command (contract parser audit)

### Option A: audit an existing raw markets snapshot file

```bash
kalshi-weather-day3 --input-file data/raw/20260224T050124Z_f1b56ddef75f_markets.json --max-print 3
```

### Option B: fetch first page from Kalshi API, then audit

```bash
kalshi-weather-day3 --limit 200 --only-weather --min-confidence 0.6
```

Day 3 summary output includes:
- total markets scanned
- weather candidates
- parsed/ambiguous/unsupported/rejected counts
- top rejection reasons
- examples from each bucket

Day 3 journals:
- `contract_parse_start`
- `contract_parse_summary`
- `contract_parse_rejection_sample` (capped)
- `contract_parse_shutdown`

Supported Day 3 weather parsing patterns (initial foundation):
- temperature (`high temp`, `low temp`, `temperature`, `degrees`)
- precipitation/rainfall
- snowfall
- wind speed/gust
- threshold operators (`above`, `below`, `at least`, `at most`, symbols, `between`, simple ranges)
- basic location extraction (`in City, ST`, `for City, ST`)

Known Day 3 limitations:
- title formats outside these patterns are intentionally rejected/ambiguous
- location normalization is heuristic and currently US city/state-focused
- contract time windows rely mostly on API timestamps (`close_time`, `expiration_time`, etc.)
- Kalshi-specific edge schema variants should be added incrementally as snapshots are collected

## Day 4 command (risk + dry-run execution)

Default offline plumbing demo:

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day4_cli
```

Run with fixture intents:

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day4_cli --input-json tests/fixtures/day4_intents.json
```

Optional:

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day4_cli --cancel-first-accepted --max-print 5
```

Day 4 behavior:
- evaluates each `TradeIntent` through explicit risk checks
- rejects unsafe intents with structured reason codes
- accepts safe intents into in-memory dry-run open orders only
- never calls live Kalshi order placement endpoints

Day 4 journals:
- `risk_evaluation`
- `risk_rejection`
- `dry_run_order_submitted`
- `dry_run_order_cancelled`
- `dry_run_execution_summary`

Key Day 4 risk env vars:
- `RISK_MIN_PARSER_CONFIDENCE`
- `RISK_MAX_WEATHER_AGE_SECONDS`
- `RISK_MAX_STAKE_PER_TRADE_CENTS`
- `RISK_MAX_TOTAL_EXPOSURE_CENTS`
- `RISK_MAX_EXPOSURE_PER_MARKET_CENTS`
- `RISK_MAX_CONCURRENT_OPEN_ORDERS`
- `RISK_DUPLICATE_INTENT_COOLDOWN_SECONDS`
- `DRY_RUN_MODE` (must remain `true` in Day 4)

## Day 5 command (signal loop -> dry-run only)

Offline deterministic fixture mode:

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day5_cli \
  --input-markets-file tests/fixtures/day5_markets.json \
  --input-weather-file tests/fixtures/day5_weather_snapshot.json \
  --max-candidates 3 \
  --print-rejections 5
```

Optional read-only live scan mode (no order placement, still dry-run only):

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day5_cli \
  --max-markets-to-scan 200 \
  --lat 40.7128 --lon -74.0060 \
  --forecast-type hourly
```

Day 5 output summary includes:
- `scanned`
- `weather_candidates`
- `matched`
- `estimated`
- `filtered`
- `risk_rejected`
- `dry_run_submitted`

Day 5 journals:
- `signal_scan_start`
- `signal_match_result`
- `signal_estimate_result`
- `signal_edge_result`
- `signal_candidate_rejected`
- `signal_candidate_selected`
- `signal_batch_summary`
- `signal_scan_shutdown`

Day 5 v1 supported contract/estimate types:
- temperature threshold contracts (deterministic period-threshold hit rate)
- precipitation occurrence/probability proxy contracts (PoP-based, conservative)
- wind threshold contracts (parsed numeric wind speed strings)

Day 5 assumptions/limitations:
- timezone semantics are conservative; matcher logs assumptions when contract window semantics are unclear
- precipitation accumulation units (`in`, `mm`, `cm`) are rejected as unsupported in v1
- missing/ambiguous market pricing fields fail safe (no candidate)
- candidate ranking is transparent score-based (`edge_after_buffers` weighted by matcher/model confidence)
- no live trading path is implemented; accepted candidates only flow into Day 4 dry-run executor

## Day 6 command (live cancel-only validation)

Day 6 is not fill-seeking trading. It is an operator-run submit/cancel/reconcile validation path with strict guards.

Example single attempt:

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day6_cli \
  --mode live_cancel_only \
  --market KXWEATHER-NYC-TEST \
  --side yes \
  --price-cents 1 \
  --qty 1 \
  --attempts 1 \
  --poll-timeout-seconds 15 \
  --poll-interval-seconds 0.5 \
  --cancel-delay-ms 100
```

Required guard env vars for Day 6 live API use:
- `EXECUTION_MODE=live_cancel_only`
- `ALLOW_LIVE_API=true`
- `CANCEL_ONLY_ENABLED=true`

Important Day 6 safety behavior:
- any unexpected `filled` or `partially_filled` status is elevated to critical event
- when `CANCEL_ONLY_HALT_ON_ANY_FILL=true`, the run halts immediately after first fill detection
- unresolved reconciliation is not treated as success

Day 6 success criteria for a clean attempt:
- submit acknowledged
- cancel acknowledged
- reconciliation matches local terminal state as canceled

Day 6 failure indicators:
- submit rejected
- cancel timeout/unresolved terminal state
- reconciliation mismatch
- unexpected partial/full fill

Day 6 journals include:
- `order_lifecycle_start`
- `order_submit_requested`
- `order_submit_ack` / `order_submit_rejected`
- `order_cancel_requested`
- `order_cancel_ack`
- `order_status_polled`
- `order_reconciliation_result`
- `order_unexpected_fill_detected`
- `cancel_only_batch_summary`
- `cancel_only_halt` (when halted early)

## Day 7 command (supervised micro-live)

Day 7 is intentional tiny-fill collection with strict operator gating, not autonomous trading.

Safe first-run pattern (single supervised trade max):

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day7_cli \
  --mode live_micro \
  --supervised \
  --max-cycles 1 \
  --max-trades-this-run 1 \
  --max-candidates 1 \
  --print-rejections 5
```

Offline deterministic smoke path:

```bash
PYTHONPATH=src python -m kalshi_weather_bot.day7_cli \
  --mode dry_run \
  --input-markets-file tests/fixtures/day5_markets.json \
  --input-weather-file tests/fixtures/day5_weather_snapshot.json \
  --max-cycles 1 \
  --max-trades-this-run 1
```

Required guard env vars for Day 7 live mode:
- `EXECUTION_MODE=live_micro`
- `ALLOW_LIVE_API=true`
- `ALLOW_LIVE_FILLS=true`
- `MICRO_MODE_ENABLED=true`
- `MICRO_REQUIRE_SUPERVISED_MODE=true` (recommended)

Default Day 7 caps (conservative):
- `MICRO_MAX_NOTIONAL_PER_TRADE_DOLLARS=0.50`
- `MICRO_MAX_TRADES_PER_RUN=1`
- `MICRO_MAX_TRADES_PER_DAY=3`
- `MICRO_MAX_OPEN_POSITIONS=1`
- `MICRO_MAX_DAILY_GROSS_EXPOSURE_DOLLARS=5.00`
- `MICRO_MAX_DAILY_REALIZED_LOSS_DOLLARS=2.00`
- `MICRO_MIN_SECONDS_BETWEEN_TRADES=60`

Day 7 halts when configured:
- reconciliation mismatch / unresolved terminal state
- unexpected fill-state inconsistency
- supervised-mode guard not satisfied

Day 7 journals include:
- `trade_policy_evaluated`
- `trade_policy_denied`
- `trade_policy_approved`
- `micro_order_submitted`
- `micro_order_fill_detected`
- `micro_order_partial_fill_detected`
- `micro_order_rejected`
- `micro_order_unresolved`
- `position_opened`
- `position_updated`
- `position_closed`
- `risk_cap_hit`
- `micro_session_summary`
- `micro_halt`

Before increasing trade count above 1, review:
1. Last session `micro_session_summary` and any `micro_halt` events
2. Reconciliation outcomes and unresolved count
3. Position state and realized PnL assumptions
4. Actual exchange response fields for create/status semantics (`TODO(KALSHI_API)` markers)

## Day 2 success output

A successful run prints:
- provider and forecast type
- location summary
- number of periods
- first `N` normalized periods (start, temp, wind, precip, short forecast)

It journals:
- lifecycle events (`weather_startup`, `weather_request_start`, `weather_request_success`, `weather_shutdown`)
- normalized snapshot event (`weather_snapshot_normalized`)
- raw points/forecast snapshots (`weather_raw_snapshot`) when enabled

Default paths:
- events JSONL: `JOURNAL_DIR` (default `./data/journal`)
- raw weather payloads: `WEATHER_RAW_PAYLOAD_DIR` (default `./data/raw/weather`)

## Common Day 2 failure modes

1. Invalid input mode
- Symptom: CLI error about `--url` and `--lat/--lon`
- Fix: provide only one mode

2. Invalid coordinates
- Symptom: latitude/longitude range error
- Fix: use lat `[-90, 90]`, lon `[-180, 180]`

3. NWS transient/network outage
- Symptom: request failure with NWS context in error message
- Fix: retry later, confirm connectivity

4. NWS payload shape drift
- Symptom: missing `properties`/`periods` style errors
- Fix: inspect raw snapshot files and update normalization assumptions in `weather/nws.py`

## NWS etiquette notes

- NWS generally does not require an API key.
- Use a descriptive `User-Agent` (`NWS_USER_AGENT`) with contact info.
- Keep request frequency reasonable.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Day 2 tests focus on:
- points endpoint parsing
- forecast normalization
- malformed payload handling
- datetime parsing
- journaling compatibility of normalized weather snapshots
- Day 3 parser normalization, confidence boundaries, and audit CLI smoke behavior
- Day 4 risk rejection/approval rules, duplicate suppression, dry-run submit/cancel/list behavior
- Day 5 matcher/estimator/edge/selector deterministic behavior and Day 5 offline CLI smoke flow
- Day 6 cancel-only lifecycle transitions, reconciliation, startup guards, and mocked CLI smoke behavior

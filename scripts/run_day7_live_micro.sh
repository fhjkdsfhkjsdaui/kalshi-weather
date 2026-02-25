#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${SERIES_TICKER:=KXHIGHNY}"
: "${MAX_CYCLES:=1}"
: "${MAX_TRADES_THIS_RUN:=1}"
: "${SIGNAL_MAX_MARKETS_TO_SCAN:=200}"
: "${KALSHI_MAX_MARKETS_FETCH:=500}"
: "${KALSHI_MAX_PAGES_PER_FETCH:=2}"
: "${MICRO_POLL_TIMEOUT_SECONDS:=45}"
: "${UI_MODE:=plain}"
: "${USE_CAFFEINATE:=1}"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable ${PYTHON_BIN}. Activate your venv first." >&2
  exit 1
fi

export EXECUTION_MODE=live_micro
export ALLOW_LIVE_API=true
export ALLOW_LIVE_FILLS=true
export MICRO_MODE_ENABLED=true
export KALSHI_MARKETS_ENDPOINT="/trade-api/v2/markets?series_ticker=${SERIES_TICKER}&status=open"
export KALSHI_ORDER_STATUS_ENDPOINT_TEMPLATE="/trade-api/v2/portfolio/orders/{order_id}/status"
export KALSHI_ORDER_CANCEL_ENDPOINT_TEMPLATE="/trade-api/v2/portfolio/orders/{order_id}/cancel"
export SIGNAL_MAX_MARKETS_TO_SCAN
export KALSHI_MAX_MARKETS_FETCH
export KALSHI_MAX_PAGES_PER_FETCH
export MICRO_POLL_TIMEOUT_SECONDS

CMD=(
  "${PYTHON_BIN}" -m kalshi_weather_bot.day7_cli
  --mode live_micro
  --no-dry-run
  --supervised
  --max-cycles "${MAX_CYCLES}"
  --max-trades-this-run "${MAX_TRADES_THIS_RUN}"
  --ui-mode "${UI_MODE}"
)

echo "Running Day 7 live_micro:"
echo "  SERIES_TICKER=${SERIES_TICKER} MAX_CYCLES=${MAX_CYCLES} MAX_TRADES_THIS_RUN=${MAX_TRADES_THIS_RUN}"
echo "  SIGNAL_MAX_MARKETS_TO_SCAN=${SIGNAL_MAX_MARKETS_TO_SCAN} KALSHI_MAX_MARKETS_FETCH=${KALSHI_MAX_MARKETS_FETCH}"

if [[ "${USE_CAFFEINATE}" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
  PYTHONPATH=src caffeinate -dims "${CMD[@]}"
else
  PYTHONPATH=src "${CMD[@]}"
fi

JDATE="$(date -u +%Y%m%d)"
JOURNAL_PATH="data/journal/${JDATE}.jsonl"
if [[ ! -f "${JOURNAL_PATH}" ]]; then
  echo "Journal not found at ${JOURNAL_PATH}" >&2
  exit 0
fi

SESSION_ID="$(grep '"event_type": "micro_session_start"' "${JOURNAL_PATH}" | tail -1 | sed -E 's/.*"session_id": "([^"]+)".*/\1/')"
if [[ -z "${SESSION_ID}" ]]; then
  echo "No micro_session_start found in ${JOURNAL_PATH}" >&2
  exit 0
fi

echo
echo "Latest session: ${SESSION_ID}"
grep "\"session_id\": \"${SESSION_ID}\"" "${JOURNAL_PATH}" \
  | grep -E 'micro_session_summary|micro_order_fill_detected|micro_order_unresolved|micro_halt' \
  || true

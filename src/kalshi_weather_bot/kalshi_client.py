"""Kalshi API adapter for Day 1 read-only market discovery."""

from __future__ import annotations

import base64
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

import httpx

from .config import Settings
from .exceptions import KalshiAPIError, KalshiRequestError
from .models import MarketSummary
from .redaction import sanitize_text

# RSA-PSS signing is only needed for key_secret auth mode.
# Import lazily to keep startup fast when using bearer mode.
try:
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False


def _sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    """Sign text using RSA-PSS with SHA256, per Kalshi API spec.

    Returns the base64-encoded signature string.
    """
    message = text.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _load_rsa_private_key(pem_text: str) -> rsa.RSAPrivateKey:
    """Parse a PEM-encoded RSA private key string into an RSAPrivateKey object."""
    if not _HAS_CRYPTOGRAPHY:
        raise KalshiAPIError(
            "The 'cryptography' package is required for key_secret auth mode. "
            "Install it with: pip install cryptography"
        )
    try:
        key = serialization.load_pem_private_key(pem_text.encode("utf-8"), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise KalshiAPIError(
            f"Failed to load RSA private key from PEM data: {exc}"
        ) from exc
    if not isinstance(key, rsa.RSAPrivateKey):
        raise KalshiAPIError(
            f"Expected RSA private key, got {type(key).__name__}."
        )
    return key


class KalshiClient:
    """Thin Kalshi API adapter with normalization and weather filtering."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        max_retries: int = 1,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._client = httpx.Client(
            base_url=str(settings.kalshi_api_base_url),
            timeout=settings.kalshi_timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": "kalshi-weather-bot/day1",
            },
        )
        # Pre-load RSA private key at init for key_secret mode.
        self._rsa_private_key: rsa.RSAPrivateKey | None = None
        if settings.kalshi_auth_mode == "key_secret" and settings.kalshi_api_key_secret:
            self._rsa_private_key = _load_rsa_private_key(settings.kalshi_api_key_secret)

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close underlying HTTP client."""
        self._client.close()

    def validate_connection(self) -> dict[str, Any]:
        """
        Validate auth/connectivity via a small read-only markets request.

        This method intentionally uses the same endpoint as regular market fetching
        so endpoint/auth mismatches are surfaced early.
        """
        payload = self.fetch_markets_raw(limit=1)
        markets = self.extract_market_records(payload)
        if not self._has_market_like_sample(markets):
            payload_keys = list(payload.keys()) if isinstance(payload, dict) else None
            self.logger.error(
                "Kalshi connection validation failed: payload did not contain market-like records.",
                extra={"payload_keys": payload_keys},
            )
            raise KalshiAPIError(
                "Kalshi markets payload shape invalid during auth validation: "
                "expected at least one market-like object with id/ticker and title/status fields."
            )
        return {"ok": True, "sample_market_count": len(markets)}

    def fetch_markets_raw(
        self,
        limit: int | None = None,
        *,
        candidate_predicate: Callable[[dict[str, Any]], bool] | None = None,
        candidate_target: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Fetch raw markets payload from Kalshi, with optional cursor pagination.

        ``extra_params`` are merged into every page request (e.g.
        ``{"series_ticker": "KXHIGHNY", "status": "open"}``).
        """
        effective_limit = limit or self.settings.kalshi_default_limit
        params: dict[str, Any] = {"limit": effective_limit}
        if extra_params:
            params.update(extra_params)
        first_payload = self._request_markets_page_with_retry(
            params=params,
            page_index=1,
            cursor_present=False,
            pages_fetched_so_far=0,
            total_markets_fetched_so_far=0,
        )

        # Preserve legacy behavior for list payloads (no cursor metadata expected).
        if not isinstance(first_payload, dict):
            return first_payload

        records, list_key = self._extract_market_records_with_key(first_payload)
        if records is None:
            return first_payload

        pagination_enabled = self._get_bool_setting("kalshi_enable_pagination", True)
        max_pages = self._get_int_setting("kalshi_max_pages_per_fetch", 10)
        max_markets = self._get_int_setting("kalshi_max_markets_fetch", 2000)
        page_sleep_seconds = (
            self._get_non_negative_int_setting("kalshi_page_sleep_ms", 100) / 1000.0
        )
        effective_candidate_target = (
            candidate_target
            if isinstance(candidate_target, int)
            and candidate_target > 0
            and candidate_predicate is not None
            else None
        )
        # When candidate-target pagination is active, continue scanning pages until the
        # target is met (or page cap hit) rather than stopping early at max_markets.
        effective_max_markets = max_markets
        if effective_candidate_target is not None and max_pages > 0 and effective_limit > 0:
            effective_max_markets = max(max_markets, max_pages * effective_limit)

        deduped_records: list[Any] = []
        seen_keys: set[str] = set()
        seen_cursors: set[str] = set()
        deduped_count = 0
        candidate_matches_found = 0
        page_diagnostics: list[dict[str, Any]] = []
        stop_reason: str | None = None

        def _sample_tickers(page_records: list[Any], *, max_items: int = 3) -> list[str]:
            sample: list[str] = []
            for record in page_records:
                if not isinstance(record, dict):
                    continue
                ticker = (
                    record.get("ticker")
                    or record.get("event_ticker")
                    or record.get("symbol")
                    or record.get("market_id")
                    or record.get("id")
                )
                if isinstance(ticker, str) and ticker.strip():
                    sample.append(ticker.strip())
                if len(sample) >= max_items:
                    break
            return sample

        def _append_records(page_records: list[Any]) -> int:
            nonlocal deduped_count, candidate_matches_found
            page_candidate_matches = 0
            for record in page_records:
                if len(deduped_records) >= effective_max_markets:
                    break
                key = self._market_dedupe_key(record)
                if key is not None:
                    if key in seen_keys:
                        deduped_count += 1
                        continue
                    seen_keys.add(key)
                deduped_records.append(record)
                if effective_candidate_target is not None and isinstance(record, dict):
                    try:
                        if candidate_predicate(record):
                            candidate_matches_found += 1
                            page_candidate_matches += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.debug(
                            "Market candidate predicate failed for one record: %s",
                            type(exc).__name__,
                        )
            return page_candidate_matches

        first_page_matches = _append_records(records)
        page_diagnostics.append(
            {
                "page_index": 1,
                "records_on_page": len(records),
                "sample_tickers": _sample_tickers(records),
                "candidate_matches_page": first_page_matches,
                "candidate_matches_cumulative": candidate_matches_found,
            }
        )
        self.logger.info(
            "Kalshi markets page diagnostics: page_index=%d records=%d "
            "sample_tickers=%s candidate_matches_page=%d candidate_matches_cumulative=%d",
            1,
            len(records),
            _sample_tickers(records),
            first_page_matches,
            candidate_matches_found,
        )

        pages_fetched = 1
        next_cursor = self._extract_pagination_cursor(first_payload)
        if not pagination_enabled and next_cursor:
            stop_reason = "pagination_disabled"
            self.logger.warning(
                "Kalshi pagination disabled for this run; additional market pages "
                "were not fetched.",
                extra={"cursor_present": True},
            )
        elif pagination_enabled:
            while (
                next_cursor
                and pages_fetched < max_pages
                and len(deduped_records) < effective_max_markets
                and (
                    effective_candidate_target is None
                    or candidate_matches_found < effective_candidate_target
                )
            ):
                if next_cursor in seen_cursors:
                    stop_reason = "repeated_cursor"
                    self.logger.warning(
                        "Kalshi pagination: repeated cursor detected, stopping. "
                        "pages_fetched=%d",
                        pages_fetched,
                    )
                    break
                seen_cursors.add(next_cursor)
                page_params: dict[str, Any] = {"limit": effective_limit, "cursor": next_cursor}
                if extra_params:
                    page_params.update(extra_params)
                page_payload = self._request_markets_page_with_retry(
                    params=page_params,
                    page_index=pages_fetched + 1,
                    cursor_present=True,
                    pages_fetched_so_far=pages_fetched,
                    total_markets_fetched_so_far=len(deduped_records),
                )
                if not isinstance(page_payload, dict):
                    raise KalshiAPIError(
                        "Kalshi paginated markets payload shape invalid: expected object page."
                    )
                page_records, _unused_key = self._extract_market_records_with_key(page_payload)
                if page_records is None:
                    raise KalshiAPIError(
                        "Kalshi paginated markets payload missing records list."
                    )
                if not page_records:
                    stop_reason = "empty_page"
                    self.logger.warning(
                        "Kalshi pagination: empty page received, stopping. "
                        "pages_fetched=%d",
                        pages_fetched,
                    )
                    pages_fetched += 1
                    next_cursor = self._extract_pagination_cursor(page_payload)
                    break
                page_candidate_matches = _append_records(page_records)
                pages_fetched += 1
                next_cursor = self._extract_pagination_cursor(page_payload)
                page_diagnostics.append(
                    {
                        "page_index": pages_fetched,
                        "records_on_page": len(page_records),
                        "sample_tickers": _sample_tickers(page_records),
                        "candidate_matches_page": page_candidate_matches,
                        "candidate_matches_cumulative": candidate_matches_found,
                    }
                )
                self.logger.info(
                    "Kalshi markets page diagnostics: page_index=%d records=%d "
                    "sample_tickers=%s candidate_matches_page=%d candidate_matches_cumulative=%d",
                    pages_fetched,
                    len(page_records),
                    _sample_tickers(page_records),
                    page_candidate_matches,
                    candidate_matches_found,
                )
                if len(deduped_records) >= effective_max_markets:
                    stop_reason = "market_cap_reached"
                    break

                if next_cursor and page_sleep_seconds > 0:
                    time.sleep(page_sleep_seconds)

        had_more_pages = bool(next_cursor)
        candidate_target_met = bool(
            effective_candidate_target is not None
            and candidate_matches_found >= effective_candidate_target
        )
        stopped_on_candidate_target = bool(candidate_target_met and had_more_pages)
        if stop_reason is None:
            if candidate_target_met:
                stop_reason = "target_reached"
            elif had_more_pages and pages_fetched >= max_pages:
                stop_reason = "page_cap_reached"
            elif had_more_pages and len(deduped_records) >= effective_max_markets:
                stop_reason = "market_cap_reached"
            else:
                stop_reason = "no_more_pages"
        summary = {
            "pagination_enabled": pagination_enabled,
            "pages_fetched": pages_fetched,
            "total_markets_fetched": len(deduped_records),
            "had_more_pages": had_more_pages,
            "deduped_count": deduped_count,
            "candidate_target": effective_candidate_target,
            "candidate_matches_found": candidate_matches_found,
            "candidate_target_met": candidate_target_met,
            "stopped_on_candidate_target": stopped_on_candidate_target,
            "stop_reason": stop_reason,
            "max_pages_per_fetch": max_pages,
            "max_markets_fetch": max_markets,
            "effective_max_markets": effective_max_markets,
            "page_diagnostics": page_diagnostics,
        }
        if effective_candidate_target is not None:
            self.logger.info(
                "Kalshi markets fetch summary: pages_fetched=%d total_markets_fetched=%d "
                "had_more_pages=%s deduped_count=%d candidate_matches_found=%d "
                "candidate_target=%d stopped_on_candidate_target=%s stop_reason=%s",
                pages_fetched,
                len(deduped_records),
                had_more_pages,
                deduped_count,
                candidate_matches_found,
                effective_candidate_target,
                stopped_on_candidate_target,
                stop_reason,
            )
        else:
            self.logger.info(
                "Kalshi markets fetch summary: pages_fetched=%d total_markets_fetched=%d "
                "had_more_pages=%s deduped_count=%d stop_reason=%s",
                pages_fetched,
                len(deduped_records),
                had_more_pages,
                deduped_count,
                stop_reason,
            )

        result: dict[str, Any] = dict(first_payload)
        canonical_key = list_key or "markets"
        result[canonical_key] = deduped_records
        if canonical_key != "markets":
            result["markets"] = deduped_records
        for cursor_key in ("cursor", "next_cursor", "next"):
            result.pop(cursor_key, None)
        if had_more_pages and next_cursor:
            result["cursor"] = next_cursor
        result["pagination"] = summary
        return result

    def create_order_raw(self, order_payload: dict[str, Any]) -> dict[str, Any]:
        """Submit one order payload to Kalshi.

        TODO(KALSHI_API): confirm exact request fields required by the orders endpoint.
        """
        payload = self._request_json(
            method="POST",
            endpoint=self.settings.kalshi_orders_endpoint,
            json_body=order_payload,
        )
        if not isinstance(payload, dict):
            raise KalshiAPIError("Order submit payload shape invalid: expected object response.")
        return payload

    def cancel_order_raw(self, order_id: str) -> dict[str, Any]:
        """Cancel an existing order by id.

        TODO(KALSHI_API): confirm cancel endpoint semantics and response schema.
        """
        attempts = self._build_cancel_request_attempts(order_id)
        last_error: KalshiRequestError | None = None
        for method, endpoint in attempts:
            try:
                payload = self._request_json(
                    method=method,
                    endpoint=endpoint,
                    json_body={} if method == "POST" else None,
                )
            except KalshiRequestError as exc:
                last_error = exc
                # Only continue fallback probing for route/method incompatibility.
                if exc.status_code in {404, 405}:
                    continue
                raise
            if not isinstance(payload, dict):
                raise KalshiAPIError(
                    "Order cancel payload shape invalid: expected object response."
                )
            return payload

        if last_error is not None:
            raise last_error
        raise KalshiAPIError("Order cancel request failed: no compatible endpoint/method found.")

    def get_order_raw(self, order_id: str) -> dict[str, Any]:
        """Fetch one order by id.

        TODO(KALSHI_API): confirm order status endpoint response schema.
        """
        endpoints = [
            self._render_order_endpoint(template=template, order_id=order_id)
            for template in self._build_status_endpoint_templates()
        ]
        seen: set[str] = set()
        first_payload: dict[str, Any] | None = None
        last_error: KalshiRequestError | None = None
        for endpoint in endpoints:
            if endpoint in seen:
                continue
            seen.add(endpoint)
            try:
                payload = self._request_json(method="GET", endpoint=endpoint)
            except KalshiRequestError as exc:
                last_error = exc
                if exc.status_code == 404:
                    continue
                raise
            if not isinstance(payload, dict):
                raise KalshiAPIError(
                    "Order status payload shape invalid: expected object response."
                )
            if first_payload is None:
                first_payload = payload
            if self._looks_like_order_status_payload(payload):
                return payload
        if first_payload is not None:
            return first_payload
        if last_error is not None:
            raise last_error
        raise KalshiAPIError("Order status request failed: no compatible endpoint found.")

    def list_orders_raw(self, limit: int = 50) -> dict[str, Any] | list[Any]:
        """Fetch recent orders.

        TODO(KALSHI_API): confirm accepted list params/filter names.
        """
        return self._request_json(
            method="GET",
            endpoint=self.settings.kalshi_orders_endpoint,
            params={"limit": limit},
        )

    def normalize_markets(self, payload: dict[str, Any] | list[Any]) -> list[MarketSummary]:
        """Normalize API-specific market payloads into internal summaries."""
        records = self.extract_market_records(payload)
        normalized: list[MarketSummary] = []
        for market in records:
            if isinstance(market, dict):
                normalized.append(self._normalize_market(market))
        return normalized

    def extract_market_records(self, payload: dict[str, Any] | list[Any]) -> list[Any]:
        """Extract market list from possible response container shapes."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            raise KalshiAPIError("Unexpected markets response type; expected dict or list.")

        records, key = self._extract_market_records_with_key(payload)
        if records is not None:
            if key not in {"markets", "data", "results", "items"}:
                self.logger.warning(
                    "Market records found under unexpected key %r; "
                    "update candidate_keys to silence this warning.",
                    key,
                )
            return records

        raise KalshiAPIError(
            "Unable to find market records in response payload. "
            "Verify KALSHI_MARKETS_ENDPOINT and response schema."
        )

    @staticmethod
    def _extract_market_records_with_key(
        payload: dict[str, Any],
    ) -> tuple[list[Any] | None, str | None]:
        candidate_keys = ("markets", "data", "results", "items")
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value, key

        for key, value in payload.items():
            if isinstance(value, list):
                return value, key
        return None, None

    @staticmethod
    def _extract_pagination_cursor(payload: dict[str, Any]) -> str | None:
        for key in ("cursor", "next_cursor", "next"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _looks_like_order_status_payload(payload: dict[str, Any]) -> bool:
        sources: list[dict[str, Any]] = [payload]
        for key in ("order", "data", "result"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                sources.append(nested)

        status_keys = {"status", "order_status", "state"}
        quantity_keys = {
            "filled_quantity",
            "filled_count",
            "filled_size",
            "remaining_quantity",
            "remaining_count",
            "remaining_size",
            "count",
            "quantity",
            "size",
            "requested_quantity",
            "requested_count",
        }
        id_keys = {"order_id", "id", "client_order_id"}

        for source in sources:
            if any(key in source for key in status_keys):
                return True
            if any(key in source for key in quantity_keys) and any(
                key in source for key in id_keys
            ):
                return True
        return False

    @staticmethod
    def _market_dedupe_key(record: Any) -> str | None:
        if not isinstance(record, dict):
            return None

        ticker = record.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            return f"ticker:{ticker.strip().upper()}"

        for key in ("id", "market_id"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return f"id:{value.strip()}"
            if isinstance(value, int):
                return f"id:{value}"
        return None

    def _get_bool_setting(self, name: str, default: bool) -> bool:
        value = getattr(self.settings, name, default)
        return value if isinstance(value, bool) else default

    def _get_int_setting(self, name: str, default: int) -> int:
        value = getattr(self.settings, name, default)
        if isinstance(value, bool):
            return default
        return value if isinstance(value, int) and value > 0 else default

    def _get_non_negative_int_setting(self, name: str, default: int) -> int:
        value = getattr(self.settings, name, default)
        if isinstance(value, bool):
            return default
        return value if isinstance(value, int) and value >= 0 else default

    def _request_markets_page_with_retry(
        self,
        *,
        params: dict[str, Any],
        page_index: int,
        cursor_present: bool,
        pages_fetched_so_far: int,
        total_markets_fetched_so_far: int,
    ) -> dict[str, Any] | list[Any]:
        """Fetch one markets page with page-scoped retry/backoff diagnostics."""
        max_retries = self._get_non_negative_int_setting("kalshi_page_fetch_max_retries", 2)
        base_ms = self._get_non_negative_int_setting("kalshi_page_retry_base_ms", 250)
        jitter_ms = self._get_non_negative_int_setting("kalshi_page_retry_jitter_ms", 100)

        last_error: KalshiRequestError | None = None
        attempts_used = 0
        for attempt in range(max_retries + 1):
            attempts_used = attempt + 1
            try:
                return self._request_json(
                    "GET",
                    self.settings.kalshi_markets_endpoint,
                    params=params,
                )
            except KalshiRequestError as exc:
                last_error = exc
                retryable = exc.category in {"network", "rate_limit", "server", "unknown"}
                if retryable and attempt < max_retries:
                    backoff_ms = base_ms * (2 ** attempt)
                    jitter = random.randint(0, jitter_ms) if jitter_ms > 0 else 0
                    delay_seconds = (backoff_ms + jitter) / 1000.0
                    self.logger.warning(
                        "Kalshi markets page request failed; retrying page=%d attempt=%d/%d "
                        "category=%s status=%s delay_ms=%d",
                        page_index,
                        attempt + 1,
                        max_retries + 1,
                        exc.category,
                        exc.status_code,
                        int(backoff_ms + jitter),
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                break

        error_message = (
            "Kalshi paginated markets fetch failed after retries "
            f"(page_index={page_index} cursor_present={cursor_present} "
            f"pages_fetched_so_far={pages_fetched_so_far} "
            f"total_markets_fetched_so_far={total_markets_fetched_so_far} "
            f"attempts_used={attempts_used}): "
            f"{sanitize_text(str(last_error)) if last_error else 'unknown error'}"
        )
        self.logger.error(error_message)
        raise KalshiAPIError(error_message) from last_error

    def _build_status_endpoint_templates(self) -> list[str]:
        primary = self.settings.kalshi_order_status_endpoint_template
        candidates: list[str] = [primary]

        if primary.endswith("/{order_id}"):
            candidates.append(f"{primary}/status")
        if primary.endswith("/{order_id}/cancel"):
            candidates.append(primary[: -len("/cancel")])

        swapped = self._swap_orders_path(primary)
        if swapped is not None:
            candidates.append(swapped)
            if swapped.endswith("/{order_id}"):
                candidates.append(f"{swapped}/status")

        return self._dedupe_preserve_order(candidates)

    def _build_cancel_request_attempts(self, order_id: str) -> list[tuple[str, str]]:
        primary_template = self.settings.kalshi_order_cancel_endpoint_template
        templates: list[str] = [primary_template]
        if not primary_template.endswith("/{order_id}/cancel"):
            templates.append(f"{primary_template}/cancel")
        if primary_template.endswith("/{order_id}/cancel"):
            templates.append(primary_template[: -len("/cancel")])

        swapped = self._swap_orders_path(primary_template)
        if swapped is not None:
            templates.append(swapped)
            if not swapped.endswith("/{order_id}/cancel"):
                templates.append(f"{swapped}/cancel")

        methods = ("DELETE", "POST")
        attempts: list[tuple[str, str]] = []
        for template in self._dedupe_preserve_order(templates):
            endpoint = self._render_order_endpoint(template=template, order_id=order_id)
            for method in methods:
                attempts.append((method, endpoint))
        return self._dedupe_attempts(attempts)

    @staticmethod
    def _swap_orders_path(template: str) -> str | None:
        if "/portfolio/orders/" in template:
            return template.replace("/portfolio/orders/", "/orders/")
        if "/orders/" in template and "/portfolio/orders/" not in template:
            return template.replace("/orders/", "/portfolio/orders/")
        return None

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _dedupe_attempts(attempts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for attempt in attempts:
            if attempt in seen:
                continue
            seen.add(attempt)
            result.append(attempt)
        return result

    def filter_open_weather_markets(self, markets: list[MarketSummary]) -> list[MarketSummary]:
        """Return markets that appear to be both weather-related and currently open."""
        result: list[MarketSummary] = []
        for market in markets:
            is_open = self._looks_open(market)
            is_weather = self._looks_weather(market)
            if is_open and is_weather:
                result.append(market)
            elif is_weather and not is_open:
                self.logger.debug(
                    "Weather market filtered out (not open): %s", market.market_id
                )
        self.logger.info(
            "Weather filter: %d/%d markets passed", len(result), len(markets)
        )
        return result

    def _request_json(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        last_error: Exception | None = None
        request_endpoint, endpoint_query_params = self._split_endpoint_query(endpoint)
        merged_params = self._merge_request_params(endpoint_query_params, params)

        for attempt in range(self.max_retries + 1):
            try:
                # Build auth headers per-attempt: RSA-PSS signatures include
                # a fresh timestamp so they must be regenerated on retry.
                auth_headers = self._build_auth_headers(method=method, path=request_endpoint)
                response = self._client.request(
                    method=method,
                    url=request_endpoint,
                    params=merged_params,
                    json=json_body,
                    headers=auth_headers,
                )
                if response.status_code in (401, 403):
                    raise KalshiRequestError(
                        f"Authentication failed with status {response.status_code}. "
                        "Verify credentials and auth mode.",
                        category="auth",
                        status_code=response.status_code,
                    )
                response.raise_for_status()
                try:
                    body: dict[str, Any] | list[Any] = response.json()
                except ValueError as exc:
                    raise KalshiRequestError(
                        "Response was not valid JSON.",
                        category="unknown",
                        status_code=response.status_code,
                    ) from exc
                return body
            except httpx.HTTPStatusError as exc:
                last_error = exc
                # Never retry client errors (4xx) except 429 rate-limit.
                sc = exc.response.status_code
                if 400 <= sc < 500 and sc != 429:
                    if sc in {400, 404, 409, 422}:
                        category = "validation"
                    elif sc in {401, 403}:
                        category = "permission"
                    else:
                        category = "unknown"
                    raise KalshiRequestError(
                        "Kalshi API client error "
                        f"{sc}: {sanitize_text(exc.response.text[:300])}",
                        category=category,
                        status_code=sc,
                    ) from exc
                if attempt < self.max_retries:
                    self.logger.warning(
                        "Kalshi request failed (HTTP %d); retrying",
                        sc,
                        extra={"attempt": attempt + 1, "max_retries": self.max_retries},
                    )
                    time.sleep(self.retry_delay_seconds)
                    continue
                category = "rate_limit" if sc == 429 else "server" if sc >= 500 else "unknown"
                raise KalshiRequestError(
                    f"Kalshi API request failed with status {sc}: "
                    f"{sanitize_text(exc.response.text[:300])}",
                    category=category,
                    status_code=sc,
                ) from exc
            except httpx.HTTPError as exc:
                # Catches all transport/protocol errors: TimeoutException,
                # NetworkError, DecodingError, TooManyRedirects, etc.
                last_error = exc
                if attempt < self.max_retries:
                    self.logger.warning(
                        "Kalshi request failed (%s); retrying",
                        type(exc).__name__,
                        extra={"attempt": attempt + 1, "max_retries": self.max_retries},
                    )
                    time.sleep(self.retry_delay_seconds)
                    continue
                raise KalshiRequestError(
                    f"Kalshi API request failed: {sanitize_text(str(exc))}",
                    category="network",
                    status_code=None,
                ) from exc

        raise KalshiRequestError(
            "Kalshi API request failed after retries: "
            f"{sanitize_text(str(last_error)) if last_error else 'unknown error'}",
            category="unknown",
            status_code=None,
        )

    @staticmethod
    def _split_endpoint_query(endpoint: str) -> tuple[str, dict[str, Any]]:
        """Split endpoint path and query params for robust param merging."""
        split = urlsplit(endpoint)
        path = split.path or endpoint
        query_params: dict[str, Any] = {}
        if split.query:
            for key, value in parse_qsl(split.query, keep_blank_values=True):
                query_params[key] = value
        return path, query_params

    @staticmethod
    def _merge_request_params(
        endpoint_params: dict[str, Any],
        request_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Merge endpoint query params with explicit request params.

        Explicit request params take precedence.
        """
        merged: dict[str, Any] = dict(endpoint_params)
        if request_params:
            merged.update(request_params)
        return merged or None

    @staticmethod
    def _render_order_endpoint(template: str, order_id: str) -> str:
        if not order_id:
            raise KalshiAPIError("order_id must not be empty.")
        return template.format(order_id=quote(order_id, safe=""))

    def _build_auth_headers(
        self, method: str = "GET", path: str = "/"
    ) -> dict[str, str]:
        """Build auth headers from configuration.

        For bearer mode: simple Authorization header.
        For key_secret mode: RSA-PSS signed request per Kalshi API spec.

        The signed message format is: ``{timestamp_ms}{METHOD}{path}``
        where path excludes query parameters.
        """
        if self.settings.kalshi_auth_mode == "bearer":
            token = self.settings.kalshi_bearer_token
            if not token:
                raise KalshiAPIError("Missing bearer token.")
            return {"Authorization": f"Bearer {token}"}

        key_id = self.settings.kalshi_api_key_id
        if not key_id:
            raise KalshiAPIError("Missing KALSHI_API_KEY_ID.")
        if self._rsa_private_key is None:
            raise KalshiAPIError(
                "RSA private key not loaded. Check KALSHI_API_KEY_SECRET_FILE."
            )

        # Timestamp in milliseconds.
        timestamp_ms = str(int(time.time() * 1000))
        # Message to sign: {timestamp}{METHOD}{path} — path excludes query params.
        message = f"{timestamp_ms}{method.upper()}{path}"
        signature = _sign_pss_text(self._rsa_private_key, message)

        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def _normalize_market(self, market: dict[str, Any]) -> MarketSummary:
        market_id = self._first_string(
            market, ("market_id", "id", "ticker", "event_ticker"), "unknown"
        )
        ticker = self._first_string(market, ("ticker", "event_ticker", "symbol"))
        title = self._first_string(
            market,
            ("title", "market_title", "event_title", "subtitle", "name"),
            default="(untitled market)",
        )
        status = self._first_string(market, ("status", "market_status"))
        category = self._first_string(market, ("category", "series", "event_category", "group"))

        close_time = self._first_datetime(
            market, ("close_time", "close_ts", "end_date", "end_time", "expiration_time")
        )
        settlement_time = self._first_datetime(
            market,
            ("settlement_time", "settles_at", "settlement_ts", "expiry_time"),
        )

        return MarketSummary(
            market_id=market_id,
            ticker=ticker,
            title=title,
            status=status,
            close_time=close_time,
            settlement_time=settlement_time,
            category=category,
            raw=market,
        )

    @staticmethod
    def _first_string(
        payload: dict[str, Any],
        keys: tuple[str, ...],
        default: str | None = None,
    ) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    @staticmethod
    def _first_datetime(payload: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
        for key in keys:
            value = payload.get(key)
            parsed = KalshiClient._parse_datetime(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except (ValueError, OSError):
                return None
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        return None

    @staticmethod
    def _looks_open(market: MarketSummary) -> bool:
        status = (market.status or "").strip().lower()
        if status in {"open", "active", "trading", "listed", "live"}:
            return True
        if status in {"closed", "settled", "inactive", "expired"}:
            return False

        raw = market.raw
        for key in ("is_open", "open", "trading_open", "active"):
            value = raw.get(key)
            if isinstance(value, bool):
                return value
        return False

    @staticmethod
    def _looks_weather(market: MarketSummary) -> bool:
        keyword_blob_parts: list[str] = []
        for value in (
            market.ticker,
            market.title,
            market.category,
            market.raw.get("series"),
            market.raw.get("event_title"),
            market.raw.get("subtitle"),
            market.raw.get("description"),
        ):
            if isinstance(value, str):
                keyword_blob_parts.append(value.lower())

        tags = market.raw.get("tags")
        if isinstance(tags, list):
            keyword_blob_parts.extend(str(tag).lower() for tag in tags)

        blob = " ".join(keyword_blob_parts)
        # Word-boundary matching to avoid false positives like "wind" in "winding"
        # or "rain" in "training". Use multi-word phrases where single words are
        # ambiguous.
        weather_keywords = (
            "weather", "temperature", "rain(?:fall)?", "snow(?:fall)?",
            "precip(?:itation)?", "hurricane", "wind speed", "windspeed",
            "storm", "tornado", "climate", "freeze", "heatwave", "heat wave",
            "high temp", "low temp", "degrees",
        )
        return any(re.search(rf"\b{kw}\b", blob) for kw in weather_keywords)

    @staticmethod
    def _has_market_like_sample(records: list[Any]) -> bool:
        """Return True when at least one record resembles a market object."""
        for item in records:
            if not isinstance(item, dict):
                continue
            has_identity = any(
                isinstance(item.get(key), str) and item.get(key)
                for key in ("market_id", "id", "ticker", "event_ticker")
            )
            context_keys = (
                "title", "market_title", "event_title",
                "name", "status", "market_status",
            )
            has_market_context = any(
                isinstance(item.get(key), str) and item.get(key)
                for key in context_keys
            )
            if has_identity and has_market_context:
                return True
        return False

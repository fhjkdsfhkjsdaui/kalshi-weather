"""Kalshi API adapter for Day 1 read-only market discovery."""

from __future__ import annotations

import base64
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

    def fetch_markets_raw(self, limit: int | None = None) -> dict[str, Any] | list[Any]:
        """Fetch raw markets payload from Kalshi, with optional cursor pagination."""
        effective_limit = limit or self.settings.kalshi_default_limit
        params: dict[str, Any] = {"limit": effective_limit}
        # TODO(KALSHI_API): confirm if status/open filter query params are supported.
        first_payload = self._request_json(
            "GET",
            self.settings.kalshi_markets_endpoint,
            params=params,
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

        deduped_records: list[Any] = []
        seen_keys: set[str] = set()
        seen_cursors: set[str] = set()
        deduped_count = 0

        def _append_records(page_records: list[Any]) -> None:
            nonlocal deduped_count
            for record in page_records:
                key = self._market_dedupe_key(record)
                if key is not None:
                    if key in seen_keys:
                        deduped_count += 1
                        continue
                    seen_keys.add(key)
                deduped_records.append(record)

        _append_records(records)
        if len(deduped_records) > max_markets:
            deduped_records = deduped_records[:max_markets]

        pages_fetched = 1
        next_cursor = self._extract_pagination_cursor(first_payload)
        if not pagination_enabled and next_cursor:
            self.logger.warning(
                "Kalshi pagination disabled for this run; additional market pages "
                "were not fetched.",
                extra={"cursor_present": True},
            )
        elif pagination_enabled:
            while (
                next_cursor
                and pages_fetched < max_pages
                and len(deduped_records) < max_markets
            ):
                if next_cursor in seen_cursors:
                    self.logger.warning(
                        "Kalshi pagination: repeated cursor detected, stopping. "
                        "pages_fetched=%d",
                        pages_fetched,
                    )
                    break
                seen_cursors.add(next_cursor)
                page_payload = self._request_json(
                    "GET",
                    self.settings.kalshi_markets_endpoint,
                    params={"limit": effective_limit, "cursor": next_cursor},
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
                    self.logger.warning(
                        "Kalshi pagination: empty page received, stopping. "
                        "pages_fetched=%d",
                        pages_fetched,
                    )
                    pages_fetched += 1
                    next_cursor = self._extract_pagination_cursor(page_payload)
                    break
                _append_records(page_records)
                if len(deduped_records) >= max_markets:
                    deduped_records = deduped_records[:max_markets]
                    pages_fetched += 1
                    next_cursor = self._extract_pagination_cursor(page_payload)
                    break

                pages_fetched += 1
                next_cursor = self._extract_pagination_cursor(page_payload)
                if next_cursor and page_sleep_seconds > 0:
                    time.sleep(page_sleep_seconds)

        had_more_pages = bool(next_cursor)
        summary = {
            "pagination_enabled": pagination_enabled,
            "pages_fetched": pages_fetched,
            "total_markets_fetched": len(deduped_records),
            "had_more_pages": had_more_pages,
            "deduped_count": deduped_count,
            "max_pages_per_fetch": max_pages,
            "max_markets_fetch": max_markets,
        }
        self.logger.info(
            "Kalshi markets fetch summary: pages_fetched=%d total_markets_fetched=%d "
            "had_more_pages=%s deduped_count=%d",
            pages_fetched,
            len(deduped_records),
            had_more_pages,
            deduped_count,
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
        endpoint = self._render_order_endpoint(
            template=self.settings.kalshi_order_cancel_endpoint_template,
            order_id=order_id,
        )
        payload = self._request_json(method="DELETE", endpoint=endpoint)
        if not isinstance(payload, dict):
            raise KalshiAPIError("Order cancel payload shape invalid: expected object response.")
        return payload

    def get_order_raw(self, order_id: str) -> dict[str, Any]:
        """Fetch one order by id.

        TODO(KALSHI_API): confirm order status endpoint response schema.
        """
        endpoint = self._render_order_endpoint(
            template=self.settings.kalshi_order_status_endpoint_template,
            order_id=order_id,
        )
        payload = self._request_json(method="GET", endpoint=endpoint)
        if not isinstance(payload, dict):
            raise KalshiAPIError("Order status payload shape invalid: expected object response.")
        return payload

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

        for attempt in range(self.max_retries + 1):
            try:
                # Build auth headers per-attempt: RSA-PSS signatures include
                # a fresh timestamp so they must be regenerated on retry.
                auth_headers = self._build_auth_headers(method=method, path=endpoint)
                response = self._client.request(
                    method=method,
                    url=endpoint,
                    params=params,
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

"""Typed settings loader for Day 1 Kalshi weather bot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AnyUrl, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigError


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["dev", "staging", "prod"] = Field(default="dev", alias="APP_ENV")
    kalshi_api_base_url: AnyUrl = Field(alias="KALSHI_API_BASE_URL")
    kalshi_auth_mode: Literal["bearer", "key_secret"] = Field(
        default="bearer",
        alias="KALSHI_AUTH_MODE",
    )
    kalshi_bearer_token: str | None = Field(
        default=None, alias="KALSHI_BEARER_TOKEN", repr=False
    )
    kalshi_api_key_id: str | None = Field(
        default=None, alias="KALSHI_API_KEY_ID", repr=False
    )
    kalshi_api_key_secret: str | None = Field(
        default=None, alias="KALSHI_API_KEY_SECRET", repr=False
    )
    kalshi_api_key_secret_file: Path | None = Field(
        default=None, alias="KALSHI_API_KEY_SECRET_FILE", repr=False
    )

    kalshi_markets_endpoint: str = Field(
        default="/trade-api/v2/markets",
        alias="KALSHI_MARKETS_ENDPOINT",
    )
    kalshi_orders_endpoint: str = Field(
        default="/trade-api/v2/portfolio/orders",
        alias="KALSHI_ORDERS_ENDPOINT",
    )
    kalshi_order_status_endpoint_template: str = Field(
        default="/trade-api/v2/portfolio/orders/{order_id}",
        alias="KALSHI_ORDER_STATUS_ENDPOINT_TEMPLATE",
    )
    kalshi_order_cancel_endpoint_template: str = Field(
        default="/trade-api/v2/portfolio/orders/{order_id}",
        alias="KALSHI_ORDER_CANCEL_ENDPOINT_TEMPLATE",
    )
    kalshi_timeout_seconds: float = Field(default=15.0, alias="KALSHI_TIMEOUT_SECONDS")
    kalshi_default_limit: int = Field(default=200, alias="KALSHI_DEFAULT_LIMIT")
    kalshi_max_print: int = Field(default=10, alias="KALSHI_MAX_PRINT")

    journal_dir: Path = Field(default=Path("./data/journal"), alias="JOURNAL_DIR")
    raw_payload_dir: Path = Field(default=Path("./data/raw"), alias="RAW_PAYLOAD_DIR")
    weather_raw_payload_dir: Path = Field(
        default=Path("./data/raw/weather"),
        alias="WEATHER_RAW_PAYLOAD_DIR",
    )

    nws_user_agent: str = Field(
        default="kalshi-weather-bot/0.1 (contact: research@example.com)",
        alias="NWS_USER_AGENT",
    )
    weather_timeout_seconds: float = Field(default=15.0, alias="WEATHER_TIMEOUT_SECONDS")
    weather_journal_raw_payloads: bool = Field(default=True, alias="WEATHER_JOURNAL_RAW_PAYLOADS")
    weather_default_lat: float | None = Field(default=None, alias="WEATHER_DEFAULT_LAT")
    weather_default_lon: float | None = Field(default=None, alias="WEATHER_DEFAULT_LON")
    weather_max_print: int = Field(default=5, alias="WEATHER_MAX_PRINT")

    risk_enabled: bool = Field(default=True, alias="RISK_ENABLED")
    risk_min_parser_confidence: float = Field(default=0.75, alias="RISK_MIN_PARSER_CONFIDENCE")
    risk_max_weather_age_seconds: int = Field(default=1800, alias="RISK_MAX_WEATHER_AGE_SECONDS")
    risk_max_stake_per_trade_cents: int = Field(default=50, alias="RISK_MAX_STAKE_PER_TRADE_CENTS")
    risk_max_total_exposure_cents: int = Field(default=500, alias="RISK_MAX_TOTAL_EXPOSURE_CENTS")
    risk_max_exposure_per_market_cents: int = Field(
        default=200, alias="RISK_MAX_EXPOSURE_PER_MARKET_CENTS"
    )
    risk_max_concurrent_open_orders: int = Field(
        default=5,
        alias="RISK_MAX_CONCURRENT_OPEN_ORDERS",
    )
    risk_duplicate_intent_cooldown_seconds: int = Field(
        default=60,
        alias="RISK_DUPLICATE_INTENT_COOLDOWN_SECONDS",
    )
    risk_min_liquidity_contracts: int | None = Field(
        default=None,
        alias="RISK_MIN_LIQUIDITY_CONTRACTS",
    )
    risk_max_spread_cents: int | None = Field(default=None, alias="RISK_MAX_SPREAD_CENTS")
    risk_price_min_cents: int = Field(default=1, alias="RISK_PRICE_MIN_CENTS")
    risk_price_max_cents: int = Field(default=99, alias="RISK_PRICE_MAX_CENTS")
    risk_kill_switch: bool = Field(default=False, alias="RISK_KILL_SWITCH")
    dry_run_mode: bool = Field(default=True, alias="DRY_RUN_MODE")

    signal_enabled: bool = Field(default=True, alias="SIGNAL_ENABLED")
    signal_min_parser_confidence: float = Field(
        default=0.75,
        alias="SIGNAL_MIN_PARSER_CONFIDENCE",
    )
    signal_min_matcher_confidence: float = Field(
        default=0.6,
        alias="SIGNAL_MIN_MATCHER_CONFIDENCE",
    )
    signal_min_model_confidence: float = Field(
        default=0.6,
        alias="SIGNAL_MIN_MODEL_CONFIDENCE",
    )
    signal_min_edge: float = Field(default=0.03, alias="SIGNAL_MIN_EDGE")
    signal_max_candidates: int = Field(default=3, alias="SIGNAL_MAX_CANDIDATES")
    signal_max_markets_to_scan: int = Field(default=250, alias="SIGNAL_MAX_MARKETS_TO_SCAN")
    signal_staleness_override_seconds: int | None = Field(
        default=None,
        alias="SIGNAL_STALENESS_OVERRIDE_SECONDS",
    )
    edge_spread_buffer: float = Field(default=0.01, alias="EDGE_SPREAD_BUFFER")
    edge_fee_buffer: float = Field(default=0.0, alias="EDGE_FEE_BUFFER")
    signal_allow_unsupported: bool = Field(default=False, alias="SIGNAL_ALLOW_UNSUPPORTED")

    execution_mode: Literal["dry_run", "live_cancel_only", "live_micro"] = Field(
        default="dry_run",
        alias="EXECUTION_MODE",
    )
    allow_live_api: bool = Field(default=False, alias="ALLOW_LIVE_API")
    allow_live_fills: bool = Field(default=False, alias="ALLOW_LIVE_FILLS")
    cancel_only_enabled: bool = Field(default=False, alias="CANCEL_ONLY_ENABLED")
    cancel_only_max_attempts_per_run: int = Field(
        default=3,
        alias="CANCEL_ONLY_MAX_ATTEMPTS_PER_RUN",
    )
    cancel_only_max_qty: int = Field(default=1, alias="CANCEL_ONLY_MAX_QTY")
    cancel_only_poll_timeout_seconds: float = Field(
        default=15.0,
        alias="CANCEL_ONLY_POLL_TIMEOUT_SECONDS",
    )
    cancel_only_poll_interval_seconds: float = Field(
        default=0.5,
        alias="CANCEL_ONLY_POLL_INTERVAL_SECONDS",
    )
    cancel_only_cancel_delay_ms: int = Field(default=100, alias="CANCEL_ONLY_CANCEL_DELAY_MS")
    cancel_only_min_delay_between_attempts_ms: int = Field(
        default=100,
        alias="CANCEL_ONLY_MIN_DELAY_BETWEEN_ATTEMPTS_MS",
    )
    cancel_only_halt_on_any_fill: bool = Field(
        default=True,
        alias="CANCEL_ONLY_HALT_ON_ANY_FILL",
    )
    micro_mode_enabled: bool = Field(default=False, alias="MICRO_MODE_ENABLED")
    micro_max_notional_per_trade_dollars: float = Field(
        default=0.50,
        alias="MICRO_MAX_NOTIONAL_PER_TRADE_DOLLARS",
    )
    micro_max_trades_per_run: int = Field(default=1, alias="MICRO_MAX_TRADES_PER_RUN")
    micro_max_trades_per_day: int = Field(default=3, alias="MICRO_MAX_TRADES_PER_DAY")
    micro_max_open_positions: int = Field(default=1, alias="MICRO_MAX_OPEN_POSITIONS")
    micro_max_daily_gross_exposure_dollars: float = Field(
        default=5.00,
        alias="MICRO_MAX_DAILY_GROSS_EXPOSURE_DOLLARS",
    )
    micro_max_daily_realized_loss_dollars: float = Field(
        default=2.00,
        alias="MICRO_MAX_DAILY_REALIZED_LOSS_DOLLARS",
    )
    micro_min_seconds_between_trades: int = Field(
        default=60,
        alias="MICRO_MIN_SECONDS_BETWEEN_TRADES",
    )
    micro_halt_on_reconciliation_mismatch: bool = Field(
        default=True,
        alias="MICRO_HALT_ON_RECONCILIATION_MISMATCH",
    )
    micro_halt_on_any_unexpected_fill_state: bool = Field(
        default=True,
        alias="MICRO_HALT_ON_ANY_UNEXPECTED_FILL_STATE",
    )
    micro_require_supervised_mode: bool = Field(
        default=True,
        alias="MICRO_REQUIRE_SUPERVISED_MODE",
    )
    micro_min_parser_confidence: float = Field(
        default=0.80,
        alias="MICRO_MIN_PARSER_CONFIDENCE",
    )
    micro_min_edge: float = Field(default=0.05, alias="MICRO_MIN_EDGE")
    micro_max_weather_age_seconds: int = Field(
        default=900,
        alias="MICRO_MAX_WEATHER_AGE_SECONDS",
    )
    micro_poll_timeout_seconds: float = Field(
        default=20.0,
        alias="MICRO_POLL_TIMEOUT_SECONDS",
    )
    micro_poll_interval_seconds: float = Field(
        default=0.5,
        alias="MICRO_POLL_INTERVAL_SECONDS",
    )
    micro_market_scope_whitelist: str | None = Field(
        default=None,
        alias="MICRO_MARKET_SCOPE_WHITELIST",
    )

    @field_validator(
        "weather_default_lat",
        "weather_default_lon",
        "micro_market_scope_whitelist",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: Any) -> Any:
        """Treat empty env-string values as unset optional coordinates."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def validate_auth(self) -> Settings:
        """Validate auth fields according to the selected auth mode."""
        if self.kalshi_auth_mode == "bearer":
            if not self.kalshi_bearer_token:
                raise ValueError(
                    "KALSHI_BEARER_TOKEN is required when KALSHI_AUTH_MODE='bearer'."
                )
        elif self.kalshi_auth_mode == "key_secret":
            if not self.kalshi_api_key_id:
                raise ValueError(
                    "KALSHI_API_KEY_ID is required when KALSHI_AUTH_MODE='key_secret'."
                )
            # Load key from file if KALSHI_API_KEY_SECRET_FILE is set.
            if self.kalshi_api_key_secret_file is not None:
                key_path = self.kalshi_api_key_secret_file
                if not key_path.exists():
                    raise ValueError(
                        f"KALSHI_API_KEY_SECRET_FILE does not exist: {key_path}"
                    )
                try:
                    self.kalshi_api_key_secret = key_path.read_text().strip()
                except OSError as exc:
                    raise ValueError(
                        f"Failed reading KALSHI_API_KEY_SECRET_FILE ({key_path}): {exc}"
                    ) from exc
            if not self.kalshi_api_key_secret:
                raise ValueError(
                    "KALSHI_API_KEY_SECRET or KALSHI_API_KEY_SECRET_FILE is required "
                    "when KALSHI_AUTH_MODE='key_secret'."
                )

        if not self.kalshi_markets_endpoint.startswith("/"):
            raise ValueError("KALSHI_MARKETS_ENDPOINT must start with '/'.")
        if not self.kalshi_orders_endpoint.startswith("/"):
            raise ValueError("KALSHI_ORDERS_ENDPOINT must start with '/'.")
        if not self.kalshi_order_status_endpoint_template.startswith("/"):
            raise ValueError("KALSHI_ORDER_STATUS_ENDPOINT_TEMPLATE must start with '/'.")
        if "{order_id}" not in self.kalshi_order_status_endpoint_template:
            raise ValueError(
                "KALSHI_ORDER_STATUS_ENDPOINT_TEMPLATE must include '{order_id}'."
            )
        if not self.kalshi_order_cancel_endpoint_template.startswith("/"):
            raise ValueError("KALSHI_ORDER_CANCEL_ENDPOINT_TEMPLATE must start with '/'.")
        if "{order_id}" not in self.kalshi_order_cancel_endpoint_template:
            raise ValueError(
                "KALSHI_ORDER_CANCEL_ENDPOINT_TEMPLATE must include '{order_id}'."
            )
        if self.kalshi_timeout_seconds <= 0:
            raise ValueError("KALSHI_TIMEOUT_SECONDS must be > 0.")
        if self.kalshi_default_limit <= 0:
            raise ValueError("KALSHI_DEFAULT_LIMIT must be > 0.")
        if self.kalshi_max_print <= 0:
            raise ValueError("KALSHI_MAX_PRINT must be > 0.")
        if not self.nws_user_agent.strip():
            raise ValueError("NWS_USER_AGENT must not be empty.")
        if self.weather_timeout_seconds <= 0:
            raise ValueError("WEATHER_TIMEOUT_SECONDS must be > 0.")
        if self.weather_max_print <= 0:
            raise ValueError("WEATHER_MAX_PRINT must be > 0.")

        has_default_lat = self.weather_default_lat is not None
        has_default_lon = self.weather_default_lon is not None
        if has_default_lat != has_default_lon:
            raise ValueError("WEATHER_DEFAULT_LAT and WEATHER_DEFAULT_LON must be set together.")
        if has_default_lat and not (-90 <= self.weather_default_lat <= 90):
            raise ValueError("WEATHER_DEFAULT_LAT must be between -90 and 90.")
        if has_default_lon and not (-180 <= self.weather_default_lon <= 180):
            raise ValueError("WEATHER_DEFAULT_LON must be between -180 and 180.")
        if not (0 <= self.risk_min_parser_confidence <= 1):
            raise ValueError("RISK_MIN_PARSER_CONFIDENCE must be between 0 and 1.")
        if self.risk_max_weather_age_seconds <= 0:
            raise ValueError("RISK_MAX_WEATHER_AGE_SECONDS must be > 0.")
        if self.risk_max_stake_per_trade_cents <= 0:
            raise ValueError("RISK_MAX_STAKE_PER_TRADE_CENTS must be > 0.")
        if self.risk_max_total_exposure_cents <= 0:
            raise ValueError("RISK_MAX_TOTAL_EXPOSURE_CENTS must be > 0.")
        if self.risk_max_exposure_per_market_cents <= 0:
            raise ValueError("RISK_MAX_EXPOSURE_PER_MARKET_CENTS must be > 0.")
        if self.risk_max_exposure_per_market_cents > self.risk_max_total_exposure_cents:
            raise ValueError(
                "RISK_MAX_EXPOSURE_PER_MARKET_CENTS cannot exceed "
                "RISK_MAX_TOTAL_EXPOSURE_CENTS."
            )
        if self.risk_max_concurrent_open_orders <= 0:
            raise ValueError("RISK_MAX_CONCURRENT_OPEN_ORDERS must be > 0.")
        if self.risk_duplicate_intent_cooldown_seconds < 0:
            raise ValueError("RISK_DUPLICATE_INTENT_COOLDOWN_SECONDS must be >= 0.")
        if self.risk_price_min_cents <= 0:
            raise ValueError("RISK_PRICE_MIN_CENTS must be > 0.")
        if self.risk_price_max_cents > 99:
            raise ValueError("RISK_PRICE_MAX_CENTS must be <= 99.")
        if self.risk_price_min_cents >= self.risk_price_max_cents:
            raise ValueError("RISK_PRICE_MIN_CENTS must be less than RISK_PRICE_MAX_CENTS.")
        if (
            self.risk_min_liquidity_contracts is not None
            and self.risk_min_liquidity_contracts <= 0
        ):
            raise ValueError("RISK_MIN_LIQUIDITY_CONTRACTS must be > 0 when set.")
        if self.risk_max_spread_cents is not None and self.risk_max_spread_cents < 0:
            raise ValueError("RISK_MAX_SPREAD_CENTS must be >= 0 when set.")
        if not self.dry_run_mode:
            raise ValueError(
                "DRY_RUN_MODE must be true for the Day 4 dry-run execution skeleton."
            )
        if not (0 <= self.signal_min_parser_confidence <= 1):
            raise ValueError("SIGNAL_MIN_PARSER_CONFIDENCE must be between 0 and 1.")
        if not (0 <= self.signal_min_matcher_confidence <= 1):
            raise ValueError("SIGNAL_MIN_MATCHER_CONFIDENCE must be between 0 and 1.")
        if not (0 <= self.signal_min_model_confidence <= 1):
            raise ValueError("SIGNAL_MIN_MODEL_CONFIDENCE must be between 0 and 1.")
        if not (0 <= self.signal_min_edge <= 1):
            raise ValueError("SIGNAL_MIN_EDGE must be between 0 and 1.")
        if self.signal_max_candidates <= 0:
            raise ValueError("SIGNAL_MAX_CANDIDATES must be > 0.")
        if self.signal_max_markets_to_scan <= 0:
            raise ValueError("SIGNAL_MAX_MARKETS_TO_SCAN must be > 0.")
        if (
            self.signal_staleness_override_seconds is not None
            and self.signal_staleness_override_seconds <= 0
        ):
            raise ValueError("SIGNAL_STALENESS_OVERRIDE_SECONDS must be > 0 when set.")
        if not (0 <= self.edge_spread_buffer <= 1):
            raise ValueError("EDGE_SPREAD_BUFFER must be between 0 and 1.")
        if not (0 <= self.edge_fee_buffer <= 1):
            raise ValueError("EDGE_FEE_BUFFER must be between 0 and 1.")
        if self.cancel_only_max_attempts_per_run <= 0:
            raise ValueError("CANCEL_ONLY_MAX_ATTEMPTS_PER_RUN must be > 0.")
        if self.cancel_only_max_qty <= 0:
            raise ValueError("CANCEL_ONLY_MAX_QTY must be > 0.")
        if self.cancel_only_poll_timeout_seconds <= 0:
            raise ValueError("CANCEL_ONLY_POLL_TIMEOUT_SECONDS must be > 0.")
        if self.cancel_only_poll_interval_seconds <= 0:
            raise ValueError("CANCEL_ONLY_POLL_INTERVAL_SECONDS must be > 0.")
        if self.cancel_only_poll_interval_seconds > self.cancel_only_poll_timeout_seconds:
            raise ValueError(
                "CANCEL_ONLY_POLL_INTERVAL_SECONDS cannot exceed "
                "CANCEL_ONLY_POLL_TIMEOUT_SECONDS."
            )
        if self.cancel_only_cancel_delay_ms < 0:
            raise ValueError("CANCEL_ONLY_CANCEL_DELAY_MS must be >= 0.")
        if self.cancel_only_min_delay_between_attempts_ms < 0:
            raise ValueError("CANCEL_ONLY_MIN_DELAY_BETWEEN_ATTEMPTS_MS must be >= 0.")
        if self.execution_mode == "live_cancel_only":
            if not self.allow_live_api:
                raise ValueError(
                    "ALLOW_LIVE_API must be true when EXECUTION_MODE='live_cancel_only'."
                )
            if not self.cancel_only_enabled:
                raise ValueError(
                    "CANCEL_ONLY_ENABLED must be true when EXECUTION_MODE='live_cancel_only'."
                )
        if self.micro_max_notional_per_trade_dollars <= 0:
            raise ValueError("MICRO_MAX_NOTIONAL_PER_TRADE_DOLLARS must be > 0.")
        if self.micro_max_trades_per_run <= 0:
            raise ValueError("MICRO_MAX_TRADES_PER_RUN must be > 0.")
        if self.micro_max_trades_per_day <= 0:
            raise ValueError("MICRO_MAX_TRADES_PER_DAY must be > 0.")
        if self.micro_max_open_positions <= 0:
            raise ValueError("MICRO_MAX_OPEN_POSITIONS must be > 0.")
        if self.micro_max_daily_gross_exposure_dollars <= 0:
            raise ValueError("MICRO_MAX_DAILY_GROSS_EXPOSURE_DOLLARS must be > 0.")
        if self.micro_max_daily_realized_loss_dollars <= 0:
            raise ValueError("MICRO_MAX_DAILY_REALIZED_LOSS_DOLLARS must be > 0.")
        if self.micro_min_seconds_between_trades < 0:
            raise ValueError("MICRO_MIN_SECONDS_BETWEEN_TRADES must be >= 0.")
        if not (0 <= self.micro_min_parser_confidence <= 1):
            raise ValueError("MICRO_MIN_PARSER_CONFIDENCE must be between 0 and 1.")
        if not (0 <= self.micro_min_edge <= 1):
            raise ValueError("MICRO_MIN_EDGE must be between 0 and 1.")
        if self.micro_max_weather_age_seconds <= 0:
            raise ValueError("MICRO_MAX_WEATHER_AGE_SECONDS must be > 0.")
        if self.micro_poll_timeout_seconds <= 0:
            raise ValueError("MICRO_POLL_TIMEOUT_SECONDS must be > 0.")
        if self.micro_poll_interval_seconds <= 0:
            raise ValueError("MICRO_POLL_INTERVAL_SECONDS must be > 0.")
        if self.micro_poll_interval_seconds > self.micro_poll_timeout_seconds:
            raise ValueError(
                "MICRO_POLL_INTERVAL_SECONDS cannot exceed MICRO_POLL_TIMEOUT_SECONDS."
            )
        micro_notional_cents = int(round(self.micro_max_notional_per_trade_dollars * 100))
        if micro_notional_cents > self.risk_max_stake_per_trade_cents:
            raise ValueError(
                "MICRO_MAX_NOTIONAL_PER_TRADE_DOLLARS cannot exceed "
                "RISK_MAX_STAKE_PER_TRADE_CENTS."
            )
        if self.micro_max_open_positions > self.risk_max_concurrent_open_orders:
            raise ValueError(
                "MICRO_MAX_OPEN_POSITIONS cannot exceed RISK_MAX_CONCURRENT_OPEN_ORDERS."
            )
        if self.execution_mode == "live_micro":
            if not self.allow_live_api:
                raise ValueError("ALLOW_LIVE_API must be true when EXECUTION_MODE='live_micro'.")
            if not self.allow_live_fills:
                raise ValueError(
                    "ALLOW_LIVE_FILLS must be true when EXECUTION_MODE='live_micro'."
                )
            if not self.micro_mode_enabled:
                raise ValueError(
                    "MICRO_MODE_ENABLED must be true when EXECUTION_MODE='live_micro'."
                )
        return self

    def safe_summary(self) -> dict[str, Any]:
        """Return config summary safe for journaling (no credentials)."""
        return {
            "app_env": self.app_env,
            "base_url": str(self.kalshi_api_base_url),
            "markets_endpoint": self.kalshi_markets_endpoint,
            "auth_mode": self.kalshi_auth_mode,
            "timeout_seconds": self.kalshi_timeout_seconds,
            "default_limit": self.kalshi_default_limit,
            "orders_endpoint": self.kalshi_orders_endpoint,
            "weather_timeout_seconds": self.weather_timeout_seconds,
            "weather_raw_journaling": self.weather_journal_raw_payloads,
            "risk_enabled": self.risk_enabled,
            "risk_min_parser_confidence": self.risk_min_parser_confidence,
            "risk_max_stake_per_trade_cents": self.risk_max_stake_per_trade_cents,
            "risk_max_total_exposure_cents": self.risk_max_total_exposure_cents,
            "dry_run_mode": self.dry_run_mode,
            "signal_enabled": self.signal_enabled,
            "signal_min_matcher_confidence": self.signal_min_matcher_confidence,
            "signal_min_model_confidence": self.signal_min_model_confidence,
            "signal_min_edge": self.signal_min_edge,
            "signal_max_candidates": self.signal_max_candidates,
            "execution_mode": self.execution_mode,
            "allow_live_api": self.allow_live_api,
            "allow_live_fills": self.allow_live_fills,
            "cancel_only_enabled": self.cancel_only_enabled,
            "cancel_only_max_attempts_per_run": self.cancel_only_max_attempts_per_run,
            "cancel_only_max_qty": self.cancel_only_max_qty,
            "micro_mode_enabled": self.micro_mode_enabled,
            "micro_max_notional_per_trade_dollars": self.micro_max_notional_per_trade_dollars,
            "micro_max_trades_per_run": self.micro_max_trades_per_run,
            "micro_max_trades_per_day": self.micro_max_trades_per_day,
            "micro_max_open_positions": self.micro_max_open_positions,
            "micro_max_daily_gross_exposure_dollars": self.micro_max_daily_gross_exposure_dollars,
            "micro_max_daily_realized_loss_dollars": self.micro_max_daily_realized_loss_dollars,
        }


def load_settings() -> Settings:
    """Load and validate settings, raising ConfigError on failure."""
    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed reading environment/.env: {exc}") from exc

    settings.journal_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_payload_dir.mkdir(parents=True, exist_ok=True)
    settings.weather_raw_payload_dir.mkdir(parents=True, exist_ok=True)
    return settings

"""Application exception classes."""


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


class KalshiAPIError(Exception):
    """Raised when Kalshi API calls fail or return malformed data."""


class KalshiRequestError(KalshiAPIError):
    """Raised for Kalshi request failures with category/status metadata."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


class JournalError(Exception):
    """Raised when writing to journal files fails."""


class WeatherProviderError(Exception):
    """Raised when weather provider requests or normalization fail."""


class ContractParserError(Exception):
    """Raised when contract parsing or audit input processing fails."""


class SignalProcessingError(Exception):
    """Raised when Day 5 signal matching/estimation/selection fails."""

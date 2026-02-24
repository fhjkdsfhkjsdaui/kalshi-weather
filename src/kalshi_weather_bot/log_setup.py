"""Logging setup for Day 1 command-line execution."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from .redaction import sanitize_text


class JsonConsoleFormatter(logging.Formatter):
    """Simple JSON formatter for structured console logs."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_text(record.getMessage()),
        }
        if record.exc_info:
            event["exception"] = sanitize_text(self.formatException(record.exc_info))
        return json.dumps(event, default=str)


def setup_logger(name: str = "kalshi_weather_bot", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a process-wide logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(JsonConsoleFormatter())
    logger.addHandler(handler)
    return logger

"""Typed event/state models for terminal dashboard presentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

Severity = Literal["INFO", "WARN", "ERROR", "CRITICAL"]


@dataclass(slots=True)
class DashboardEvent:
    """One terminal-visible event entry with optional dedupe metadata."""

    ts: datetime
    severity: Severity
    message: str
    count: int = 1
    last_seen: datetime | None = None
    dedupe_key: str | None = None

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            self.ts = self.ts.replace(tzinfo=UTC)
        else:
            self.ts = self.ts.astimezone(UTC)
        if self.last_seen is None:
            self.last_seen = self.ts
        elif self.last_seen.tzinfo is None:
            self.last_seen = self.last_seen.replace(tzinfo=UTC)
        else:
            self.last_seen = self.last_seen.astimezone(UTC)


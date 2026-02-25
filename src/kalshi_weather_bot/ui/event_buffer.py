"""Bounded event feed with repeat-deduplication for terminal dashboards."""

from __future__ import annotations

from datetime import UTC, datetime

from .models import DashboardEvent, Severity


class EventBuffer:
    """Keep recent events and collapse repeated warning/error lines."""

    def __init__(
        self,
        *,
        max_events: int = 80,
        dedupe_window_seconds: int = 30,
    ) -> None:
        self.max_events = max_events
        self.dedupe_window_seconds = dedupe_window_seconds
        self._events: list[DashboardEvent] = []
        self._dedupe_index: dict[str, DashboardEvent] = {}

    def add(
        self,
        *,
        severity: Severity,
        message: str,
        dedupe_key: str | None = None,
        ts: datetime | None = None,
    ) -> DashboardEvent:
        now = ts or datetime.now(UTC)
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)

        should_dedupe = severity in {"WARN", "ERROR"} and dedupe_key is not None
        if should_dedupe:
            existing = self._dedupe_index.get(dedupe_key)
            if existing and existing.last_seen is not None:
                age_seconds = (now - existing.last_seen).total_seconds()
                if age_seconds <= self.dedupe_window_seconds:
                    existing.count += 1
                    existing.last_seen = now
                    return existing

        event = DashboardEvent(
            ts=now,
            severity=severity,
            message=message,
            count=1,
            dedupe_key=dedupe_key if should_dedupe else None,
        )
        self._events.append(event)
        if should_dedupe and dedupe_key is not None:
            self._dedupe_index[dedupe_key] = event

        while len(self._events) > self.max_events:
            dropped = self._events.pop(0)
            if dropped.dedupe_key:
                indexed = self._dedupe_index.get(dropped.dedupe_key)
                if indexed is dropped:
                    self._dedupe_index.pop(dropped.dedupe_key, None)
        return event

    def snapshot(self, *, newest_first: bool = False) -> list[DashboardEvent]:
        """Return a copy of tracked events in display order."""
        items = list(self._events)
        if newest_first:
            items.reverse()
        return items

    def count_matching(
        self,
        *,
        severity: Severity | None = None,
        contains: str | None = None,
    ) -> int:
        """Count events, weighted by dedupe counts, for simple health metrics."""
        total = 0
        search = contains.lower() if contains else None
        for event in self._events:
            if severity is not None and event.severity != severity:
                continue
            if search is not None and search not in event.message.lower():
                continue
            total += event.count
        return total

"""Tests for terminal event dedupe buffer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_weather_bot.ui.event_buffer import EventBuffer


def test_warning_events_dedupe_within_window() -> None:
    buffer = EventBuffer(max_events=10, dedupe_window_seconds=30)
    start = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    buffer.add(
        severity="WARN",
        message="Pagination cursor seen; additional pages remain.",
        dedupe_key="warn:pagination_cursor",
        ts=start,
    )
    buffer.add(
        severity="WARN",
        message="Pagination cursor seen; additional pages remain.",
        dedupe_key="warn:pagination_cursor",
        ts=start + timedelta(seconds=5),
    )
    events = buffer.snapshot()
    assert len(events) == 1
    assert events[0].count == 2


def test_warning_events_new_entry_after_window() -> None:
    buffer = EventBuffer(max_events=10, dedupe_window_seconds=5)
    start = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    buffer.add(
        severity="WARN",
        message="Repeated warning",
        dedupe_key="warn:repeat",
        ts=start,
    )
    buffer.add(
        severity="WARN",
        message="Repeated warning",
        dedupe_key="warn:repeat",
        ts=start + timedelta(seconds=6),
    )
    events = buffer.snapshot()
    assert len(events) == 2
    assert events[0].count == 1
    assert events[1].count == 1


def test_critical_events_are_not_deduped() -> None:
    buffer = EventBuffer(max_events=10, dedupe_window_seconds=60)
    now = datetime(2026, 2, 24, 12, 0, tzinfo=UTC)
    buffer.add(
        severity="CRITICAL",
        message="Unexpected fill detected",
        dedupe_key="critical:fill",
        ts=now,
    )
    buffer.add(
        severity="CRITICAL",
        message="Unexpected fill detected",
        dedupe_key="critical:fill",
        ts=now + timedelta(seconds=1),
    )
    events = buffer.snapshot()
    assert len(events) == 2
    assert all(event.count == 1 for event in events)


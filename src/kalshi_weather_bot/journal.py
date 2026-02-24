"""Append-only JSONL journaling utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exceptions import JournalError
from .redaction import sanitize_for_logging, sanitize_text


def _json_default(value: Any) -> Any:
    """Fallback serializer for non-JSON native values."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    # Allow objects with custom __str__ (e.g. Pydantic AnyUrl), but reject
    # truly unknown types so schema drift surfaces early.
    if hasattr(value, "__str__") and type(value).__str__ is not object.__str__:
        return sanitize_text(str(value))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class JournalWriter:
    """Writes event records to JSONL and raw payload snapshots to disk."""

    def __init__(self, journal_dir: Path, raw_payload_dir: Path, session_id: str) -> None:
        self.journal_dir = journal_dir
        self.raw_payload_dir = raw_payload_dir
        self.session_id = session_id
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.raw_payload_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.journal_dir / f"{datetime.now(UTC):%Y%m%d}.jsonl"

    def write_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a single event record to the JSONL journal."""
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "session_id": self.session_id,
            "payload": sanitize_for_logging(payload),
            "metadata": sanitize_for_logging(metadata or {}),
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=_json_default))
                fh.write("\n")
        except (OSError, TypeError, ValueError) as exc:
            raise JournalError(f"Failed writing event journal: {exc}") from exc

    def write_raw_snapshot(self, name: str, payload: Any) -> Path:
        """Write full raw payload snapshot and return file path."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
        output_path = self.raw_payload_dir / f"{timestamp}_{self.session_id}_{safe_name}.json"
        try:
            sanitized_payload = sanitize_for_logging(payload)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    sanitized_payload,
                    fh,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
                fh.write("\n")
        except (OSError, TypeError, ValueError) as exc:
            raise JournalError(f"Failed writing raw payload snapshot: {exc}") from exc
        return output_path

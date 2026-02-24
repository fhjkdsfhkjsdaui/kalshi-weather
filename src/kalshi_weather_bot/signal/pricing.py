"""Field-aware price parsing helpers for Day 5 signal modules."""

from __future__ import annotations

import math
from typing import Any, Literal


def parse_cents_field(value: Any) -> int | None:
    """Parse a cents-denominated field to integer cents.

    This parser is strict by design: cents fields must represent whole-cent values.
    Examples:
    - ``"42"`` -> ``42``
    - ``1`` -> ``1``
    - ``"1.0000"`` -> ``1``
    - ``"0.50"`` -> ``None`` (fractional dollars should use dollars fields)
    """
    numeric = _parse_numeric(value)
    if numeric is None or numeric < 0 or numeric > 100:
        return None
    rounded = round(numeric)
    if not math.isclose(numeric, rounded, abs_tol=1e-9):
        return None
    return int(rounded)


def dollars_to_cents(value: float) -> int | None:
    """Convert dollar amount in [0, 1] to integer cents."""
    if value < 0 or value > 1:
        return None
    return int(round(value * 100))


def parse_dollars_field(value: Any) -> int | None:
    """Parse a dollars-denominated field to integer cents."""
    numeric = _parse_numeric(value)
    if numeric is None:
        return None
    return dollars_to_cents(numeric)


def parse_price_field(field_name: str, value: Any) -> int | None:
    """Parse one market price field into integer cents using field semantics."""
    if field_name.endswith("_dollars"):
        return parse_dollars_field(value)
    return parse_cents_field(value)


def parse_probability_field(value: Any, *, unit: Literal["fraction", "percent"]) -> float | None:
    """Parse known-unit probability fields to a 0..1 fraction."""
    numeric = _parse_numeric(value)
    if numeric is None:
        return None
    if unit == "fraction":
        if 0 <= numeric <= 1:
            return float(numeric)
        return None
    if 0 <= numeric <= 100:
        return float(numeric) / 100.0
    return None


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric

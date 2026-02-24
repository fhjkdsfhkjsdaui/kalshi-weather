"""Typed models for normalized Kalshi weather contract parsing."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ParseStatus = Literal["parsed", "ambiguous", "unsupported", "rejected"]
WeatherDimension = Literal["temperature", "precipitation", "snowfall", "wind", "other"]
ThresholdOperator = Literal[">", ">=", "<", "<=", "between", "exact", "range"]


class NormalizedLocation(BaseModel):
    """Normalized location fields inferred from raw contract text."""

    city: str | None = None
    state: str | None = None


class ParsedWeatherContract(BaseModel):
    """Normalized parsed representation for a single Kalshi market contract."""

    provider_market_id: str
    ticker: str | None = None
    event_id: str | None = None

    raw_title: str
    raw_subtitle: str | None = None
    raw_rules_primary: str | None = None
    raw_rules_secondary: str | None = None

    weather_candidate: bool = False
    weather_dimension: WeatherDimension | None = None
    metric_subtype: str | None = None

    location_raw: str | None = None
    location_normalized: NormalizedLocation | None = None

    threshold_operator: ThresholdOperator | None = None
    threshold_value: float | None = None
    threshold_low: float | None = None
    threshold_high: float | None = None
    threshold_unit: str | None = None

    contract_start_time: datetime | None = None
    contract_end_time: datetime | None = None
    resolution_time: datetime | None = None

    yes_side_semantics: str | None = None
    no_side_semantics: str | None = None

    parse_confidence: float = Field(ge=0.0, le=1.0)
    parse_status: ParseStatus
    rejection_reasons: list[str] = Field(default_factory=list)


class ContractParseSummary(BaseModel):
    """Aggregate summary for parser audit runs."""

    total_markets_scanned: int
    weather_candidates: int
    parsed: int
    ambiguous: int
    unsupported: int
    rejected: int
    top_rejection_reasons: dict[str, int] = Field(default_factory=dict)


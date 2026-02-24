"""Typed models for Day 5 signal matching/estimation/selection pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..contracts.models import ParsedWeatherContract
from ..weather.models import WeatherForecastPeriod


class MatchResult(BaseModel):
    """Result of aligning a parsed contract with weather forecast periods."""

    matched: bool
    decision_code: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    selected_period_indices: list[int] = Field(default_factory=list)
    selected_period_count: int = 0
    selected_periods: list[WeatherForecastPeriod] = Field(default_factory=list)
    time_alignment: dict[str, Any] = Field(default_factory=dict)
    unit_normalization: dict[str, Any] = Field(default_factory=dict)
    matcher_confidence: float = Field(ge=0.0, le=1.0)


class EstimateResult(BaseModel):
    """Result of estimating contract outcome probability from matched weather data."""

    available: bool
    decision_code: str
    estimated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    model_confidence: float = Field(ge=0.0, le=1.0)
    estimate_method: str
    assumptions: list[str] = Field(default_factory=list)
    missing_data_flags: list[str] = Field(default_factory=list)
    supporting_values: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EdgeResult(BaseModel):
    """Result of comparing model probability to market implied prices."""

    valid: bool
    decision_code: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    yes_price_cents: int | None = None
    no_price_cents: int | None = None
    market_price_field_yes: str | None = None
    market_price_field_no: str | None = None
    market_implied_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    model_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_yes: float | None = None
    edge_no: float | None = None
    recommended_side: Literal["yes", "no"] | None = None
    recommended_price_cents: int | None = None
    edge_after_buffers: float | None = None
    spread_buffer: float = 0.0
    fee_buffer: float = 0.0


class SignalEvaluation(BaseModel):
    """Full per-market signal evaluation record before candidate selection."""

    market_id: str
    ticker: str | None = None
    title: str
    market_raw: dict[str, Any] = Field(default_factory=dict)
    parsed_contract: ParsedWeatherContract
    weather_snapshot_ref: str
    weather_snapshot_retrieved_at: datetime
    weather_age_seconds: float
    match_result: MatchResult
    estimate_result: EstimateResult
    edge_result: EdgeResult


class SignalCandidate(BaseModel):
    """Candidate surviving signal filters, ready for risk evaluation."""

    market_id: str
    ticker: str | None = None
    title: str
    side: Literal["yes", "no"]
    price_cents: int = Field(ge=1, le=99)
    quantity: int = Field(gt=0)
    score: float
    parsed_contract: ParsedWeatherContract
    match_result: MatchResult
    estimate_result: EstimateResult
    edge_result: EdgeResult
    explainability: dict[str, Any] = Field(default_factory=dict)


class SignalRejection(BaseModel):
    """Structured rejection record with stage and reason codes."""

    market_id: str
    ticker: str | None = None
    title: str
    stage: str
    reason_code: str
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SignalSelectionResult(BaseModel):
    """Selected + rejected candidates with summary counts."""

    selected: list[SignalCandidate] = Field(default_factory=list)
    rejected: list[SignalRejection] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


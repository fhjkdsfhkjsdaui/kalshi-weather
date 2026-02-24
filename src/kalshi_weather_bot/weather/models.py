"""Typed models for normalized weather snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class WeatherForecastPeriod(BaseModel):
    """Normalized weather forecast period."""

    name: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    temperature: float | None = None
    temperature_unit: str | None = None
    wind_speed: str | None = None
    wind_direction: str | None = None
    short_forecast: str | None = None
    detailed_forecast: str | None = None
    probability_of_precipitation: float | None = None


class WeatherLocation(BaseModel):
    """Location metadata for a weather snapshot."""

    latitude: float | None = None
    longitude: float | None = None
    grid_id: str | None = None
    grid_x: int | None = None
    grid_y: int | None = None
    city: str | None = None
    state: str | None = None


class WeatherSnapshot(BaseModel):
    """Normalized provider snapshot for revision-history collection."""

    provider: str
    provider_version: str
    retrieval_timestamp: datetime
    forecast_type: Literal["daily", "hourly"]
    source_url: str
    points_url: str | None = None
    generated_timestamp: datetime | None = None
    updated_timestamp: datetime | None = None
    location: WeatherLocation
    periods: list[WeatherForecastPeriod] = Field(default_factory=list)
    raw_payload_path: str | None = None


class WeatherFetchResult(BaseModel):
    """Raw + normalized result returned by weather providers."""

    snapshot: WeatherSnapshot
    raw_forecast_payload: dict[str, Any]
    raw_points_payload: dict[str, Any] | None = None

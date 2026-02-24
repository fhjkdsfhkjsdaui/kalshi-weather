"""Shared typed models for Day 1 market discovery."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MarketSummary(BaseModel):
    """Normalized market representation used by Day 1 CLI output."""

    market_id: str = Field(description="Kalshi market identifier if available")
    ticker: str | None = Field(default=None, description="Ticker/symbol if available")
    title: str = Field(description="Human-readable market title")
    status: str | None = Field(default=None, description="Market status from API")
    close_time: datetime | None = Field(
        default=None, description="Market close time in UTC if present"
    )
    settlement_time: datetime | None = Field(
        default=None, description="Settlement time in UTC if present"
    )
    category: str | None = Field(default=None, description="Category/series tag if present")
    raw: dict[str, Any] = Field(description="Raw market payload for later parsing")

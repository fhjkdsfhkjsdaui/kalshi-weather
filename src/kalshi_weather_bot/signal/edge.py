"""Day 5 edge calculation module."""

from __future__ import annotations

import logging
from typing import Any

from ..config import Settings
from .models import EdgeResult, EstimateResult
from .pricing import parse_price_field


class EdgeCalculator:
    """Compute market edge from model estimate vs implied market probabilities."""

    _YES_PRICE_FIELDS = (
        "yes_ask",
        "yesAsk",
        "ask_yes",
        "last_price",
        "yes_bid",
        "yes_ask_dollars",
        "last_price_dollars",
    )
    _NO_PRICE_FIELDS = (
        "no_ask",
        "noAsk",
        "ask_no",
        "no_bid",
        "no_ask_dollars",
    )

    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("kalshi_weather_bot.signal.edge")

    def compute(self, market: dict[str, Any], estimate: EstimateResult) -> EdgeResult:
        """Compute YES/NO edge and a side recommendation after buffers."""
        if not estimate.available or estimate.estimated_probability is None:
            return EdgeResult(
                valid=False,
                decision_code="estimate_unavailable",
                reasons=["estimate_unavailable"],
                warnings=[],
                spread_buffer=self.settings.edge_spread_buffer,
                fee_buffer=self.settings.edge_fee_buffer,
            )

        warnings: list[str] = []
        reasons: list[str] = []
        yes_price, yes_field = self._extract_price(market, self._YES_PRICE_FIELDS)
        no_price, no_field = self._extract_price(market, self._NO_PRICE_FIELDS)

        if yes_price is None and no_price is None:
            return EdgeResult(
                valid=False,
                decision_code="missing_market_price",
                reasons=["missing_market_price"],
                warnings=warnings,
                spread_buffer=self.settings.edge_spread_buffer,
                fee_buffer=self.settings.edge_fee_buffer,
            )

        if yes_price is None and no_price is not None:
            yes_price = 100 - no_price
            yes_field = "derived_from_no_price"
            warnings.append("yes_price_derived_from_no_price")
        if no_price is None and yes_price is not None:
            no_price = 100 - yes_price
            no_field = "derived_from_yes_price"
            warnings.append("no_price_derived_from_yes_price")

        if yes_price is None or no_price is None:
            return EdgeResult(
                valid=False,
                decision_code="price_derivation_failed",
                reasons=["price_derivation_failed"],
                warnings=warnings,
                spread_buffer=self.settings.edge_spread_buffer,
                fee_buffer=self.settings.edge_fee_buffer,
            )
        if not (0 <= yes_price <= 100) or not (0 <= no_price <= 100):
            return EdgeResult(
                valid=False,
                decision_code="invalid_price_bounds",
                reasons=["invalid_price_bounds"],
                warnings=warnings,
                spread_buffer=self.settings.edge_spread_buffer,
                fee_buffer=self.settings.edge_fee_buffer,
            )
        if yes_price in {0, 100}:
            reasons.append("non_actionable_yes_price")
        if no_price in {0, 100}:
            reasons.append("non_actionable_no_price")
        if reasons:
            return EdgeResult(
                valid=False,
                decision_code=reasons[0],
                reasons=reasons,
                warnings=warnings,
                yes_price_cents=yes_price,
                no_price_cents=no_price,
                market_price_field_yes=yes_field,
                market_price_field_no=no_field,
                market_implied_probability=yes_price / 100.0,
                model_probability=estimate.estimated_probability,
                spread_buffer=self.settings.edge_spread_buffer,
                fee_buffer=self.settings.edge_fee_buffer,
            )

        model_prob = estimate.estimated_probability
        implied_yes = yes_price / 100.0
        implied_no = no_price / 100.0
        edge_yes = model_prob - implied_yes
        edge_no = (1.0 - model_prob) - implied_no

        total_buffer = self.settings.edge_spread_buffer + self.settings.edge_fee_buffer
        if edge_yes >= edge_no:
            best_side = "yes"
            best_price = yes_price
            best_edge = edge_yes
        else:
            best_side = "no"
            best_price = no_price
            best_edge = edge_no

        edge_after = best_edge - total_buffer
        if edge_after <= 0:
            return EdgeResult(
                valid=True,
                decision_code="insufficient_edge_after_buffers",
                reasons=["insufficient_edge_after_buffers"],
                warnings=warnings,
                yes_price_cents=yes_price,
                no_price_cents=no_price,
                market_price_field_yes=yes_field,
                market_price_field_no=no_field,
                market_implied_probability=implied_yes,
                model_probability=model_prob,
                edge_yes=round(edge_yes, 6),
                edge_no=round(edge_no, 6),
                recommended_side=None,
                recommended_price_cents=None,
                edge_after_buffers=round(edge_after, 6),
                spread_buffer=self.settings.edge_spread_buffer,
                fee_buffer=self.settings.edge_fee_buffer,
            )

        return EdgeResult(
            valid=True,
            decision_code="edge_computed",
            reasons=[],
            warnings=warnings,
            yes_price_cents=yes_price,
            no_price_cents=no_price,
            market_price_field_yes=yes_field,
            market_price_field_no=no_field,
            market_implied_probability=implied_yes,
            model_probability=model_prob,
            edge_yes=round(edge_yes, 6),
            edge_no=round(edge_no, 6),
            recommended_side=best_side,
            recommended_price_cents=best_price,
            edge_after_buffers=round(edge_after, 6),
            spread_buffer=self.settings.edge_spread_buffer,
            fee_buffer=self.settings.edge_fee_buffer,
        )

    def _extract_price(
        self,
        market: dict[str, Any],
        fields: tuple[str, ...],
    ) -> tuple[int | None, str | None]:
        for field in fields:
            if field not in market:
                continue
            price = parse_price_field(field, market.get(field))
            if price is None:
                continue
            return price, field
        return None, None

"""Day 5 market-to-weather matcher."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from ..contracts.models import ParsedWeatherContract
from ..weather.models import WeatherForecastPeriod, WeatherSnapshot
from .models import MatchResult


def time_window_overlap(
    start_a: datetime | None,
    end_a: datetime | None,
    start_b: datetime | None,
    end_b: datetime | None,
) -> bool:
    """Return True when two datetime windows overlap (inclusive endpoints)."""
    if start_a is None or start_b is None:
        return False
    a_end = end_a or (start_a + timedelta(hours=1))
    b_end = end_b or (start_b + timedelta(hours=1))
    a_start_utc = _as_utc(start_a)
    a_end_utc = _as_utc(a_end)
    b_start_utc = _as_utc(start_b)
    b_end_utc = _as_utc(b_end)
    return a_start_utc <= b_end_utc and b_start_utc <= a_end_utc


class WeatherMarketMatcher:
    """Align parsed weather contracts to relevant weather forecast periods."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("kalshi_weather_bot.signal.matcher")

    def match_contract(
        self,
        contract: ParsedWeatherContract,
        snapshot: WeatherSnapshot,
    ) -> MatchResult:
        """Match one parsed contract to weather periods with confidence scoring."""
        reasons: list[str] = []
        warnings: list[str] = []
        assumptions: list[str] = []
        time_alignment: dict[str, Any] = {}
        unit_meta: dict[str, Any] = {}

        if contract.weather_dimension is None:
            reasons.append("unsupported_or_missing_weather_dimension")
            return MatchResult(
                matched=False,
                decision_code="unsupported_metric_dimension",
                reasons=reasons,
                warnings=warnings,
                assumptions=assumptions,
                selected_period_indices=[],
                selected_period_count=0,
                selected_periods=[],
                time_alignment=time_alignment,
                unit_normalization=unit_meta,
                matcher_confidence=0.0,
            )

        if not snapshot.periods:
            reasons.append("no_forecast_periods")
            return MatchResult(
                matched=False,
                decision_code="no_forecast_periods",
                reasons=reasons,
                warnings=warnings,
                assumptions=assumptions,
                selected_period_indices=[],
                selected_period_count=0,
                selected_periods=[],
                time_alignment=time_alignment,
                unit_normalization=unit_meta,
                matcher_confidence=0.0,
            )

        location_match, location_confidence = self._location_match(contract, snapshot)
        if not location_match:
            reasons.append("location_mismatch")
            return MatchResult(
                matched=False,
                decision_code="location_mismatch",
                reasons=reasons,
                warnings=warnings,
                assumptions=assumptions,
                selected_period_indices=[],
                selected_period_count=0,
                selected_periods=[],
                time_alignment=time_alignment,
                unit_normalization=unit_meta,
                matcher_confidence=0.0,
            )
        if location_confidence < 1.0:
            assumptions.append(
                "ASSUMPTION: location alignment is partial (city/state incomplete)."
            )

        contract_start, contract_end = self._contract_window(contract, assumptions)
        if contract_start is None and contract_end is None:
            time_alignment["mode"] = "all_periods_due_to_missing_contract_window"
            assumptions.append(
                "ASSUMPTION: contract time window missing; "
                "evaluating all available forecast periods."
            )
            candidate_indices = list(range(len(snapshot.periods)))
        else:
            candidate_indices = [
                idx
                for idx, period in enumerate(snapshot.periods)
                if time_window_overlap(
                    contract_start,
                    contract_end,
                    period.start_time,
                    period.end_time,
                )
            ]
            time_alignment["mode"] = "contract_window_overlap"
            time_alignment["contract_start_utc"] = (
                _as_utc(contract_start).isoformat() if contract_start else None
            )
            time_alignment["contract_end_utc"] = (
                _as_utc(contract_end).isoformat() if contract_end else None
            )
            if not candidate_indices:
                reasons.append("no_time_window_overlap")
                return MatchResult(
                    matched=False,
                    decision_code="no_time_window_overlap",
                    reasons=reasons,
                    warnings=warnings,
                    assumptions=assumptions,
                    selected_period_indices=[],
                    selected_period_count=0,
                    selected_periods=[],
                    time_alignment=time_alignment,
                    unit_normalization=unit_meta,
                    matcher_confidence=0.1,
                )

        selected_indices = self._select_metric_periods(
            contract=contract,
            periods=snapshot.periods,
            candidate_indices=candidate_indices,
        )
        if not selected_indices:
            reasons.append("metric_data_unavailable")
            return MatchResult(
                matched=False,
                decision_code="metric_data_unavailable",
                reasons=reasons,
                warnings=warnings,
                assumptions=assumptions,
                selected_period_indices=[],
                selected_period_count=0,
                selected_periods=[],
                time_alignment=time_alignment,
                unit_normalization=unit_meta,
                matcher_confidence=0.2,
            )

        selected_periods = [snapshot.periods[idx] for idx in selected_indices]
        unit_meta["contract_threshold_unit"] = contract.threshold_unit
        unit_meta["forecast_temperature_units"] = sorted(
            {
                period.temperature_unit
                for period in selected_periods
                if period.temperature_unit is not None
            }
        )
        unit_meta["forecast_type"] = snapshot.forecast_type
        if contract.location_normalized and (
            contract.location_normalized.city or contract.location_normalized.state
        ):
            assumptions.append(
                "ASSUMPTION: contract settlement timezone interpreted in UTC for matching."
            )

        confidence = self._match_confidence(
            location_confidence=location_confidence,
            selected_count=len(selected_periods),
            used_explicit_window=contract_start is not None or contract_end is not None,
            assumption_count=len(assumptions),
        )
        return MatchResult(
            matched=True,
            decision_code="matched",
            reasons=reasons,
            warnings=warnings,
            assumptions=assumptions,
            selected_period_indices=selected_indices,
            selected_period_count=len(selected_periods),
            selected_periods=selected_periods,
            time_alignment=time_alignment,
            unit_normalization=unit_meta,
            matcher_confidence=confidence,
        )

    def _contract_window(
        self,
        contract: ParsedWeatherContract,
        assumptions: list[str],
    ) -> tuple[datetime | None, datetime | None]:
        start = contract.contract_start_time
        end = contract.contract_end_time or contract.resolution_time
        if start and end and _as_utc(end) < _as_utc(start):
            assumptions.append(
                "ASSUMPTION: contract window start/end were inverted; window endpoints swapped."
            )
            return end, start
        if start is None and end is not None:
            assumptions.append(
                "ASSUMPTION: contract start missing; using 24h lookback from contract end."
            )
            return end - timedelta(hours=24), end
        return start, end

    def _location_match(
        self,
        contract: ParsedWeatherContract,
        snapshot: WeatherSnapshot,
    ) -> tuple[bool, float]:
        contract_city = (
            (contract.location_normalized.city or "").strip().lower()
            if contract.location_normalized
            else ""
        )
        contract_state = (
            (contract.location_normalized.state or "").strip().lower()
            if contract.location_normalized
            else ""
        )
        snap_city = (snapshot.location.city or "").strip().lower()
        snap_state = (snapshot.location.state or "").strip().lower()

        if not contract_city and not contract_state:
            return True, 0.6
        if contract_state and snap_state and contract_state != snap_state:
            return False, 0.0
        if contract_city and snap_city:
            if _normalize_place(contract_city) != _normalize_place(snap_city):
                return False, 0.0
            if contract_state and snap_state and contract_state == snap_state:
                return True, 1.0
            return True, 0.85
        if contract_state and snap_state and contract_state == snap_state:
            return True, 0.7
        return True, 0.65

    def _select_metric_periods(
        self,
        contract: ParsedWeatherContract,
        periods: list[WeatherForecastPeriod],
        candidate_indices: list[int],
    ) -> list[int]:
        dim = contract.weather_dimension
        if dim == "temperature":
            return [idx for idx in candidate_indices if periods[idx].temperature is not None]
        if dim == "precipitation":
            indices = [
                idx
                for idx in candidate_indices
                if periods[idx].probability_of_precipitation is not None
            ]
            if indices:
                return indices
            return [
                idx
                for idx in candidate_indices
                if _contains_precip_keywords(
                    periods[idx].short_forecast,
                    periods[idx].detailed_forecast,
                )
            ]
        if dim == "wind":
            return [
                idx
                for idx in candidate_indices
                if _extract_first_float(periods[idx].wind_speed) is not None
            ]
        if dim == "snowfall":
            return [
                idx
                for idx in candidate_indices
                if _contains_snow_keywords(
                    periods[idx].short_forecast,
                    periods[idx].detailed_forecast,
                )
            ]
        return []

    @staticmethod
    def _match_confidence(
        *,
        location_confidence: float,
        selected_count: int,
        used_explicit_window: bool,
        assumption_count: int,
    ) -> float:
        confidence = 0.35
        confidence += 0.3 * max(0.0, min(location_confidence, 1.0))
        confidence += 0.2 if used_explicit_window else 0.08
        confidence += min(0.2, 0.05 * selected_count)
        confidence -= min(0.25, 0.05 * assumption_count)
        return round(max(0.0, min(confidence, 1.0)), 3)


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _normalize_place(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def _contains_precip_keywords(short_text: str | None, detail_text: str | None) -> bool:
    blob = " ".join(part for part in [short_text, detail_text] if part).lower()
    return bool(re.search(r"\b(rain|shower|precip|storm|drizzle)\b", blob))


def _contains_snow_keywords(short_text: str | None, detail_text: str | None) -> bool:
    blob = " ".join(part for part in [short_text, detail_text] if part).lower()
    return bool(re.search(r"\b(snow|sleet|flurr(?:y|ies)|blizzard)\b", blob))


def _extract_first_float(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

"""Day 5 transparent probability estimator (v1, deterministic heuristics)."""

from __future__ import annotations

import logging
import math
import re

from ..contracts.models import ParsedWeatherContract, ThresholdOperator
from .models import EstimateResult, MatchResult


class WeatherProbabilityEstimator:
    """Estimate outcome probability from matched forecast periods."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("kalshi_weather_bot.signal.estimator")

    def estimate(self, contract: ParsedWeatherContract, match: MatchResult) -> EstimateResult:
        """Return deterministic estimate object for the parsed contract."""
        if not match.matched:
            return EstimateResult(
                available=False,
                decision_code="match_failed",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="none",
                assumptions=[],
                missing_data_flags=["unmatched_weather_data"],
                supporting_values={},
                reasons=["unmatched_weather_data"],
                warnings=[],
            )

        if contract.weather_dimension == "temperature":
            return self._estimate_temperature(contract, match)
        if contract.weather_dimension == "precipitation":
            return self._estimate_precipitation(contract, match)
        if contract.weather_dimension == "wind":
            return self._estimate_wind(contract, match)

        return EstimateResult(
            available=False,
            decision_code="unsupported_metric",
            estimated_probability=None,
            model_confidence=0.0,
            estimate_method="unsupported",
            assumptions=[],
            missing_data_flags=["unsupported_metric_type"],
            supporting_values={},
            reasons=["unsupported_metric_type"],
            warnings=[],
        )

    def _estimate_temperature(
        self,
        contract: ParsedWeatherContract,
        match: MatchResult,
    ) -> EstimateResult:
        values: list[float] = []
        assumptions = list(match.assumptions)
        missing: list[str] = []
        warnings: list[str] = []
        reasons: list[str] = []

        target_unit = contract.threshold_unit or self._first_temperature_unit(match) or "F"
        for period in match.selected_periods:
            if period.temperature is None:
                continue
            source_unit = period.temperature_unit or target_unit
            converted = _convert_temperature(
                period.temperature,
                from_unit=source_unit,
                to_unit=target_unit,
            )
            if converted is None:
                warnings.append(f"temperature_unit_conversion_failed:{source_unit}->{target_unit}")
                continue
            values.append(converted)

        if not values:
            missing.append("temperature_values_missing")
            return EstimateResult(
                available=False,
                decision_code="temperature_data_missing",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="temperature_threshold_v1",
                assumptions=assumptions,
                missing_data_flags=missing,
                supporting_values={},
                reasons=["temperature_data_missing"],
                warnings=warnings,
            )

        threshold_ok = _has_threshold(contract)
        if not threshold_ok:
            missing.append("threshold_not_parseable")
            return EstimateResult(
                available=False,
                decision_code="threshold_not_parseable",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="temperature_threshold_v1",
                assumptions=assumptions,
                missing_data_flags=missing,
                supporting_values={"sample_size": len(values)},
                reasons=["threshold_not_parseable"],
                warnings=warnings,
            )

        met = [value for value in values if _threshold_met(value, contract)]
        probability = len(met) / len(values)
        confidence = 0.55 + min(0.25, 0.05 * len(values))
        confidence -= min(0.25, 0.05 * len(assumptions))
        if len(values) == 1:
            confidence -= 0.1
            assumptions.append("ASSUMPTION: single forecast period used for threshold estimate.")
        return EstimateResult(
            available=True,
            decision_code="estimated",
            estimated_probability=max(0.0, min(probability, 1.0)),
            model_confidence=round(max(0.0, min(confidence, 1.0)), 3),
            estimate_method="temperature_threshold_v1",
            assumptions=assumptions,
            missing_data_flags=missing,
            supporting_values={
                "sample_size": len(values),
                "threshold_operator": contract.threshold_operator,
                "threshold_value": contract.threshold_value,
                "threshold_low": contract.threshold_low,
                "threshold_high": contract.threshold_high,
                "threshold_unit": target_unit,
                "values_min": min(values),
                "values_max": max(values),
                "met_count": len(met),
            },
            reasons=reasons,
            warnings=warnings,
        )

    def _estimate_precipitation(
        self,
        contract: ParsedWeatherContract,
        match: MatchResult,
    ) -> EstimateResult:
        assumptions = list(match.assumptions)
        missing: list[str] = []
        warnings: list[str] = []

        # Conservative Day 5 behavior: only support probability/occurrence style
        # precipitation estimates from NWS PoP fields, not accumulation amounts.
        if contract.threshold_unit and contract.threshold_unit.lower() in {"in", "mm", "cm"}:
            return EstimateResult(
                available=False,
                decision_code="precip_amount_not_supported",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="precip_occurrence_proxy_v1",
                assumptions=assumptions,
                missing_data_flags=["precip_amount_data_unavailable"],
                supporting_values={},
                reasons=["precip_amount_not_supported"],
                warnings=warnings,
            )

        pop_probs = [
            max(0.0, min(1.0, period.probability_of_precipitation / 100.0))
            for period in match.selected_periods
            if period.probability_of_precipitation is not None
        ]
        if not pop_probs:
            missing.append("probability_of_precipitation_missing")
            return EstimateResult(
                available=False,
                decision_code="precip_probability_missing",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="precip_occurrence_proxy_v1",
                assumptions=assumptions,
                missing_data_flags=missing,
                supporting_values={},
                reasons=["precip_probability_missing"],
                warnings=warnings,
            )

        assumptions.append(
            "ASSUMPTION: NWS PoP used as proxy for contract precipitation condition."
        )
        p_any = 1.0 - math.prod(1.0 - p for p in pop_probs)
        probability = p_any

        # If threshold appears to be a percent value (e.g. ">= 40"), use frequency
        # of periods meeting that threshold as a conservative alternative.
        if (
            contract.threshold_value is not None
            and contract.threshold_value > 1
            and contract.threshold_value <= 100
        ):
            threshold_p = contract.threshold_value / 100.0
            met = [
                p
                for p in pop_probs
                if _prob_threshold_met(p, contract.threshold_operator, threshold_p)
            ]
            if met:
                probability = len(met) / len(pop_probs)
                assumptions.append(
                    "ASSUMPTION: interpreted precip threshold as period-level PoP percentage."
                )

        confidence = 0.42 + min(0.2, 0.04 * len(pop_probs))
        confidence -= min(0.2, 0.04 * len(assumptions))
        return EstimateResult(
            available=True,
            decision_code="estimated",
            estimated_probability=max(0.0, min(probability, 1.0)),
            model_confidence=round(max(0.0, min(confidence, 1.0)), 3),
            estimate_method="precip_occurrence_proxy_v1",
            assumptions=assumptions,
            missing_data_flags=missing,
            supporting_values={
                "period_count": len(pop_probs),
                "pop_mean": round(sum(pop_probs) / len(pop_probs), 4),
                "pop_max": round(max(pop_probs), 4),
                "proxy_probability_any_precip": round(p_any, 4),
            },
            reasons=[],
            warnings=warnings,
        )

    def _estimate_wind(self, contract: ParsedWeatherContract, match: MatchResult) -> EstimateResult:
        assumptions = list(match.assumptions)
        missing: list[str] = []
        warnings: list[str] = []

        target_unit = contract.threshold_unit or "mph"
        values: list[float] = []
        for period in match.selected_periods:
            value, source_unit = _extract_wind_speed(period.wind_speed)
            if value is None:
                continue
            source = source_unit or target_unit
            converted = _convert_speed(value, from_unit=source, to_unit=target_unit)
            if converted is None:
                warnings.append(f"wind_unit_conversion_failed:{source}->{target_unit}")
                continue
            values.append(converted)

        if not values:
            return EstimateResult(
                available=False,
                decision_code="wind_data_missing",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="wind_threshold_v1",
                assumptions=assumptions,
                missing_data_flags=["wind_speed_missing"],
                supporting_values={},
                reasons=["wind_speed_missing"],
                warnings=warnings,
            )

        if not _has_threshold(contract):
            return EstimateResult(
                available=False,
                decision_code="threshold_not_parseable",
                estimated_probability=None,
                model_confidence=0.0,
                estimate_method="wind_threshold_v1",
                assumptions=assumptions,
                missing_data_flags=["threshold_not_parseable"],
                supporting_values={"sample_size": len(values)},
                reasons=["threshold_not_parseable"],
                warnings=warnings,
            )

        met = [value for value in values if _threshold_met(value, contract)]
        probability = len(met) / len(values)
        assumptions.append("ASSUMPTION: wind string parsed to representative numeric speed.")
        confidence = 0.5 + min(0.2, 0.04 * len(values))
        confidence -= min(0.2, 0.04 * len(assumptions))
        return EstimateResult(
            available=True,
            decision_code="estimated",
            estimated_probability=max(0.0, min(probability, 1.0)),
            model_confidence=round(max(0.0, min(confidence, 1.0)), 3),
            estimate_method="wind_threshold_v1",
            assumptions=assumptions,
            missing_data_flags=missing,
            supporting_values={
                "sample_size": len(values),
                "threshold_operator": contract.threshold_operator,
                "threshold_value": contract.threshold_value,
                "threshold_low": contract.threshold_low,
                "threshold_high": contract.threshold_high,
                "threshold_unit": target_unit,
                "values_min": min(values),
                "values_max": max(values),
                "met_count": len(met),
            },
            reasons=[],
            warnings=warnings,
        )

    @staticmethod
    def _first_temperature_unit(match: MatchResult) -> str | None:
        for period in match.selected_periods:
            if period.temperature_unit:
                return period.temperature_unit
        return None


def _has_threshold(contract: ParsedWeatherContract) -> bool:
    operator = contract.threshold_operator
    if operator is None:
        return False
    if operator in {">", ">=", "<", "<=", "exact"}:
        return contract.threshold_value is not None
    if operator in {"between", "range"}:
        return contract.threshold_low is not None and contract.threshold_high is not None
    return False


def _threshold_met(value: float, contract: ParsedWeatherContract) -> bool:
    op = contract.threshold_operator
    if op in {">", ">=", "<", "<=", "exact"} and contract.threshold_value is not None:
        threshold = contract.threshold_value
        return _compare_scalar(value, op, threshold)
    if op in {"between", "range"}:
        if contract.threshold_low is None or contract.threshold_high is None:
            return False
        low = min(contract.threshold_low, contract.threshold_high)
        high = max(contract.threshold_low, contract.threshold_high)
        return low <= value <= high
    return False


def _compare_scalar(value: float, operator: ThresholdOperator, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "exact":
        return math.isclose(value, threshold, abs_tol=1e-9)
    return False


def _prob_threshold_met(
    probability: float,
    operator: ThresholdOperator | None,
    threshold_probability: float,
) -> bool:
    if operator is None:
        return probability >= threshold_probability
    if operator in {">", ">=", "<", "<=", "exact"}:
        return _compare_scalar(probability, operator, threshold_probability)
    return probability >= threshold_probability


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float | None:
    from_u = from_unit.strip().upper()
    to_u = to_unit.strip().upper()
    if from_u == to_u:
        return value
    if from_u == "F" and to_u == "C":
        return (value - 32.0) * 5.0 / 9.0
    if from_u == "C" and to_u == "F":
        return value * 9.0 / 5.0 + 32.0
    return None


def _extract_wind_speed(raw: str | None) -> tuple[float | None, str | None]:
    if raw is None:
        return None, None
    numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    if not numbers:
        return None, None
    # Use max of ranges like "8 to 15 mph" for conservative threshold checks.
    speed = max(numbers)
    unit_match = re.search(r"\b(mph|kph|kt|kts|knots?)\b", raw.lower())
    unit = unit_match.group(1) if unit_match else None
    return speed, unit


def _convert_speed(value: float, from_unit: str, to_unit: str) -> float | None:
    from_u = from_unit.strip().lower()
    to_u = to_unit.strip().lower()
    if from_u == to_u:
        return value

    mph_per_unit = {
        "mph": 1.0,
        "kph": 0.621371,
        "kt": 1.15078,
        "kts": 1.15078,
        "knot": 1.15078,
        "knots": 1.15078,
    }
    if from_u not in mph_per_unit or to_u not in mph_per_unit:
        return None
    mph_value = value * mph_per_unit[from_u]
    return mph_value / mph_per_unit[to_u]

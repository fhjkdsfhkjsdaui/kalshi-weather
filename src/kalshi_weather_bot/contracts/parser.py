"""Layered parser for weather-focused Kalshi contract metadata."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .models import (
    ContractParseSummary,
    NormalizedLocation,
    ParsedWeatherContract,
    ParseStatus,
    ThresholdOperator,
    WeatherDimension,
)

_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
_UNIT_PATTERN = (
    r"(?:°?\s*[fc]|degrees?\s*(?:fahrenheit|celsius|[fc])?|"
    r"inches?|inch|in\.?|mm|cm|mph|kph|knots?|kts?)"
)

_BETWEEN_RE = re.compile(
    rf"\bbetween\s+(?P<low>{_NUMBER_PATTERN})\s*(?P<u1>{_UNIT_PATTERN})?"
    rf"\s+(?:and|to)\s+(?P<high>{_NUMBER_PATTERN})\s*(?P<u2>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_AT_LEAST_RE = re.compile(
    rf"\b(?:at\s+least|no\s+less\s+than|minimum(?:\s+of)?)\s+"
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_AT_MOST_RE = re.compile(
    rf"\b(?:at\s+most|no\s+more\s+than|maximum(?:\s+of)?)\s+"
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_ABOVE_RE = re.compile(
    rf"\b(?:above|over|greater\s+than|more\s+than)\s+"
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_BELOW_RE = re.compile(
    rf"\b(?:below|under|less\s+than)\s+"
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_EXACT_RE = re.compile(
    rf"\b(?:exactly|equal\s+to)\s+"
    rf"(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(
    rf"(?P<op>>=|<=|>|<)\s*(?P<value>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})?",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    rf"\b(?P<low>{_NUMBER_PATTERN})\s*(?P<u1>{_UNIT_PATTERN})?\s*"
    rf"(?:to|-)\s*(?P<high>{_NUMBER_PATTERN})\s*(?P<u2>{_UNIT_PATTERN})?\b",
    re.IGNORECASE,
)

_LOCATION_PREP_RE = re.compile(
    r"\b(?:in|for|at)\s+(?P<loc>[A-Z][A-Za-z .'-]{1,60}(?:,\s*[A-Z]{2})?)"
    r"(?=\s+(?:be|is|are|will|to|on|by|between|above|below|over|under|at|>=|<=|>|<)|"
    r"[?!.]|$)"
)
_LOCATION_PREFIX_RE = re.compile(
    r"^(?P<loc>[A-Z][A-Za-z .'-]{1,60}?)\s+"
    r"(?=(?:high|low|temperature|temp|rain|rainfall|snow|snowfall|wind)\b)"
)
_STATE_CODE_RE = re.compile(r"^[A-Z]{2}$")


@dataclass
class _ThresholdExtraction:
    operator: ThresholdOperator | None = None
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str | None = None

    @property
    def has_value(self) -> bool:
        return self.value is not None or (self.low is not None and self.high is not None)


class KalshiWeatherContractParser:
    """Parse Kalshi market records into normalized weather contract objects."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("kalshi_weather_bot.contracts")

    def parse_markets(self, markets: Iterable[dict[str, Any]]) -> list[ParsedWeatherContract]:
        """Parse an iterable of raw market dictionaries."""
        return [self.parse_market(market) for market in markets]

    def parse_market(self, market: dict[str, Any]) -> ParsedWeatherContract:
        """Parse one raw market record using layered extraction + scoring."""
        title = self._first_string(market, ("title", "market_title", "name")) or ""
        subtitle = self._first_string(market, ("subtitle", "market_subtitle", "event_subtitle"))
        rules_primary = self._first_string(market, ("rules_primary", "rules", "rulesPrimary"))
        rules_secondary = self._first_string(market, ("rules_secondary", "rulesSecondary"))

        search_text = " ".join(
            part for part in [title, subtitle, rules_primary, rules_secondary] if part
        )
        weather_candidate = self._is_weather_candidate(search_text, market)

        weather_dimension: WeatherDimension | None = None
        metric_subtype: str | None = None
        threshold = _ThresholdExtraction()
        location_raw: str | None = None
        location_normalized: NormalizedLocation | None = None
        rejection_reasons: list[str] = []

        if weather_candidate:
            weather_dimension, metric_subtype = self._extract_dimension(search_text)
            threshold = self._extract_threshold(search_text)
            location_raw = self._extract_location(title, subtitle)
            location_normalized = self._normalize_location(location_raw)

            if weather_dimension is None:
                rejection_reasons.append("missing_dimension")
            if threshold.operator is None:
                rejection_reasons.append("missing_threshold_operator")
            if not threshold.has_value:
                rejection_reasons.append("missing_threshold_value")
        else:
            rejection_reasons.append("non_weather_market")

        contract_start_time = self._first_datetime(market, ("open_time", "created_time"))
        contract_end_time = self._first_datetime(
            market,
            ("close_time", "expiration_time", "expected_expiration_time", "latest_expiration_time"),
        )
        resolution_time = self._first_datetime(
            market,
            ("settlement_time", "settles_at", "expiry_time", "expiration_time"),
        )

        confidence = self._score_confidence(
            weather_candidate=weather_candidate,
            weather_dimension=weather_dimension,
            metric_subtype=metric_subtype,
            threshold=threshold,
            location_raw=location_raw,
            contract_end_time=contract_end_time,
            resolution_time=resolution_time,
        )
        status = self._determine_status(
            weather_candidate=weather_candidate,
            weather_dimension=weather_dimension,
            threshold=threshold,
            confidence=confidence,
        )
        if status == "parsed":
            rejection_reasons = []

        provider_market_id = self._first_string(market, ("market_id", "id", "ticker")) or "unknown"
        return ParsedWeatherContract(
            provider_market_id=provider_market_id,
            ticker=self._first_string(market, ("ticker", "symbol")),
            event_id=self._first_string(market, ("event_ticker", "event_id")),
            raw_title=title or "(untitled market)",
            raw_subtitle=subtitle,
            raw_rules_primary=rules_primary,
            raw_rules_secondary=rules_secondary,
            weather_candidate=weather_candidate,
            weather_dimension=weather_dimension,
            metric_subtype=metric_subtype,
            location_raw=location_raw,
            location_normalized=location_normalized,
            threshold_operator=threshold.operator,
            threshold_value=threshold.value,
            threshold_low=threshold.low,
            threshold_high=threshold.high,
            threshold_unit=threshold.unit,
            contract_start_time=contract_start_time,
            contract_end_time=contract_end_time,
            resolution_time=resolution_time,
            yes_side_semantics=self._first_string(market, ("yes_sub_title", "yes_subtitle")),
            no_side_semantics=self._first_string(market, ("no_sub_title", "no_subtitle")),
            parse_confidence=confidence,
            parse_status=status,
            rejection_reasons=rejection_reasons,
        )

    def _is_weather_candidate(self, search_text: str, market: dict[str, Any]) -> bool:
        tags = market.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip().lower() == "weather":
                    return True

        category = self._first_string(market, ("category", "series", "event_category"))
        if category and "weather" in category.lower():
            return True

        lowered = search_text.lower()
        weather_patterns = (
            r"\btemperature\b",
            r"\btemp\b",
            r"\bhigh\s+temp(?:erature)?\b",
            r"\blow\s+temp(?:erature)?\b",
            r"\bdegrees?\b",
            r"\brain(?:fall)?\b",
            r"\bprecip(?:itation)?\b",
            r"\bsnow(?:fall)?\b",
            r"\bwind\s+(?:speed|gusts?)\b",
            r"\bgusts?\b",
            r"\bweather\b",
        )
        return any(re.search(pattern, lowered) for pattern in weather_patterns)

    def _extract_dimension(self, search_text: str) -> tuple[WeatherDimension | None, str | None]:
        lowered = search_text.lower()
        if re.search(r"\bhigh\s+temp(?:erature)?\b", lowered):
            return "temperature", "high_temp"
        if re.search(r"\blow\s+temp(?:erature)?\b", lowered):
            return "temperature", "low_temp"
        if re.search(r"\btemp(?:erature)?\b|\bdegrees?\b|°\s*[fc]", lowered):
            return "temperature", "temperature"
        if re.search(r"\bsnow(?:fall)?\b", lowered):
            return "snowfall", "snowfall_accumulation"
        if re.search(r"\brain(?:fall)?\b|\bprecip(?:itation)?\b", lowered):
            return "precipitation", "total_precipitation"
        if re.search(r"\bwind\s+gusts?\b|\bgusts?\b", lowered):
            return "wind", "wind_gust"
        if re.search(r"\bwind\s+speed\b|\bwindspeed\b|\bwind\b", lowered):
            return "wind", "wind_speed"
        # Bare "weather" keyword is too vague to assign a dimension.
        # It will still be flagged as weather_candidate but dimension=None
        # forces the contract through the rejection path, preventing
        # overconfident parsing of ambiguous titles.
        return None, None

    def _extract_threshold(self, search_text: str) -> _ThresholdExtraction:
        lowered = search_text.lower()

        match = _BETWEEN_RE.search(lowered)
        if match:
            low = self._to_float(match.group("low"))
            high = self._to_float(match.group("high"))
            unit = self._normalize_unit(match.group("u1") or match.group("u2"))
            if low is not None and high is not None:
                if high < low:
                    low, high = high, low
                return _ThresholdExtraction(
                    operator="between",
                    low=low,
                    high=high,
                    unit=unit,
                )

        match = _AT_LEAST_RE.search(lowered)
        if match:
            value = self._to_float(match.group("value"))
            if value is not None:
                return _ThresholdExtraction(
                    operator=">=",
                    value=value,
                    unit=self._normalize_unit(match.group("unit")),
                )

        match = _AT_MOST_RE.search(lowered)
        if match:
            value = self._to_float(match.group("value"))
            if value is not None:
                return _ThresholdExtraction(
                    operator="<=",
                    value=value,
                    unit=self._normalize_unit(match.group("unit")),
                )

        match = _ABOVE_RE.search(lowered)
        if match:
            value = self._to_float(match.group("value"))
            if value is not None:
                return _ThresholdExtraction(
                    operator=">",
                    value=value,
                    unit=self._normalize_unit(match.group("unit")),
                )

        match = _BELOW_RE.search(lowered)
        if match:
            value = self._to_float(match.group("value"))
            if value is not None:
                return _ThresholdExtraction(
                    operator="<",
                    value=value,
                    unit=self._normalize_unit(match.group("unit")),
                )

        match = _EXACT_RE.search(lowered)
        if match:
            value = self._to_float(match.group("value"))
            if value is not None:
                return _ThresholdExtraction(
                    operator="exact",
                    value=value,
                    unit=self._normalize_unit(match.group("unit")),
                )

        match = _SYMBOL_RE.search(lowered)
        if match:
            value = self._to_float(match.group("value"))
            op = match.group("op")
            if value is not None and op in {">", ">=", "<", "<="}:
                return _ThresholdExtraction(
                    operator=op,
                    value=value,
                    unit=self._normalize_unit(match.group("unit")),
                )

        match = _RANGE_RE.search(lowered)
        if match:
            low = self._to_float(match.group("low"))
            high = self._to_float(match.group("high"))
            unit = self._normalize_unit(match.group("u1") or match.group("u2"))
            if low is not None and high is not None and unit:
                if high < low:
                    low, high = high, low
                return _ThresholdExtraction(operator="range", low=low, high=high, unit=unit)

        return _ThresholdExtraction()

    # Trailing verbs/prepositions that can leak into greedy location captures.
    _LOCATION_STOP_WORDS = frozenset({
        "be", "is", "are", "will", "to", "on", "by", "at", "have", "has",
    })

    def _extract_location(self, title: str, subtitle: str | None) -> str | None:
        for text in (title, subtitle):
            if not text:
                continue
            match = _LOCATION_PREP_RE.search(text)
            if match:
                return self._clean_location(match.group("loc"))
            match = _LOCATION_PREFIX_RE.search(text)
            if match:
                return self._clean_location(match.group("loc"))
        return None

    @classmethod
    def _clean_location(cls, raw: str) -> str:
        """Strip trailing stop-words and punctuation from a location capture."""
        text = raw.strip(" .")
        # The greedy regex may swallow trailing verbs like "be" in "Seattle be".
        words = text.split()
        while words and words[-1].lower() in cls._LOCATION_STOP_WORDS:
            words.pop()
        return " ".join(words) if words else text

    def _normalize_location(self, location_raw: str | None) -> NormalizedLocation | None:
        if not location_raw:
            return None
        text = location_raw.strip()
        if not text:
            return None

        if "," in text:
            city_part, state_part = [chunk.strip() for chunk in text.rsplit(",", 1)]
            if city_part and _STATE_CODE_RE.match(state_part):
                return NormalizedLocation(city=city_part, state=state_part)

        split_state = re.match(r"^(?P<city>.+?)\s+(?P<state>[A-Z]{2})$", text)
        if split_state:
            return NormalizedLocation(
                city=split_state.group("city").strip(),
                state=split_state.group("state").strip(),
            )
        return NormalizedLocation(city=text, state=None)

    def _score_confidence(
        self,
        *,
        weather_candidate: bool,
        weather_dimension: WeatherDimension | None,
        metric_subtype: str | None,
        threshold: _ThresholdExtraction,
        location_raw: str | None,
        contract_end_time: datetime | None,
        resolution_time: datetime | None,
    ) -> float:
        if not weather_candidate:
            return 0.0

        score = 0.25
        if weather_dimension is not None:
            score += 0.20
        if metric_subtype:
            score += 0.10
        if threshold.operator is not None:
            score += 0.20
        if threshold.has_value:
            score += 0.10
        if threshold.unit is not None:
            score += 0.05
        if location_raw:
            score += 0.05
        if contract_end_time is not None or resolution_time is not None:
            score += 0.05
        return round(min(score, 1.0), 3)

    @staticmethod
    def _determine_status(
        *,
        weather_candidate: bool,
        weather_dimension: WeatherDimension | None,
        threshold: _ThresholdExtraction,
        confidence: float,
    ) -> ParseStatus:
        if not weather_candidate:
            return "unsupported"
        if weather_dimension is None:
            return "rejected"
        if threshold.operator is None or not threshold.has_value:
            return "rejected"
        if confidence >= 0.75:
            return "parsed"
        return "ambiguous"

    @staticmethod
    def _normalize_unit(unit: str | None) -> str | None:
        if not unit:
            return None
        lowered = unit.strip().lower().replace(" ", "")
        if lowered in {"f", "°f"} or "fahrenheit" in lowered:
            return "F"
        if lowered in {"c", "°c"} or "celsius" in lowered:
            return "C"
        if lowered.startswith("degree"):
            return "deg"
        if lowered in {"inch", "inches", "in", "in."}:
            return "in"
        if lowered == "mm":
            return "mm"
        if lowered == "cm":
            return "cm"
        if lowered == "mph":
            return "mph"
        if lowered == "kph":
            return "kph"
        if lowered in {"knot", "knots", "kt", "kts"}:
            return "kt"
        return unit.strip()

    @staticmethod
    def _to_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _first_datetime(payload: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
        for key in keys:
            parsed = KalshiWeatherContractParser._parse_datetime(payload.get(key))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except (OSError, ValueError):
                return None
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return None
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return None


def summarize_parse_results(results: list[ParsedWeatherContract]) -> ContractParseSummary:
    """Build aggregate parser audit summary from parsed records."""
    reason_counter: Counter[str] = Counter()
    for result in results:
        reason_counter.update(result.rejection_reasons)

    status_counts: Counter[str] = Counter(result.parse_status for result in results)
    return ContractParseSummary(
        total_markets_scanned=len(results),
        weather_candidates=sum(1 for result in results if result.weather_candidate),
        parsed=status_counts.get("parsed", 0),
        ambiguous=status_counts.get("ambiguous", 0),
        unsupported=status_counts.get("unsupported", 0),
        rejected=status_counts.get("rejected", 0),
        top_rejection_reasons=dict(reason_counter.most_common(8)),
    )

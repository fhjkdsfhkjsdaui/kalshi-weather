"""Day 2 CLI: fetch NWS forecast snapshot and journal raw/normalized data."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC

from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .exceptions import ConfigError, JournalError, WeatherProviderError
from .journal import JournalWriter
from .log_setup import setup_logger
from .weather.models import WeatherSnapshot
from .weather.nws import NWSWeatherProvider


def parse_args() -> argparse.Namespace:
    """Parse Day 2 weather CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch and journal NWS weather forecast snapshot (Day 2)."
    )
    parser.add_argument("--lat", type=float, default=None, help="Latitude for points lookup.")
    parser.add_argument("--lon", type=float, default=None, help="Longitude for points lookup.")
    parser.add_argument("--url", type=str, default=None, help="Direct NWS forecast URL.")
    parser.add_argument(
        "--forecast-type",
        choices=["daily", "hourly"],
        default="daily",
        help="Forecast type to fetch when using lat/lon mode.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=None,
        help="Number of forecast periods to print.",
    )
    return parser.parse_args()


def _resolve_coords(
    args: argparse.Namespace, settings: Settings
) -> tuple[float | None, float | None]:
    lat = args.lat if args.lat is not None else settings.weather_default_lat
    lon = args.lon if args.lon is not None else settings.weather_default_lon
    return lat, lon


def _validate_cli_input(
    args: argparse.Namespace, settings: Settings
) -> tuple[float | None, float | None]:
    if args.max_print is not None and args.max_print <= 0:
        raise WeatherProviderError("--max-print must be > 0 when provided.")

    if args.url:
        if args.lat is not None or args.lon is not None:
            raise WeatherProviderError("Use either --url or --lat/--lon, not both.")
        return None, None

    lat, lon = _resolve_coords(args, settings)

    if lat is None or lon is None:
        raise WeatherProviderError(
            "Missing location input: pass --lat and --lon, set "
            "WEATHER_DEFAULT_LAT/LON, or use --url."
        )
    if not (-90 <= lat <= 90):
        raise WeatherProviderError(f"Invalid latitude {lat}; expected between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise WeatherProviderError(f"Invalid longitude {lon}; expected between -180 and 180.")
    return lat, lon


def _print_snapshot_summary(console: Console, snapshot: WeatherSnapshot, max_print: int) -> None:
    location_parts = []
    if snapshot.location.city and snapshot.location.state:
        location_parts.append(f"{snapshot.location.city}, {snapshot.location.state}")
    if snapshot.location.latitude is not None and snapshot.location.longitude is not None:
        location_parts.append(
            f"({snapshot.location.latitude:.4f}, {snapshot.location.longitude:.4f})"
        )
    location_str = " | ".join(location_parts) if location_parts else "unknown"

    console.print(
        f"Provider={snapshot.provider} type={snapshot.forecast_type} "
        f"periods={len(snapshot.periods)} location={location_str}"
    )

    if not snapshot.periods:
        console.print("No forecast periods found.")
        return

    table = Table(title="NWS Forecast Periods")
    table.add_column("Name", overflow="fold")
    table.add_column("Start (UTC)")
    table.add_column("Temp")
    table.add_column("Wind")
    table.add_column("Precip %")
    table.add_column("Short Forecast", overflow="fold")

    for period in snapshot.periods[:max_print]:
        temp = (
            f"{period.temperature:g} {period.temperature_unit}"
            if period.temperature is not None and period.temperature_unit
            else "-"
        )
        wind = (
            " ".join(part for part in [period.wind_speed, period.wind_direction] if part)
            if period.wind_speed or period.wind_direction
            else "-"
        )
        precip = (
            f"{period.probability_of_precipitation:g}"
            if period.probability_of_precipitation is not None
            else "-"
        )
        table.add_row(
            period.name or "-",
            period.start_time.astimezone(UTC).isoformat(),
            temp,
            wind,
            precip,
            period.short_forecast or "-",
        )
    console.print(table)


def main() -> int:
    """Run Day 2 weather ingestion flow."""
    args = parse_args()
    logger = setup_logger()
    console = Console()
    session_id = uuid.uuid4().hex[:12]
    journal: JournalWriter | None = None

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("Configuration failure: %s", exc)
        return 2

    try:
        journal = JournalWriter(
            journal_dir=settings.journal_dir,
            raw_payload_dir=settings.weather_raw_payload_dir,
            session_id=session_id,
        )
        journal.write_event(
            event_type="weather_startup",
            payload={
                "provider": "nws",
                "forecast_type": args.forecast_type,
                "raw_journaling_enabled": settings.weather_journal_raw_payloads,
            },
            metadata={"session_id": session_id},
        )
    except JournalError as exc:
        logger.error("Failed to initialize weather journal: %s", exc)
        return 3

    exit_code = 0
    try:
        lat, lon = _validate_cli_input(args, settings)
        request_payload = {
            "forecast_type": args.forecast_type,
            "url_mode": bool(args.url),
            "lat": lat,
            "lon": lon,
            "url": args.url,
        }
        journal.write_event(
            "weather_request_start",
            payload=request_payload,
            metadata={"session_id": session_id},
        )

        with NWSWeatherProvider(settings=settings, logger=logger) as provider:
            fetch_result = provider.fetch_forecast(
                lat=lat,
                lon=lon,
                forecast_url=args.url,
                forecast_type=args.forecast_type,
            )

            forecast_raw_path: str | None = None
            points_raw_path: str | None = None
            if settings.weather_journal_raw_payloads:
                if fetch_result.raw_points_payload is not None:
                    points_raw_path = str(
                        journal.write_raw_snapshot("nws_points", fetch_result.raw_points_payload)
                    )
                forecast_raw_path = str(
                    journal.write_raw_snapshot("nws_forecast", fetch_result.raw_forecast_payload)
                )
                journal.write_event(
                    "weather_raw_snapshot",
                    payload={
                        "points_raw_path": points_raw_path,
                        "forecast_raw_path": forecast_raw_path,
                    },
                    metadata={"session_id": session_id},
                )

            snapshot = fetch_result.snapshot
            if forecast_raw_path:
                snapshot = snapshot.model_copy(update={"raw_payload_path": forecast_raw_path})

            journal.write_event(
                "weather_snapshot_normalized",
                payload=snapshot.model_dump(mode="json"),
                metadata={"session_id": session_id},
            )
            journal.write_event(
                "weather_request_success",
                payload={
                    "provider": snapshot.provider,
                    "forecast_type": snapshot.forecast_type,
                    "period_count": len(snapshot.periods),
                },
                metadata={"session_id": session_id},
            )

            max_print = args.max_print or settings.weather_max_print
            _print_snapshot_summary(console, snapshot=snapshot, max_print=max_print)
    except (WeatherProviderError, JournalError) as exc:
        exit_code = 4
        logger.error("Weather ingestion failure: %s", exc)
        try:
            journal.write_event(
                "weather_request_failure",
                payload={"error": str(exc)},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write weather_request_failure event.")
    except Exception as exc:  # pragma: no cover - defensive catch for CLI runtime
        exit_code = 99
        logger.exception("Unexpected weather CLI failure: %s", exc)
        try:
            journal.write_event(
                "weather_request_failure_unhandled",
                payload={"error": str(exc), "type": type(exc).__name__},
                metadata={"session_id": session_id},
            )
        except JournalError:
            logger.error("Failed to write weather_request_failure_unhandled event.")
    finally:
        if journal is not None:
            try:
                journal.write_event(
                    "weather_shutdown",
                    payload={"exit_code": exit_code},
                    metadata={"session_id": session_id},
                )
            except JournalError:
                logger.error("Failed to write weather_shutdown event.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

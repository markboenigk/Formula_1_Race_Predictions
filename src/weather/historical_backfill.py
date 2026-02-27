"""
Historical weather backfill for strict bronze -> silver -> gold layering.

This script fetches Open-Meteo archive data for 2022+ race days and writes
bronze JSON payloads that match the schema produced by fetch_weather_forecast.
It updates the race (R) DynamoDB row pointer so weather_to_silver can ingest
without any additional adapter code.

Run locally: python -m src.weather.historical_backfill
"""
import json
import os
import time
from datetime import datetime, timezone

import boto3
import fastf1
import requests
from botocore.exceptions import ClientError

from src.common.circuit_coordinates import get_circuit_coords, normalize_event_name

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_BRONZE_PATH = os.environ.get("S3_BRONZE_PATH", "bronze").rstrip("/")
S3_WEATHER_FORECAST_HISTORICAL_PREFIX = os.environ.get(
    "S3_WEATHER_FORECAST_HISTORICAL_PREFIX", "weather_forecast_historical"
).rstrip("/")
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
BACKFILL_START_YEAR = 2022
BACKFILL_END_YEAR = datetime.now().year
WET_RACE_THRESHOLD_MM = 1.0
API_DELAY_SECONDS = 0.15

PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"
RACE_ABR = "R"
S3_HISTORICAL_WEATHER_FORECAST_LOCATION = "s3_historical_weather_forecast_location"


def _event_partition_key(year: int, event_name: str) -> str:
    return f"{year}_{str(event_name).strip().replace(' ', '_')}"


def s3_key_for_event(year: int, event_name: str, run_ts: str) -> str:
    """Return bronze key: bronze/weather_forecast_historical/.../forecast_window.json"""
    slug = normalize_event_name(event_name)
    return (
        f"{S3_BRONZE_PATH}/{S3_WEATHER_FORECAST_HISTORICAL_PREFIX}"
        f"/event_year={year}/event={slug}/run_ts={run_ts}/forecast_window.json"
    )


def event_already_cached(s3_client, table, bucket: str, event_partition_key: str) -> bool:
    """
    Check race-row historical pointer; if present and object exists in S3, treat as cached.
    """
    try:
        resp = table.get_item(Key={PARTITION_KEY: event_partition_key, SORT_KEY: RACE_ABR})
        item = resp.get("Item") or {}
        existing_key = item.get(S3_HISTORICAL_WEATHER_FORECAST_LOCATION)
        if not existing_key or not isinstance(existing_key, str):
            return False
        s3_client.head_object(Bucket=bucket, Key=existing_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def _coalesce_precip(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _coalesce_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_get(values, idx: int, default=None):
    if not isinstance(values, list):
        return default
    if idx < 0 or idx >= len(values):
        return default
    val = values[idx]
    return default if val is None else val


def _parse_hourly_time(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_race_hour_features(payload: dict, race_dt_utc: datetime) -> dict:
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    if not isinstance(timestamps, list) or not timestamps:
        return {
            "race_hour_temperature_c": None,
            "race_hour_precipitation_probability": None,  # archive endpoint does not provide this
            "race_hour_precipitation_mm": None,
            "race_hour_rain_mm": None,
            "race_hour_wind_speed_kmh": None,
            "race_hour_timestamp_utc": None,
            "race_hour_offset_minutes": None,
        }

    parsed = []
    for raw in timestamps:
        try:
            parsed.append(_parse_hourly_time(raw))
        except Exception:
            parsed.append(None)

    best_idx = None
    best_delta = None
    for idx, dt in enumerate(parsed):
        if dt is None:
            continue
        delta = abs((dt - race_dt_utc).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx

    if best_idx is None:
        return {
            "race_hour_temperature_c": None,
            "race_hour_precipitation_probability": None,
            "race_hour_precipitation_mm": None,
            "race_hour_rain_mm": None,
            "race_hour_wind_speed_kmh": None,
            "race_hour_timestamp_utc": None,
            "race_hour_offset_minutes": None,
        }

    selected_time = parsed[best_idx]
    offset_minutes = int((selected_time - race_dt_utc).total_seconds() / 60.0)
    return {
        "race_hour_temperature_c": _safe_get(hourly.get("temperature_2m"), best_idx),
        "race_hour_precipitation_probability": None,  # archive endpoint does not provide this
        "race_hour_precipitation_mm": _safe_get(hourly.get("precipitation"), best_idx),
        "race_hour_rain_mm": _safe_get(hourly.get("rain"), best_idx),
        "race_hour_wind_speed_kmh": _safe_get(hourly.get("wind_speed_10m"), best_idx),
        "race_hour_timestamp_utc": selected_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "race_hour_offset_minutes": offset_minutes,
    }


def fetch_weather_from_open_meteo(lat: float, lng: float, date_str: str) -> dict:
    """
    Compatibility helper used in local tests: returns daily weather features only.
    """
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily") or {}
    return {
        "precipitation_sum_mm": _coalesce_precip(_safe_get(daily.get("precipitation_sum"), 0, 0.0)),
        "rain_sum_mm": _coalesce_precip(_safe_get(daily.get("rain_sum"), 0, 0.0)),
        "temperature_max_c": _coalesce_float(_safe_get(daily.get("temperature_2m_max"), 0)),
        "temperature_min_c": _coalesce_float(_safe_get(daily.get("temperature_2m_min"), 0)),
        "wind_speed_max_kmh": _coalesce_float(_safe_get(daily.get("wind_speed_10m_max"), 0)),
        "wind_gusts_max_kmh": _coalesce_float(_safe_get(daily.get("wind_gusts_10m_max"), 0)),
    }


def fetch_weather_bundle_from_open_meteo(lat: float, lng: float, race_dt_utc: datetime) -> dict:
    """
    Fetch archive payload and return bronze-compatible daily_features + race_hour_features.
    """
    date_str = race_dt_utc.date().isoformat()
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max",
        "hourly": "temperature_2m,precipitation,rain,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily") or {}

    daily_features = {
        "precipitation_sum_mm": _coalesce_precip(_safe_get(daily.get("precipitation_sum"), 0, 0.0)),
        "precipitation_probability_max": None,  # archive endpoint does not provide this
        "temperature_max_c": _coalesce_float(_safe_get(daily.get("temperature_2m_max"), 0)),
        "temperature_min_c": _coalesce_float(_safe_get(daily.get("temperature_2m_min"), 0)),
        "wind_speed_max_kmh": _coalesce_float(_safe_get(daily.get("wind_speed_10m_max"), 0)),
    }

    return {
        "open_meteo_response": payload,
        "daily_features": daily_features,
        "race_hour_features": _extract_race_hour_features(payload, race_dt_utc),
    }


def build_weather_row(year: int, event_name: str, race_date: str, lat: float, lng: float, weather: dict) -> dict:
    """
    Compatibility helper kept for local tests from plan 01-02.
    """
    precip = weather.get("precipitation_sum_mm", 0) or 0
    return {
        "event_year": year,
        "event": normalize_event_name(event_name),
        "event_name": event_name,
        "race_date": race_date,
        "circuit_lat": lat,
        "circuit_lng": lng,
        **weather,
        "is_wet_race": precip >= WET_RACE_THRESHOLD_MM,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_json_to_s3(s3_client, bucket: str, key: str, payload: dict) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def _update_race_historical_pointer(table, event_partition_key: str, s3_key: str) -> None:
    table.update_item(
        Key={PARTITION_KEY: event_partition_key, SORT_KEY: RACE_ABR},
        UpdateExpression="SET #hist = :s3_key",
        ExpressionAttributeNames={"#hist": S3_HISTORICAL_WEATHER_FORECAST_LOCATION},
        ExpressionAttributeValues={":s3_key": s3_key},
    )


def get_race_events(year: int) -> list[dict]:
    """
    Enumerate race events for year via FastF1. Skips testing and future races.
    Returns list of {"year": int, "event_name": str, "race_dt_utc": datetime}.
    """
    schedule = fastf1.get_event_schedule(year)
    now_utc = datetime.now(timezone.utc)
    events = []
    for _, row in schedule.iterrows():
        if row.get("EventFormat") == "testing":
            continue
        try:
            event = fastf1.get_event(year, row["RoundNumber"])
            race_session = event.get_session("Race")
            race_dt = race_session.date
            if race_dt.tzinfo is None:
                race_dt = race_dt.replace(tzinfo=timezone.utc)
            race_dt = race_dt.astimezone(timezone.utc)
            if race_dt.date() > now_utc.date():
                continue
            events.append(
                {
                    "year": year,
                    "event_name": row["EventName"],
                    "race_dt_utc": race_dt,
                }
            )
        except Exception as e:
            print(
                f"  [WARN] Skipping round {row.get('RoundNumber')} "
                f"({row.get('EventName', '?')}): {e}"
            )
    return events


def run_backfill() -> dict:
    """
    Main orchestrator: fetch archive weather and write bronze JSON + DynamoDB pointer.
    This keeps strict bronze -> silver -> gold layering.
    """
    cache_dir = "/tmp/f1cache" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "./f1cache"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    s3 = boto3.client("s3")
    ddb_table = boto3.resource("dynamodb").Table(DYNAMODB_TABLE_NAME)

    total_events = 0
    cached_skipped = 0
    fetched = 0
    errors = []

    for year in range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1):
        try:
            race_events = get_race_events(year)
        except Exception as e:
            print(f"[ERROR] get_race_events({year}): {e}")
            errors.append({"year": year, "stage": "get_race_events", "error": str(e)})
            continue

        for ev in race_events:
            total_events += 1
            event_year = ev["year"]
            event_name = ev["event_name"]
            race_dt_utc = ev["race_dt_utc"]
            event_pk = _event_partition_key(event_year, event_name)

            try:
                # Require R row to exist so we can set the pointer (weather_to_silver reads via DynamoDB).
                resp = ddb_table.get_item(Key={PARTITION_KEY: event_pk, SORT_KEY: RACE_ABR})
                if not resp.get("Item"):
                    print(f"  [WARN] No R row for {event_pk} — run create_tracking_table first; skipping")
                    errors.append({"year": event_year, "event": event_name, "error": "No R row in tracking table"})
                    continue

                if event_already_cached(s3, ddb_table, S3_BUCKET, event_pk):
                    cached_skipped += 1
                    print(f"  [skip] {event_year} {event_name} (already cached via race-row pointer)")
                    continue

                lat, lng = get_circuit_coords(event_name)
                bundle = fetch_weather_bundle_from_open_meteo(lat, lng, race_dt_utc)
                run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                key = s3_key_for_event(event_year, event_name, run_ts)

                bronze_payload = {
                    "metadata": {
                        PARTITION_KEY: event_pk,
                        "EventName": event_name,
                        "event_year": event_year,
                        "qualifying_dt_utc": None,
                        "race_dt_utc": race_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "circuit_lat": lat,
                        "circuit_lng": lng,
                        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "forecast_endpoint_mode": "historical",
                        "forecast_endpoint": OPEN_METEO_ARCHIVE_URL,
                    },
                    "daily_features": bundle["daily_features"],
                    "race_hour_features": bundle["race_hour_features"],
                    "open_meteo_response": bundle["open_meteo_response"],
                }

                _write_json_to_s3(s3, S3_BUCKET, key, bronze_payload)
                _update_race_historical_pointer(ddb_table, event_pk, key)
                fetched += 1
                print(f"  [ok] {event_year} {event_name} -> s3://{S3_BUCKET}/{key}")
            except KeyError as e:
                print(f"  [WARN] No coordinates for {event_name}, skipping: {e}")
                errors.append({"year": event_year, "event": event_name, "error": str(e)})
            except Exception as e:
                print(f"  [ERROR] {event_year} {event_name}: {e}")
                errors.append({"year": event_year, "event": event_name, "error": str(e)})

            time.sleep(API_DELAY_SECONDS)

    result = {
        "total_events": total_events,
        "cached_skipped": cached_skipped,
        "fetched": fetched,
        "errors": len(errors),
    }
    if errors:
        print(f"\nErrors summary: {errors}")
    return result


if __name__ == "__main__":
    result = run_backfill()
    print(f"\nBackfill complete: {result}")

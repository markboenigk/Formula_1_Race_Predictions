import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
import requests
from boto3.dynamodb.conditions import Attr, Key

# ============================================================================
# DynamoDB Schema Constants
# ============================================================================
PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"
EVENT_YEAR = "event_year"
EVENT_NAME = "EventName"
SESSION_DATE_UTC = "session_date_utc"
CIRCUIT_LAT = "circuit_lat"
CIRCUIT_LNG = "circuit_lng"
S3_HISTORICAL_WEATHER_FORECAST_LOCATION = "s3_historical_weather_forecast_location"
S3_LIVE_WEATHER_FORECAST_LOCATION = "s3_live_weather_forecast_location"

# Session abbreviations used for forecast window bounds
QUALIFYING_ABR = "Q"
RACE_ABR = "R"

# ============================================================================
# Configuration
# ============================================================================
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_BRONZE_PATH = os.environ.get("S3_BRONZE_PATH") or os.environ.get("S3_WL_BRONZE_PATH", "bronze")
S3_WEATHER_FORECAST_LIVE_PREFIX = os.environ.get(
    "S3_WEATHER_FORECAST_LIVE_PREFIX", "weather_forecast_live"
)
S3_WEATHER_FORECAST_HISTORICAL_PREFIX = os.environ.get(
    "S3_WEATHER_FORECAST_HISTORICAL_PREFIX", "weather_forecast_historical"
)


from src.common.circuit_coordinates import normalize_event_name as _normalize_event_name


def _parse_utc_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _load_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        return {}
    if isinstance(body, dict):
        return body
    if isinstance(body, str) and body.strip():
        return json.loads(body)
    return {}


def _scan_candidate_sessions(table_name: str, event_partition_key: Optional[str] = None) -> List[Dict[str, Any]]:
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    if event_partition_key:
        response = table.query(KeyConditionExpression=Key(PARTITION_KEY).eq(event_partition_key))
        items = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            response = table.query(
                KeyConditionExpression=Key(PARTITION_KEY).eq(event_partition_key),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))
        return items

    filter_expression = (
        (Attr(SORT_KEY).eq(QUALIFYING_ABR) | Attr(SORT_KEY).eq(RACE_ABR))
        & Attr(SESSION_DATE_UTC).exists()
        & Attr(CIRCUIT_LAT).exists()
        & Attr(CIRCUIT_LNG).exists()
    )
    items: List[Dict[str, Any]] = []
    response = table.scan(FilterExpression=filter_expression)
    items.extend(response.get("Items", []))
    while "LastEvaluatedKey" in response:
        response = table.scan(FilterExpression=filter_expression, ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response.get("Items", []))
    return items


def _build_event_windows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_event: Dict[str, Dict[str, Any]] = {}
    for item in items:
        event_key = str(item.get(PARTITION_KEY, "")).strip()
        session_abr = str(item.get(SORT_KEY, "")).strip()
        if not event_key or session_abr not in (QUALIFYING_ABR, RACE_ABR):
            continue
        by_event.setdefault(event_key, {})[session_abr] = item

    windows: List[Dict[str, Any]] = []
    for event_key, sessions in by_event.items():
        q = sessions.get(QUALIFYING_ABR)
        r = sessions.get(RACE_ABR)
        if not q or not r:
            continue

        q_ts = q.get(SESSION_DATE_UTC)
        r_ts = r.get(SESSION_DATE_UTC)
        if not q_ts or not r_ts:
            continue

        try:
            q_dt = _parse_utc_iso(str(q_ts))
            r_dt = _parse_utc_iso(str(r_ts))
        except ValueError:
            continue

        if r_dt < q_dt:
            continue

        lat = _to_float(r.get(CIRCUIT_LAT))
        lng = _to_float(r.get(CIRCUIT_LNG))
        if lat is None or lng is None:
            lat = _to_float(q.get(CIRCUIT_LAT))
            lng = _to_float(q.get(CIRCUIT_LNG))
        if lat is None or lng is None:
            continue

        event_name = str(r.get(EVENT_NAME) or q.get(EVENT_NAME) or "")
        event_year = int(r.get(EVENT_YEAR) or q.get(EVENT_YEAR) or 0)
        windows.append(
            {
                PARTITION_KEY: event_key,
                EVENT_NAME: event_name,
                EVENT_YEAR: event_year,
                "qualifying_dt_utc": q_dt,
                "race_dt_utc": r_dt,
                CIRCUIT_LAT: lat,
                CIRCUIT_LNG: lng,
            }
        )

    windows.sort(key=lambda x: x["race_dt_utc"])
    return windows


def _select_target_window(windows: List[Dict[str, Any]], now_utc: datetime) -> Optional[Dict[str, Any]]:
    active = [w for w in windows if w["qualifying_dt_utc"] <= now_utc <= w["race_dt_utc"]]
    if active:
        active.sort(key=lambda x: x["race_dt_utc"])
        return active[0]

    # Open-Meteo "16-day" forecast can reject boundary dates, so keep a 1-day safety buffer.
    upcoming = [
        w
        for w in windows
        if w["qualifying_dt_utc"] > now_utc and (w["race_dt_utc"].date() - now_utc.date()).days <= 15
    ]
    if upcoming:
        upcoming.sort(key=lambda x: x["qualifying_dt_utc"])
        return upcoming[0]
    return None


def _select_open_meteo_endpoint(start_date_utc: datetime, now_utc: datetime) -> Dict[str, str]:
    """
    Use historical-forecast for past windows and live forecast for current/future windows.
    """
    if start_date_utc.date() < now_utc.date():
        return {"mode": "historical", "url": OPEN_METEO_HISTORICAL_FORECAST_URL}
    return {"mode": "live", "url": OPEN_METEO_FORECAST_URL}


def _fetch_forecast_window(
    lat: float,
    lng: float,
    qualifying_dt_utc: datetime,
    race_dt_utc: datetime,
    endpoint_url: str,
) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": qualifying_dt_utc.date().isoformat(),
        "end_date": race_dt_utc.date().isoformat(),
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "hourly": "temperature_2m,precipitation_probability,precipitation,rain,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(endpoint_url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _safe_get(values: Optional[List[Any]], idx: int, default: Any = None) -> Any:
    if not isinstance(values, list):
        return default
    if idx < 0 or idx >= len(values):
        return default
    val = values[idx]
    if val is None:
        return default
    return val


def _extract_daily_features(payload: Dict[str, Any], race_dt_utc: datetime) -> Dict[str, Any]:
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    race_date = race_dt_utc.date().isoformat()
    idx = dates.index(race_date) if race_date in dates else (len(dates) - 1 if dates else 0)
    return {
        "precipitation_sum_mm": _safe_get(daily.get("precipitation_sum"), idx, 0.0) or 0.0,
        "precipitation_probability_max": _safe_get(daily.get("precipitation_probability_max"), idx, 0.0) or 0.0,
        "temperature_max_c": _safe_get(daily.get("temperature_2m_max"), idx),
        "temperature_min_c": _safe_get(daily.get("temperature_2m_min"), idx),
        "wind_speed_max_kmh": _safe_get(daily.get("wind_speed_10m_max"), idx),
    }


def _parse_hourly_time(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_race_hour_features(payload: Dict[str, Any], race_dt_utc: datetime) -> Dict[str, Any]:
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    if not isinstance(timestamps, list) or not timestamps:
        return {
            "race_hour_temperature_c": None,
            "race_hour_precipitation_probability": None,
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
        "race_hour_precipitation_probability": _safe_get(hourly.get("precipitation_probability"), best_idx),
        "race_hour_precipitation_mm": _safe_get(hourly.get("precipitation"), best_idx),
        "race_hour_rain_mm": _safe_get(hourly.get("rain"), best_idx),
        "race_hour_wind_speed_kmh": _safe_get(hourly.get("wind_speed_10m"), best_idx),
        "race_hour_timestamp_utc": selected_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "race_hour_offset_minutes": offset_minutes,
    }


def _build_s3_key(event_year: int, event_name: str, run_ts: str, mode: str) -> str:
    slug = _normalize_event_name(event_name)
    prefix = S3_BRONZE_PATH.rstrip("/")
    weather_prefix = (
        S3_WEATHER_FORECAST_HISTORICAL_PREFIX
        if mode == "historical"
        else S3_WEATHER_FORECAST_LIVE_PREFIX
    )
    return f"{prefix}/{weather_prefix}/event_year={event_year}/event={slug}/run_ts={run_ts}/forecast_window.json"


def _write_json_to_s3(payload: Dict[str, Any], key: str) -> None:
    s3_client = boto3.client("s3")
    body = json.dumps(payload, default=str).encode("utf-8")
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=body,
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def _update_race_forecast_location(
    table_name: str,
    event_partition_key: str,
    s3_key: str,
    endpoint_mode: str,
) -> None:
    """
    Attach forecast artifact location to the race (R) row for the event.
    Uses separate fields for historical vs live forecast outputs.
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    if endpoint_mode == "historical":
        update_expression = "SET #hist = :s3_key"
        attr_names = {"#hist": S3_HISTORICAL_WEATHER_FORECAST_LOCATION}
    else:
        update_expression = "SET #live = :s3_key"
        attr_names = {"#live": S3_LIVE_WEATHER_FORECAST_LOCATION}

    table.update_item(
        Key={PARTITION_KEY: event_partition_key, SORT_KEY: RACE_ABR},
        UpdateExpression=update_expression,
        ExpressionAttributeNames=attr_names,
        ExpressionAttributeValues={":s3_key": s3_key},
    )


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        body = _load_body(event)
        target_event_key = body.get(PARTITION_KEY)
        dry_run = bool(body.get("dry_run", False))

        items = _scan_candidate_sessions(DYNAMODB_TABLE_NAME, target_event_key)
        windows = _build_event_windows(items)
        if not windows:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {"error": "No qualifying/race windows with coordinates found in tracking table."}
                ),
            }

        now_utc = datetime.now(timezone.utc)
        if target_event_key:
            matching = [w for w in windows if w.get(PARTITION_KEY) == target_event_key]
            target = matching[0] if matching else None
        else:
            target = _select_target_window(windows, now_utc)

        if not target:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "No target event found for forecast fetch."}),
            }

        q_dt = target["qualifying_dt_utc"]
        r_dt = target["race_dt_utc"]
        endpoint_cfg = _select_open_meteo_endpoint(q_dt, now_utc)
        endpoint_mode = endpoint_cfg["mode"]
        endpoint_url = endpoint_cfg["url"]

        days_to_race = (r_dt.date() - now_utc.date()).days
        if endpoint_mode == "live" and days_to_race > 16:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": "Target race is outside Open-Meteo forecast window (16 days).",
                        PARTITION_KEY: target[PARTITION_KEY],
                        "days_to_race": days_to_race,
                    }
                ),
            }

        open_meteo_response = _fetch_forecast_window(
            lat=target[CIRCUIT_LAT],
            lng=target[CIRCUIT_LNG],
            qualifying_dt_utc=q_dt,
            race_dt_utc=r_dt,
            endpoint_url=endpoint_url,
        )
        daily_features = _extract_daily_features(open_meteo_response, r_dt)
        race_hour_features = _extract_race_hour_features(open_meteo_response, r_dt)

        run_ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
        s3_key = _build_s3_key(target[EVENT_YEAR], target[EVENT_NAME], run_ts, endpoint_mode)

        bronze_payload = {
            "metadata": {
                PARTITION_KEY: target[PARTITION_KEY],
                EVENT_NAME: target[EVENT_NAME],
                EVENT_YEAR: target[EVENT_YEAR],
                "qualifying_dt_utc": q_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "race_dt_utc": r_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                CIRCUIT_LAT: target[CIRCUIT_LAT],
                CIRCUIT_LNG: target[CIRCUIT_LNG],
                "fetched_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "forecast_endpoint_mode": endpoint_mode,
                "forecast_endpoint": endpoint_url,
            },
            "daily_features": daily_features,
            "race_hour_features": race_hour_features,
            "open_meteo_response": open_meteo_response,
        }

        if not dry_run:
            _write_json_to_s3(bronze_payload, s3_key)
            _update_race_forecast_location(
                table_name=DYNAMODB_TABLE_NAME,
                event_partition_key=target[PARTITION_KEY],
                s3_key=s3_key,
                endpoint_mode=endpoint_mode,
            )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "success": True,
                    "dry_run": dry_run,
                    PARTITION_KEY: target[PARTITION_KEY],
                    "s3_key": s3_key if not dry_run else None,
                    "forecast_endpoint_mode": endpoint_mode,
                    "forecast_endpoint": endpoint_url,
                    "daily_features": daily_features,
                    "race_hour_features": race_hour_features,
                }
            ),
        }
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        return {
            "statusCode": status_code,
            "body": json.dumps({"error": f"Open-Meteo HTTP error: {str(e)}"}),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

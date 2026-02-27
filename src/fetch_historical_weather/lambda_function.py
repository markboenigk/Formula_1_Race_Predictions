"""
Fetch historical qualifying weather from Open-Meteo archive API.
Reads Q row from DynamoDB, calls archive-api.open-meteo.com for qualifying day (hourly),
writes bronze and updates Q row with s3_historical_qualifying_weather_location.
"""
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
S3_HISTORICAL_QUALIFYING_WEATHER_LOCATION = "s3_historical_qualifying_weather_location"

QUALIFYING_ABR = "Q"
RACE_ABR = "R"

# ============================================================================
# Configuration
# ============================================================================
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_BRONZE_PATH = os.environ.get("S3_BRONZE_PATH") or os.environ.get("S3_WL_BRONZE_PATH", "bronze")
S3_WEATHER_HISTORICAL_QUALIFYING_PREFIX = os.environ.get(
    "S3_WEATHER_HISTORICAL_QUALIFYING_PREFIX", "weather_historical_qualifying"
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


def _scan_candidate_q_sessions(
    table_name: str, event_partition_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get items that include Q (and optionally R for lat/lng fallback)."""
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
        response = table.scan(
            FilterExpression=filter_expression, ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response.get("Items", []))
    return items


def _build_qualifying_windows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one record per event with qualifying_dt_utc and circuit coords (from Q or R)."""
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
        if not q:
            continue

        q_ts = q.get(SESSION_DATE_UTC)
        if not q_ts:
            continue

        try:
            q_dt = _parse_utc_iso(str(q_ts))
        except (ValueError, TypeError):
            continue

        lat = _to_float(r.get(CIRCUIT_LAT) if r else None) or _to_float(q.get(CIRCUIT_LAT))
        lng = _to_float(r.get(CIRCUIT_LNG) if r else None) or _to_float(q.get(CIRCUIT_LNG))
        if lat is None or lng is None:
            continue

        event_name = str(q.get(EVENT_NAME) or (r.get(EVENT_NAME) if r else "") or "")
        event_year = int(q.get(EVENT_YEAR) or (r.get(EVENT_YEAR) if r else 0) or 0)
        windows.append(
            {
                PARTITION_KEY: event_key,
                EVENT_NAME: event_name,
                EVENT_YEAR: event_year,
                "qualifying_dt_utc": q_dt,
                CIRCUIT_LAT: lat,
                CIRCUIT_LNG: lng,
            }
        )

    windows.sort(key=lambda x: x["qualifying_dt_utc"])
    return windows


def _fetch_archive(lat: float, lng: float, qualifying_date_iso: str) -> Dict[str, Any]:
    """Call Open-Meteo archive API for a single day (qualifying day) with daily + hourly."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": qualifying_date_iso,
        "end_date": qualifying_date_iso,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "hourly": "temperature_2m,precipitation,rain,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _safe_get(values: Any, idx: int, default: Any = None) -> Any:
    if values is None:
        return default
    if not isinstance(values, list):
        return default
    if idx < 0 or idx >= len(values):
        return default
    val = values[idx]
    if val is None:
        return default
    return val


def _extract_daily_features_qualifying(
    payload: Dict[str, Any], qualifying_dt_utc: datetime
) -> Dict[str, Any]:
    """Extract daily features for the qualifying day (archive uses precipitation_sum, no precip prob)."""
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    q_date = qualifying_dt_utc.date().isoformat()
    idx = dates.index(q_date) if q_date in dates else (len(dates) - 1 if dates else 0)
    return {
        "precipitation_sum_mm": _safe_get(daily.get("precipitation_sum"), idx, 0.0) or 0.0,
        "precipitation_probability_max": None,  # archive does not provide
        "temperature_max_c": _safe_get(daily.get("temperature_2m_max"), idx),
        "temperature_min_c": _safe_get(daily.get("temperature_2m_min"), idx),
        "wind_speed_max_kmh": _safe_get(daily.get("wind_speed_10m_max"), idx),
    }


def _parse_hourly_time(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_qualifying_hour_features(
    payload: Dict[str, Any], qualifying_dt_utc: datetime
) -> Dict[str, Any]:
    """Hour closest to qualifying session start; same shape as race_hour_features (precip_prob = null)."""
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    if not isinstance(timestamps, list) or not timestamps:
        return {
            "qualifying_hour_temperature_c": None,
            "qualifying_hour_precipitation_probability": None,
            "qualifying_hour_precipitation_mm": None,
            "qualifying_hour_rain_mm": None,
            "qualifying_hour_wind_speed_kmh": None,
            "qualifying_hour_timestamp_utc": None,
            "qualifying_hour_offset_minutes": None,
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
        delta = abs((dt - qualifying_dt_utc).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx

    if best_idx is None:
        return {
            "qualifying_hour_temperature_c": None,
            "qualifying_hour_precipitation_probability": None,
            "qualifying_hour_precipitation_mm": None,
            "qualifying_hour_rain_mm": None,
            "qualifying_hour_wind_speed_kmh": None,
            "qualifying_hour_timestamp_utc": None,
            "qualifying_hour_offset_minutes": None,
        }

    selected_time = parsed[best_idx]
    offset_minutes = int((selected_time - qualifying_dt_utc).total_seconds() / 60.0)
    return {
        "qualifying_hour_temperature_c": _safe_get(hourly.get("temperature_2m"), best_idx),
        "qualifying_hour_precipitation_probability": None,  # archive has no precip probability
        "qualifying_hour_precipitation_mm": _safe_get(hourly.get("precipitation"), best_idx),
        "qualifying_hour_rain_mm": _safe_get(hourly.get("rain"), best_idx),
        "qualifying_hour_wind_speed_kmh": _safe_get(hourly.get("wind_speed_10m"), best_idx),
        "qualifying_hour_timestamp_utc": selected_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "qualifying_hour_offset_minutes": offset_minutes,
    }


def _build_s3_key(event_year: int, event_name: str, run_ts: str) -> str:
    slug = _normalize_event_name(event_name)
    prefix = S3_BRONZE_PATH.rstrip("/")
    return f"{prefix}/{S3_WEATHER_HISTORICAL_QUALIFYING_PREFIX}/event_year={event_year}/event={slug}/run_ts={run_ts}/qualifying_weather.json"


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


def _update_q_row_qualifying_weather_location(
    table_name: str, event_partition_key: str, s3_key: str
) -> None:
    """Write S3 key to the Q row for this event."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    table.update_item(
        Key={PARTITION_KEY: event_partition_key, SORT_KEY: QUALIFYING_ABR},
        UpdateExpression="SET #loc = :s3_key",
        ExpressionAttributeNames={"#loc": S3_HISTORICAL_QUALIFYING_WEATHER_LOCATION},
        ExpressionAttributeValues={":s3_key": s3_key},
    )


def _process_one_event(
    target: Dict[str, Any], dry_run: bool
) -> Dict[str, Any]:
    """Fetch archive for one event, write bronze, update Q row. Returns result dict for response."""
    now_utc = datetime.now(timezone.utc)
    q_dt = target["qualifying_dt_utc"]
    qualifying_date_iso = q_dt.date().isoformat()

    open_meteo_response = _fetch_archive(
        lat=target[CIRCUIT_LAT],
        lng=target[CIRCUIT_LNG],
        qualifying_date_iso=qualifying_date_iso,
    )
    daily_features = _extract_daily_features_qualifying(open_meteo_response, q_dt)
    qualifying_hour_features = _extract_qualifying_hour_features(open_meteo_response, q_dt)

    run_ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    s3_key = _build_s3_key(target[EVENT_YEAR], target[EVENT_NAME], run_ts)

    bronze_payload = {
        "metadata": {
            PARTITION_KEY: target[PARTITION_KEY],
            EVENT_NAME: target[EVENT_NAME],
            EVENT_YEAR: target[EVENT_YEAR],
            "qualifying_dt_utc": q_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            CIRCUIT_LAT: target[CIRCUIT_LAT],
            CIRCUIT_LNG: target[CIRCUIT_LNG],
            "fetched_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "archive",
            "archive_endpoint": OPEN_METEO_ARCHIVE_URL,
        },
        "daily_features": daily_features,
        "qualifying_hour_features": qualifying_hour_features,
        "open_meteo_response": open_meteo_response,
    }

    if not dry_run:
        _write_json_to_s3(bronze_payload, s3_key)
        _update_q_row_qualifying_weather_location(
            table_name=DYNAMODB_TABLE_NAME,
            event_partition_key=target[PARTITION_KEY],
            s3_key=s3_key,
        )

    return {
        PARTITION_KEY: target[PARTITION_KEY],
        "s3_key": s3_key if not dry_run else None,
        "daily_features": daily_features,
        "qualifying_hour_features": qualifying_hour_features,
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        body = _load_body(event)
        target_event_key = body.get(PARTITION_KEY)
        dry_run = bool(body.get("dry_run", False))
        limit = body.get("limit")  # optional max events when processing multiple

        items = _scan_candidate_q_sessions(DYNAMODB_TABLE_NAME, target_event_key)
        windows = _build_qualifying_windows(items)
        if not windows:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {"error": "No qualifying sessions with coordinates found in tracking table."}
                ),
            }

        if target_event_key:
            matching = [w for w in windows if w.get(PARTITION_KEY) == target_event_key]
            targets = matching[:1] if matching else []
        else:
            targets = windows
            if limit is not None and limit > 0:
                targets = targets[: int(limit)]

        if not targets:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "No target event(s) found for historical qualifying weather."}),
            }

        results: List[Dict[str, Any]] = []
        for target in targets:
            result = _process_one_event(target, dry_run=dry_run)
            results.append(result)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "success": True,
                    "dry_run": dry_run,
                    "processed": len(results),
                    "results": results,
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

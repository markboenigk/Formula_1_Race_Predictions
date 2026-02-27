"""
Weather bronze -> silver: read bronze forecast JSON (via DynamoDB race row pointers)
and write curated weather to silver (one row per event).

Lambda entrypoint: lambda_handler(event, context).
CLI: python lambda_function.py [--limit N] [--no-overwrite]
"""
import io
import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from boto3.dynamodb.conditions import Attr

# ---------------------------------------------------------------------------
# Config (align with fetch_weather_forecast and combine_data_into_silver)
# ---------------------------------------------------------------------------
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_SILVER_PATH = os.environ.get("S3_SILVER_PATH", "silver").rstrip("/")
S3_SILVER_OVERWRITE = os.environ.get("S3_SILVER_OVERWRITE", "true").strip().lower() in ("1", "true", "yes", "y")

S3_HISTORICAL_WEATHER_FORECAST_LOCATION = "s3_historical_weather_forecast_location"
S3_LIVE_WEATHER_FORECAST_LOCATION = "s3_live_weather_forecast_location"
S3_HISTORICAL_QUALIFYING_WEATHER_LOCATION = "s3_historical_qualifying_weather_location"
PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"
EVENT_YEAR = "event_year"
EVENT_NAME = "EventName"
RACE_ABR = "R"
QUALIFYING_ABR = "Q"

DATASET_NAME = "weather_forecast"
DATASET_HISTORICAL_QUALIFYING = "weather_historical_qualifying"
PARTITION_COLS = ["event_year", "event"]


from src.common.circuit_coordinates import normalize_event_name as _normalize_event_name


def _to_native(val: Any) -> Any:
    """Convert DynamoDB types (Decimal) to native Python for DataFrame."""
    if isinstance(val, Decimal):
        return float(val) if val % 1 else int(val)
    return val


def get_race_rows_with_forecast(
    table_name: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Scan DynamoDB for race (R) rows that have at least one forecast S3 location.
    Prefer historical over live when both exist (we use one row per event).
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    filter_expr = (
        Attr(SORT_KEY).eq(RACE_ABR)
        & (
            Attr(S3_HISTORICAL_WEATHER_FORECAST_LOCATION).exists()
            | Attr(S3_LIVE_WEATHER_FORECAST_LOCATION).exists()
        )
    )
    items: List[Dict[str, Any]] = []
    scan_kwargs = {"FilterExpression": filter_expr}
    response = table.scan(**scan_kwargs)
    items.extend(response.get("Items", []))
    while "LastEvaluatedKey" in response:
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))

    # One row per event: prefer historical forecast key if present, else live
    by_event: Dict[str, Dict[str, Any]] = {}
    for item in items:
        pk = str(item.get(PARTITION_KEY, "")).strip()
        if not pk:
            continue
        hist_key = item.get(S3_HISTORICAL_WEATHER_FORECAST_LOCATION)
        live_key = item.get(S3_LIVE_WEATHER_FORECAST_LOCATION)
        s3_key = None
        source = None
        if hist_key:
            s3_key = hist_key if isinstance(hist_key, str) else None
            source = "historical"
        if not s3_key and live_key:
            s3_key = live_key if isinstance(live_key, str) else None
            source = "live"
        if not s3_key:
            continue
        by_event[pk] = {
            PARTITION_KEY: pk,
            EVENT_YEAR: _to_native(item.get(EVENT_YEAR)),
            EVENT_NAME: str(item.get(EVENT_NAME, "")),
            "s3_key": s3_key,
            "forecast_source": source,
        }
    out = list(by_event.values())
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def get_q_rows_with_historical_qualifying_weather(
    table_name: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Scan DynamoDB for Q rows that have s3_historical_qualifying_weather_location.
    Returns one record per event (event_partition_key, event_year, EventName, s3_key).
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    filter_expr = (
        Attr(SORT_KEY).eq(QUALIFYING_ABR)
        & Attr(S3_HISTORICAL_QUALIFYING_WEATHER_LOCATION).exists()
    )
    items: List[Dict[str, Any]] = []
    scan_kwargs = {"FilterExpression": filter_expr}
    response = table.scan(**scan_kwargs)
    items.extend(response.get("Items", []))
    while "LastEvaluatedKey" in response:
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))

    out = []
    for item in items:
        pk = str(item.get(PARTITION_KEY, "")).strip()
        if not pk:
            continue
        s3_key = item.get(S3_HISTORICAL_QUALIFYING_WEATHER_LOCATION)
        if not isinstance(s3_key, str) or not s3_key:
            continue
        out.append({
            PARTITION_KEY: pk,
            EVENT_YEAR: _to_native(item.get(EVENT_YEAR)),
            EVENT_NAME: str(item.get(EVENT_NAME, "")),
            "s3_key": s3_key,
        })
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def read_bronze_forecast_json(bucket: str, key: str, s3_client: Any = None) -> Dict[str, Any]:
    """Load one bronze forecast JSON from S3."""
    if s3_client is None:
        s3_client = boto3.client("s3")
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)


def bronze_payload_to_silver_row(
    event_year: int,
    event_name: str,
    forecast_source: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one silver row from bronze payload (metadata + daily_features + race_hour_features).
    """
    meta = payload.get("metadata", {})
    daily = payload.get("daily_features", {})
    race_hour = payload.get("race_hour_features", {})

    event_slug = _normalize_event_name(event_name)
    row = {
        "event_year": event_year,
        "event": event_slug,
        "event_name": event_name,
        "forecast_source": forecast_source,
        "race_dt_utc": meta.get("race_dt_utc"),
        "qualifying_dt_utc": meta.get("qualifying_dt_utc"),
        "fetched_at_utc": meta.get("fetched_at_utc"),
        "precipitation_sum_mm": daily.get("precipitation_sum_mm"),
        "precipitation_probability_max": daily.get("precipitation_probability_max"),
        "temperature_max_c": daily.get("temperature_max_c"),
        "temperature_min_c": daily.get("temperature_min_c"),
        "wind_speed_max_kmh": daily.get("wind_speed_max_kmh"),
        "race_hour_temperature_c": race_hour.get("race_hour_temperature_c"),
        "race_hour_precipitation_probability": race_hour.get("race_hour_precipitation_probability"),
        "race_hour_precipitation_mm": race_hour.get("race_hour_precipitation_mm"),
        "race_hour_rain_mm": race_hour.get("race_hour_rain_mm"),
        "race_hour_wind_speed_kmh": race_hour.get("race_hour_wind_speed_kmh"),
        "race_hour_timestamp_utc": race_hour.get("race_hour_timestamp_utc"),
        "race_hour_offset_minutes": race_hour.get("race_hour_offset_minutes"),
    }
    return row


def bronze_payload_to_silver_row_historical_qualifying(
    event_year: int,
    event_name: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one silver row from historical qualifying bronze payload
    (metadata + daily_features + qualifying_hour_features).
    """
    meta = payload.get("metadata", {})
    daily = payload.get("daily_features", {})
    q_hour = payload.get("qualifying_hour_features", {})

    event_slug = _normalize_event_name(event_name)
    row = {
        "event_year": event_year,
        "event": event_slug,
        "event_name": event_name,
        "qualifying_dt_utc": meta.get("qualifying_dt_utc"),
        "fetched_at_utc": meta.get("fetched_at_utc"),
        "precipitation_sum_mm": daily.get("precipitation_sum_mm"),
        "precipitation_probability_max": daily.get("precipitation_probability_max"),
        "temperature_max_c": daily.get("temperature_max_c"),
        "temperature_min_c": daily.get("temperature_min_c"),
        "wind_speed_max_kmh": daily.get("wind_speed_max_kmh"),
        "qualifying_hour_temperature_c": q_hour.get("qualifying_hour_temperature_c"),
        "qualifying_hour_precipitation_probability": q_hour.get("qualifying_hour_precipitation_probability"),
        "qualifying_hour_precipitation_mm": q_hour.get("qualifying_hour_precipitation_mm"),
        "qualifying_hour_rain_mm": q_hour.get("qualifying_hour_rain_mm"),
        "qualifying_hour_wind_speed_kmh": q_hour.get("qualifying_hour_wind_speed_kmh"),
        "qualifying_hour_timestamp_utc": q_hour.get("qualifying_hour_timestamp_utc"),
        "qualifying_hour_offset_minutes": q_hour.get("qualifying_hour_offset_minutes"),
    }
    return row


def _delete_s3_prefix(bucket: str, prefix: str, s3_client: Any = None) -> int:
    if s3_client is None:
        s3_client = boto3.client("s3")
    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        keys = [{"Key": obj["Key"]} for obj in contents]
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            resp = s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            deleted += len(resp.get("Deleted", []))
    return deleted


def write_partitioned_parquet_to_s3(
    df: pd.DataFrame,
    dataset_name: str,
    partition_cols: List[str],
    bucket_name: str,
    silver_prefix: str,
    overwrite: bool = True,
    s3_client: Any = None,
) -> Dict[str, Any]:
    """Write Hive-partitioned Parquet to S3 (one file per partition)."""
    if df.empty:
        return {"success": True, "dataset": dataset_name, "files_written": 0, "rows": 0}
    if s3_client is None:
        s3_client = boto3.client("s3")

    missing = [c for c in partition_cols if c not in df.columns]
    if missing:
        return {"success": False, "dataset": dataset_name, "error": f"Missing partition columns: {missing}"}

    base_prefix = f"{silver_prefix.rstrip('/')}/{dataset_name}/"
    if overwrite:
        deleted = _delete_s3_prefix(bucket=bucket_name, prefix=base_prefix, s3_client=s3_client)
        print(f"Deleted {deleted} objects under s3://{bucket_name}/{base_prefix}")

    files_written = 0
    grouped = df.groupby(partition_cols, dropna=False, sort=False)
    for part_values, part_df in grouped:
        if not isinstance(part_values, tuple):
            part_values = (part_values,)
        partition_parts = []
        for col, val in zip(partition_cols, part_values):
            val_str = "__NULL__" if pd.isna(val) else str(val).strip().replace("/", "_")
            partition_parts.append(f"{col}={val_str}")
        partition_prefix = base_prefix + "/".join(partition_parts) + "/"
        filename = f"part-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}.parquet"
        key = partition_prefix + filename
        data_df = part_df.drop(columns=partition_cols, errors="ignore")
        table = pa.Table.from_pandas(data_df, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        s3_client.upload_fileobj(
            buf,
            bucket_name,
            key,
            ExtraArgs={"ContentType": "application/octet-stream", "ServerSideEncryption": "AES256"},
        )
        files_written += 1
    return {
        "success": True,
        "dataset": dataset_name,
        "files_written": files_written,
        "rows": len(df),
    }


def _run_forecast_silver(
    table_name: str,
    bucket_name: str,
    silver_prefix: str,
    overwrite: bool,
    limit: Optional[int],
    s3_client: Any,
) -> Dict[str, Any]:
    """Run silver for weather_forecast (race rows with forecast pointers)."""
    race_rows = get_race_rows_with_forecast(table_name, limit=limit)
    if not race_rows:
        print("No race rows with forecast location found in DynamoDB.")
        return {"success": True, "events_processed": 0, "rows_written": 0, "silver_write": {}}

    print(f"Found {len(race_rows)} events with forecast data.")
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for r in race_rows:
        s3_key = r["s3_key"]
        try:
            payload = read_bronze_forecast_json(bucket_name, s3_key, s3_client=s3_client)
        except Exception as e:
            errors.append(f"{r[PARTITION_KEY]}: {e}")
            continue
        row = bronze_payload_to_silver_row(
            event_year=int(r[EVENT_YEAR]),
            event_name=r[EVENT_NAME],
            forecast_source=r["forecast_source"],
            payload=payload,
        )
        rows.append(row)

    if errors:
        for e in errors:
            print(f"  Error: {e}")

    if not rows:
        print("No rows to write (all reads failed).")
        return {"success": False, "events_processed": len(race_rows), "rows_written": 0, "errors": errors}

    df = pd.DataFrame(rows)
    write_result = write_partitioned_parquet_to_s3(
        df,
        dataset_name=DATASET_NAME,
        partition_cols=PARTITION_COLS,
        bucket_name=bucket_name,
        silver_prefix=silver_prefix,
        overwrite=overwrite,
        s3_client=s3_client,
    )
    print(f"Silver write (forecast): {write_result}")
    return {
        "success": write_result.get("success", False),
        "events_processed": len(race_rows),
        "rows_written": len(df),
        "silver_write": write_result,
        "errors": errors if errors else None,
    }


def _run_historical_qualifying_silver(
    table_name: str,
    bucket_name: str,
    silver_prefix: str,
    overwrite: bool,
    limit: Optional[int],
    s3_client: Any,
) -> Dict[str, Any]:
    """Run silver for weather_historical_qualifying (Q rows with historical qualifying weather pointer)."""
    q_rows = get_q_rows_with_historical_qualifying_weather(table_name, limit=limit)
    if not q_rows:
        print("No Q rows with historical qualifying weather location found in DynamoDB.")
        return {"success": True, "events_processed": 0, "rows_written": 0, "silver_write": {}}

    print(f"Found {len(q_rows)} events with historical qualifying weather.")
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for r in q_rows:
        s3_key = r["s3_key"]
        try:
            payload = read_bronze_forecast_json(bucket_name, s3_key, s3_client=s3_client)
        except Exception as e:
            errors.append(f"{r[PARTITION_KEY]}: {e}")
            continue
        row = bronze_payload_to_silver_row_historical_qualifying(
            event_year=int(r[EVENT_YEAR]),
            event_name=r[EVENT_NAME],
            payload=payload,
        )
        rows.append(row)

    if errors:
        for e in errors:
            print(f"  Error: {e}")

    if not rows:
        print("No rows to write (all reads failed).")
        return {"success": False, "events_processed": len(q_rows), "rows_written": 0, "errors": errors}

    df = pd.DataFrame(rows)
    write_result = write_partitioned_parquet_to_s3(
        df,
        dataset_name=DATASET_HISTORICAL_QUALIFYING,
        partition_cols=PARTITION_COLS,
        bucket_name=bucket_name,
        silver_prefix=silver_prefix,
        overwrite=overwrite,
        s3_client=s3_client,
    )
    print(f"Silver write (historical_qualifying): {write_result}")
    return {
        "success": write_result.get("success", False),
        "events_processed": len(q_rows),
        "rows_written": len(df),
        "silver_write": write_result,
        "errors": errors if errors else None,
    }


def run(
    table_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
    silver_prefix: Optional[str] = None,
    overwrite: Optional[bool] = None,
    limit: Optional[int] = None,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry: read bronze via DynamoDB pointers, build silver DataFrame(s), write to S3.
    dataset: "forecast" | "historical_qualifying" | "all". Default "forecast" for backward compatibility.
    """
    table_name = table_name or DYNAMODB_TABLE_NAME
    bucket_name = bucket_name or S3_BUCKET_NAME
    silver_prefix = silver_prefix or S3_SILVER_PATH
    overwrite = overwrite if overwrite is not None else S3_SILVER_OVERWRITE
    dataset = (dataset or "forecast").strip().lower()

    s3_client = boto3.client("s3")

    if dataset == "forecast":
        return _run_forecast_silver(
            table_name, bucket_name, silver_prefix, overwrite, limit, s3_client
        )
    if dataset == "historical_qualifying":
        return _run_historical_qualifying_silver(
            table_name, bucket_name, silver_prefix, overwrite, limit, s3_client
        )
    if dataset == "all":
        result_f = _run_forecast_silver(
            table_name, bucket_name, silver_prefix, overwrite, limit, s3_client
        )
        result_q = _run_historical_qualifying_silver(
            table_name, bucket_name, silver_prefix, overwrite, limit, s3_client
        )
        return {
            "success": result_f.get("success", False) and result_q.get("success", False),
            "forecast": result_f,
            "historical_qualifying": result_q,
        }
    return {"success": False, "error": f"Unknown dataset: {dataset}"}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda entrypoint. Body may include: limit (int), overwrite (bool), dataset (str).
    dataset: "forecast" | "historical_qualifying" | "all". Env WEATHER_DATASET overrides body.
    Returns API Gateway-style response with statusCode and body (run result).
    """
    body = event.get("body")
    if isinstance(body, str) and body.strip():
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {}
    elif body is None:
        body = {}
    limit = body.get("limit")
    overwrite = body.get("overwrite", True)
    dataset = os.environ.get("WEATHER_DATASET") or body.get("dataset") or "forecast"
    result = run(limit=limit, overwrite=overwrite, dataset=dataset)
    status_code = 200 if result.get("success") else 500
    return {
        "statusCode": status_code,
        "body": json.dumps(result, default=str),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Weather bronze -> silver (one row per event)")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not delete existing silver before write")
    parser.add_argument("--dataset", type=str, default=None, choices=["forecast", "historical_qualifying", "all"],
                        help="Dataset to process (default: forecast)")
    args = parser.parse_args()
    result = run(limit=args.limit, overwrite=not args.no_overwrite, dataset=args.dataset or "forecast")
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

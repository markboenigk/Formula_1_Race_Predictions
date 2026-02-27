import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# DynamoDB Schema Constants (match create_tracking_table)
# ============================================================================
PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"

EVENT_YEAR = "event_year"
EVENT_NUMBER = "event_number"
EVENT_NAME = "EventName"
EVENT_FORMAT = "event_format"

SESSION_NAME = "session_name"
SESSION_DATE_UTC = "session_date_utc"
IS_BEFORE_RACE = "is_before_race"


def get_table_name() -> str:
    return os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")


# ============================================================================
# S3 / Export Configuration
# ============================================================================
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_SILVER_PATH = os.environ.get("S3_SILVER_PATH", "silver").rstrip("/")
S3_DIM_SESSIONS_OVERWRITE = (
    os.environ.get("S3_DIM_SESSIONS_OVERWRITE", "true").strip().lower() in ("1", "true", "yes", "y")
)


from src.common.circuit_coordinates import normalize_event_name


def _delete_s3_prefix(bucket: str, prefix: str, s3_client: Optional[Any] = None) -> int:
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
            resp = s3_client.delete_objects(Bucket=bucket, Delete={"Objects": keys[i : i + 1000]})
            deleted += len(resp.get("Deleted", []))
    return deleted


def write_partitioned_parquet_dataset_to_s3(
    df: pd.DataFrame,
    dataset_name: str,
    partition_cols: List[str],
    bucket_name: Optional[str] = None,
    silver_prefix: Optional[str] = None,
    overwrite: Optional[bool] = None,
    s3_client: Optional[Any] = None,
) -> Dict[str, Any]:
    if df.empty:
        return {"success": True, "dataset": dataset_name, "files_written": 0, "rows": 0}

    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    if silver_prefix is None:
        silver_prefix = S3_SILVER_PATH
    if overwrite is None:
        overwrite = S3_DIM_SESSIONS_OVERWRITE
    if s3_client is None:
        s3_client = boto3.client("s3")

    missing = [c for c in partition_cols if c not in df.columns]
    if missing:
        return {"success": False, "dataset": dataset_name, "error": f"Missing partition columns: {missing}"}

    base_prefix = f"{silver_prefix.rstrip('/')}/{dataset_name}/"
    deleted = 0
    if overwrite:
        deleted = _delete_s3_prefix(bucket=bucket_name, prefix=base_prefix, s3_client=s3_client)

    files_written = 0
    errors: List[str] = []

    grouped = df.groupby(partition_cols, dropna=False, sort=False)
    for part_values, part_df in grouped:
        if not isinstance(part_values, tuple):
            part_values = (part_values,)

        partition_parts: List[str] = []
        for col, val in zip(partition_cols, part_values):
            if pd.isna(val):
                val_str = "__NULL__"
            else:
                val_str = str(val)
            val_str = val_str.strip().replace("/", "_")
            partition_parts.append(f"{col}={val_str}")

        partition_prefix = base_prefix + "/".join(partition_parts) + "/"
        filename = (
            f"part-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}.parquet"
        )
        key = partition_prefix + filename

        try:
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
        except Exception as e:
            errors.append(f"Failed writing partition {partition_parts}: {str(e)}")

    return {
        "success": len(errors) == 0,
        "dataset": dataset_name,
        "rows": len(df),
        "files_written": files_written,
        "deleted_existing_objects": deleted,
        "errors": errors[:10],
    }


def export_dim_sessions(limit: Optional[int] = None) -> Dict[str, Any]:
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(get_table_name())

    items: List[Dict[str, Any]] = []
    scan_kwargs: Dict[str, Any] = {}
    if limit:
        scan_kwargs["Limit"] = limit

    resp = table.scan(**scan_kwargs)
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        if limit and len(items) >= limit:
            items = items[:limit]
            break

    if not items:
        return {"success": True, "message": "No DynamoDB items found", "rows": 0}

    rows: List[Dict[str, Any]] = []
    for it in items:
        event_year = it.get(EVENT_YEAR)
        event_name = it.get(EVENT_NAME)
        session = it.get(SORT_KEY)
        if event_year is None or not event_name or not session:
            continue

        rows.append(
            {
                "event_year": int(event_year) if str(event_year).isdigit() else event_year,
                "event_number": it.get(EVENT_NUMBER),
                "event_name": event_name,
                "event": normalize_event_name(event_name),
                "event_partition_key": it.get(PARTITION_KEY),
                "session": session,
                "session_name": it.get(SESSION_NAME),
                "session_date_utc": it.get(SESSION_DATE_UTC),
                "event_format": it.get(EVENT_FORMAT),
                "is_before_race": it.get(IS_BEFORE_RACE),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return {"success": True, "message": "No valid rows after filtering", "rows": 0}

    # Types
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce").astype("Int64")
    if "event_number" in df.columns:
        df["event_number"] = pd.to_numeric(df["event_number"], errors="coerce").astype("Int64")
    if "is_before_race" in df.columns:
        df["is_before_race"] = df["is_before_race"].astype("boolean")

    result = write_partitioned_parquet_dataset_to_s3(
        df,
        dataset_name="dim_sessions",
        partition_cols=["event_year"],
    )
    return {"success": result.get("success", False), "export": result, "rows": len(df)}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        body = None
        if event.get("body"):
            body = json.loads(event["body"])

        limit = body.get("limit") if body else None
        out = export_dim_sessions(limit=limit)
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(out)}
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"success": False, "error": str(e)}),
        }


"""
Verify that historical qualifying weather hourly aligns with qualifying session time.
Reads bronze JSON for up to 5 events, compares qualifying_dt_utc to qualifying_hour_timestamp_utc
and qualifying_hour_offset_minutes. Expects |offset| <= 30 minutes (hourly bucket).
Usage: cd src/fetch_historical_weather && python verify_qualifying_hour_alignment.py [N]
  N = number of events (default 5)
"""
import json
import os
import sys
from datetime import datetime, timezone

# Allow importing weather_to_silver for get_q_rows and read_bronze
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)
sys.path.insert(0, _here)

from dotenv import load_dotenv
load_dotenv()

import boto3

# Config
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
MAX_OFFSET_MINUTES = 30  # hourly data: closest hour is at most 30 min away


def _parse_utc(s: str) -> datetime:
    if not s:
        return None
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main():
    n = 5
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass

    from weather_to_silver.lambda_function import (
        get_q_rows_with_historical_qualifying_weather,
        read_bronze_forecast_json,
    )

    table_name = DYNAMODB_TABLE_NAME
    bucket = S3_BUCKET_NAME
    rows = get_q_rows_with_historical_qualifying_weather(table_name, limit=n)
    if not rows:
        print("No Q rows with historical qualifying weather found in DynamoDB.")
        sys.exit(1)

    print(f"Verifying qualifying-hour alignment for {len(rows)} event(s).\n")
    print(f"{'Event':<40} {'Qualifying (UTC)':<22} {'Hour selected':<22} {'Offset (min)':<12} {'Aligned?'}")
    print("-" * 110)

    all_ok = True
    for r in rows:
        pk = r.get("event_partition_key", "")
        s3_key = r.get("s3_key", "")
        try:
            payload = read_bronze_forecast_json(bucket, s3_key)
        except Exception as e:
            print(f"{pk:<40} ERROR: {e}")
            all_ok = False
            continue

        meta = payload.get("metadata", {})
        q_hour = payload.get("qualifying_hour_features", {})

        q_dt_str = meta.get("qualifying_dt_utc")
        hour_ts_str = q_hour.get("qualifying_hour_timestamp_utc")
        stored_offset = q_hour.get("qualifying_hour_offset_minutes")

        q_dt = _parse_utc(q_dt_str) if q_dt_str else None
        hour_dt = _parse_utc(hour_ts_str) if hour_ts_str else None

        if q_dt is None or hour_dt is None:
            print(f"{pk:<40} missing qualifying_dt_utc or qualifying_hour_timestamp_utc")
            all_ok = False
            continue

        computed_offset_min = int((hour_dt - q_dt).total_seconds() / 60.0)
        if stored_offset is not None and computed_offset_min != stored_offset:
            print(f"{pk:<40} offset mismatch: stored={stored_offset} computed={computed_offset_min}")
            all_ok = False

        in_range = abs(computed_offset_min) <= MAX_OFFSET_MINUTES
        if not in_range:
            all_ok = False

        aligned = "Yes" if in_range else "No"
        q_fmt = q_dt.strftime("%Y-%m-%d %H:%M") if q_dt else "?"
        h_fmt = hour_dt.strftime("%Y-%m-%d %H:%M") if hour_dt else "?"
        print(f"{pk:<40} {q_fmt:<22} {h_fmt:<22} {computed_offset_min:<12} {aligned}")

    print("-" * 110)
    if all_ok:
        print("All events: qualifying hour aligns with session time (|offset| <= 30 min).")
    else:
        print("Some events had alignment issues or errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

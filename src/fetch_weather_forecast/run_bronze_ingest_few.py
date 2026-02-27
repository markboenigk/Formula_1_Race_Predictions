"""
Run fetch_weather_forecast (bronze ingest) with dry_run=False.
Writes to S3 and updates DynamoDB so silver can pick them up.

Usage: cd src/fetch_weather_forecast && python run_bronze_ingest_few.py [N|all]
  N = number of past events (default 5)
  all = backfill all past events (2022–2025)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
from lambda_function import (
    PARTITION_KEY,
    _build_event_windows,
    _scan_candidate_sessions,
    lambda_handler,
)

load_dotenv()
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
SLEEP_SECONDS = float(os.environ.get("BRONZE_INGEST_SLEEP", "1.0"))


class MockContext:
    def __init__(self):
        self.function_name = "local_lambda"
        self.memory_limit_in_mb = 128
        self.invoked_function_arn = "arn:aws:lambda:local:0:function:local_lambda"
        self.aws_request_id = "local-request-id"


def main():
    arg = (sys.argv[1] or "").strip().lower() if len(sys.argv) > 1 else "5"
    if arg == "all":
        n = None
    else:
        try:
            n = int(arg)
        except ValueError:
            n = 5
    print("Fetching candidate sessions from DynamoDB...")
    items = _scan_candidate_sessions(DYNAMODB_TABLE_NAME, event_partition_key=None)
    windows = _build_event_windows(items)
    if not windows:
        print("No event windows found.")
        sys.exit(1)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    past = [w for w in windows if w["race_dt_utc"] < now]
    chosen = past if n is None else (past[:n] if len(past) >= n else past) or windows[:n]
    print(f"Running bronze ingest (dry_run=False) for {len(chosen)} events (sleep={SLEEP_SECONDS}s between calls).\n")
    ok = 0
    err = 0
    for i, w in enumerate(chosen):
        if i > 0 and SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
        pk = w[PARTITION_KEY]
        print(f"[{i+1}/{len(chosen)}] {pk} ... ", end="", flush=True)
        event = {
            "httpMethod": "POST",
            "path": "/forecast",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({PARTITION_KEY: pk, "dry_run": False}),
        }
        try:
            resp = lambda_handler(event, MockContext())
            status = resp.get("statusCode", 0)
            if status == 200:
                body = json.loads(resp.get("body", "{}")) if isinstance(resp.get("body"), str) else resp.get("body", {})
                s3_key = body.get("s3_key", "?")
                print(f"OK -> {s3_key}")
                ok += 1
            else:
                print(f"HTTP {status}: {resp.get('body', '')}")
                err += 1
        except Exception as e:
            print(f"ERROR: {e}")
            err += 1
    print(f"\nDone: {ok} OK, {err} errors. Run weather_to_silver to consolidate to silver.")


if __name__ == "__main__":
    main()

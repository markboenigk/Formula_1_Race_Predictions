"""
Run fetch_historical_weather (bronze ingest) for 2022-2025 events, then run silver.
Writes to S3 and updates DynamoDB Q row; silver step writes weather_historical_qualifying.

Usage: cd src/fetch_historical_weather && python run_bronze_ingest_few.py [N|all]
  N = number of events (default 5)
  all = backfill all 2022-2025 events
"""
import json
import os
import sys
import time

# Allow importing from sibling weather_to_silver
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)
sys.path.insert(0, _here)

from dotenv import load_dotenv
from lambda_function import (
    PARTITION_KEY,
    _build_qualifying_windows,
    _scan_candidate_q_sessions,
    lambda_handler,
)

load_dotenv()
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
SLEEP_SECONDS = float(os.environ.get("BRONZE_INGEST_SLEEP", "1.0"))
BACKFILL_YEAR_MIN = 2022
BACKFILL_YEAR_MAX = 2025


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

    print("Fetching candidate Q sessions from DynamoDB...")
    items = _scan_candidate_q_sessions(DYNAMODB_TABLE_NAME, event_partition_key=None)
    windows = _build_qualifying_windows(items)
    if not windows:
        print("No qualifying windows found.")
        sys.exit(1)

    # Restrict to 2022-2025 for backfill
    event_year_key = "event_year"
    windows = [w for w in windows if BACKFILL_YEAR_MIN <= w.get(event_year_key, 0) <= BACKFILL_YEAR_MAX]
    if not windows:
        print(f"No events in {BACKFILL_YEAR_MIN}-{BACKFILL_YEAR_MAX}.")
        sys.exit(1)

    chosen = windows if n is None else windows[:n]
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
            "path": "/historical-qualifying-weather",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({PARTITION_KEY: pk, "dry_run": False}),
        }
        try:
            resp = lambda_handler(event, MockContext())
            status = resp.get("statusCode", 0)
            if status == 200:
                body = json.loads(resp.get("body", "{}")) if isinstance(resp.get("body"), str) else resp.get("body", {})
                results = body.get("results", [])
                s3_key = results[0].get("s3_key", "?") if results else "?"
                print(f"OK -> {s3_key}")
                ok += 1
            else:
                print(f"HTTP {status}: {resp.get('body', '')}")
                err += 1
        except Exception as e:
            print(f"ERROR: {e}")
            err += 1

    print(f"\nBronze done: {ok} OK, {err} errors.")

    # Run silver for historical qualifying
    print("Running silver (weather_historical_qualifying)...")
    try:
        from weather_to_silver.lambda_function import run as weather_to_silver_run
        result = weather_to_silver_run(dataset="historical_qualifying")
        if result.get("success"):
            print(f"Silver done: {result.get('rows_written', 0)} rows written.")
        else:
            print(f"Silver failed: {result}")
    except Exception as e:
        print(f"Silver step failed: {e}. Run weather_to_silver with dataset=historical_qualifying manually.")


if __name__ == "__main__":
    main()

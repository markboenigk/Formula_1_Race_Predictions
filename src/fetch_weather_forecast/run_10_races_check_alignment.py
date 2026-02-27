"""
Run the fetch_weather_forecast Lambda (dry run) for 10 races and verify that
race_hour_features align with the actual race start time.

Usage (from repo root with .venv active):
  cd src/fetch_weather_forecast && python run_10_races_check_alignment.py
"""
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple

# Run from fetch_weather_forecast directory so lambda_function is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from lambda_function import (
    PARTITION_KEY,
    _build_event_windows,
    _parse_utc_iso,
    _scan_candidate_sessions,
    lambda_handler,
)

load_dotenv()

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")


class MockContext:
    def __init__(self):
        self.function_name = "local_lambda"
        self.memory_limit_in_mb = 128
        self.invoked_function_arn = "arn:aws:lambda:local:0:function:local_lambda"
        self.aws_request_id = "local-request-id"


def parse_race_hour_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def is_aligned(
    race_dt_utc: datetime,
    race_hour_timestamp_utc: Optional[str],
    race_hour_offset_minutes: Optional[int],
) -> Tuple[bool, str]:
    """Check if race_hour_features align with race time. Returns (aligned, message)."""
    if not race_hour_timestamp_utc:
        return False, "missing race_hour_timestamp_utc"
    hour_dt = parse_race_hour_ts(race_hour_timestamp_utc)
    if hour_dt is None:
        return False, f"could not parse race_hour_timestamp_utc: {race_hour_timestamp_utc!r}"
    # Selected hour should be the one containing (or closest to) race start: same date and same hour.
    if hour_dt.date() != race_dt_utc.date():
        return False, f"date mismatch: race_hour={hour_dt.date()} race_dt={race_dt_utc.date()}"
    if hour_dt.hour != race_dt_utc.hour:
        return False, f"hour mismatch: race_hour={hour_dt.hour} race_dt={race_dt_utc.hour}"
    # offset_minutes: 0 = race exactly at that hour start; small = race a few min into the hour
    offset = race_hour_offset_minutes if race_hour_offset_minutes is not None else 0
    if abs(offset) > 60:
        return False, f"offset too large: race_hour_offset_minutes={offset}"
    return True, f"OK (offset={offset} min)"


def main() -> None:
    print("Fetching candidate sessions from DynamoDB...")
    items = _scan_candidate_sessions(DYNAMODB_TABLE_NAME, event_partition_key=None)
    windows = _build_event_windows(items)
    if not windows:
        print("No event windows found (need Q and R with session_date_utc and circuit lat/lng).")
        sys.exit(1)
    # Prefer past events (historical forecast) so we don't depend on 16-day window
    now = datetime.now(timezone.utc)
    past = [w for w in windows if w["race_dt_utc"] < now]
    if len(past) >= 10:
        chosen = past[:10]
    else:
        chosen = windows[:10]
    print(f"Selected {len(chosen)} events to run.\n")

    results = []
    for i, window in enumerate(chosen):
        pk = window[PARTITION_KEY]
        race_dt_utc = window["race_dt_utc"]
        event_name = window.get("EventName", pk)
        print(f"[{i+1}/{len(chosen)}] {pk} ({race_dt_utc.strftime('%Y-%m-%d %H:%M')}Z) ... ", end="", flush=True)
        event = {
            "httpMethod": "POST",
            "path": "/forecast",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({PARTITION_KEY: pk, "dry_run": True}),
        }
        try:
            response = lambda_handler(event, MockContext())
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "event": pk,
                "race_dt_utc": race_dt_utc.isoformat(),
                "aligned": False,
                "message": str(e),
                "race_hour_timestamp_utc": None,
                "race_hour_offset_minutes": None,
            })
            continue
        status = response.get("statusCode", 0)
        if status != 200:
            body = response.get("body", "{}")
            try:
                body = json.loads(body) if isinstance(body, str) else body
            except Exception:
                pass
            err = body.get("error", body) if isinstance(body, dict) else body
            print(f"HTTP {status}: {err}")
            results.append({
                "event": pk,
                "race_dt_utc": race_dt_utc.isoformat(),
                "aligned": False,
                "message": str(err),
                "race_hour_timestamp_utc": None,
                "race_hour_offset_minutes": None,
            })
            continue
        body = response.get("body", "{}")
        try:
            data = json.loads(body) if isinstance(body, str) else body
        except Exception:
            data = {}
        rhr = data.get("race_hour_features") or {}
        race_hour_ts = rhr.get("race_hour_timestamp_utc")
        offset_min = rhr.get("race_hour_offset_minutes")
        aligned, msg = is_aligned(race_dt_utc, race_hour_ts, offset_min)
        results.append({
            "event": pk,
            "event_name": event_name,
            "race_dt_utc": race_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "race_hour_timestamp_utc": race_hour_ts,
            "race_hour_offset_minutes": offset_min,
            "aligned": aligned,
            "message": msg,
        })
        print(msg)

    # Summary
    aligned_count = sum(1 for r in results if r.get("aligned"))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Races run: {len(results)}")
    print(f"Aligned (race_hour matches race time): {aligned_count}/{len(results)}")
    if aligned_count < len(results):
        print("\nNot aligned:")
        for r in results:
            if not r.get("aligned"):
                print(f"  - {r['event']}: {r.get('message', '')}")
    print("\nPer-event details:")
    for r in results:
        print(f"  {r['event']}: race_dt_utc={r['race_dt_utc']} -> race_hour_ts={r.get('race_hour_timestamp_utc')} offset_min={r.get('race_hour_offset_minutes')} | {r.get('message', '')}")


if __name__ == "__main__":
    main()

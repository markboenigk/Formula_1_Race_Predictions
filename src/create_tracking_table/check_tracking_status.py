"""
Report which events/sessions in the DynamoDB tracking table have been ingested (data_loaded=true)
vs not. Uses the same table name and schema as the create_tracking_table lambda.
"""
import os
from collections import defaultdict

import boto3
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")
PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"
EVENT_YEAR = "event_year"
EVENT_NAME = "EventName"
DATA_LOADED = "data_loaded"


def main():
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(TABLE_NAME)

    items = []
    scan_kwargs = {}
    resp = table.scan(**scan_kwargs)
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))

    # By year: total sessions, loaded sessions, distinct events
    by_year = defaultdict(lambda: {"total": 0, "loaded": 0, "events": set()})
    # Per event (event_partition_key): total sessions, loaded sessions
    by_event = defaultdict(lambda: {"total": 0, "loaded": 0, "year": None, "name": None})

    for it in items:
        year = it.get(EVENT_YEAR)
        if year is not None:
            try:
                y = int(year)
            except (TypeError, ValueError):
                y = year
        else:
            y = "unknown"
        by_year[y]["total"] += 1
        if it.get(DATA_LOADED) is True:
            by_year[y]["loaded"] += 1
        pk = it.get(PARTITION_KEY) or ""
        if pk:
            by_year[y]["events"].add(pk)
        by_event[pk]["total"] += 1
        if it.get(DATA_LOADED) is True:
            by_event[pk]["loaded"] += 1
        if by_event[pk]["year"] is None:
            by_event[pk]["year"] = year
            by_event[pk]["name"] = it.get(EVENT_NAME)

    print(f"Table: {TABLE_NAME}")
    print(f"Total session rows: {len(items)}")
    print()
    print("By year:")
    for y in sorted(by_year.keys(), key=lambda x: (x if isinstance(x, int) else 0)):
        info = by_year[y]
        pct = (info["loaded"] / info["total"] * 100) if info["total"] else 0
        print(f"  {y}: {info['loaded']}/{info['total']} sessions loaded ({pct:.0f}%), {len(info['events'])} events")
    print()
    print("By event (events with any session not loaded):")
    not_fully_loaded = [
        (pk, by_event[pk])
        for pk in sorted(by_event.keys())
        if by_event[pk]["loaded"] < by_event[pk]["total"]
    ]
    for pk, info in not_fully_loaded[:40]:
        print(f"  {info['year']} {pk}: {info['loaded']}/{info['total']} loaded")
    if len(not_fully_loaded) > 40:
        print(f"  ... and {len(not_fully_loaded) - 40} more")
    print()
    print("By event (fully loaded):")
    fully_loaded = [
        (pk, by_event[pk])
        for pk in sorted(by_event.keys())
        if by_event[pk]["loaded"] == by_event[pk]["total"] and by_event[pk]["total"] > 0
    ]
    for pk, info in fully_loaded[:30]:
        print(f"  {info['year']} {pk}: {info['loaded']}/{info['total']} loaded")
    if len(fully_loaded) > 30:
        print(f"  ... and {len(fully_loaded) - 30} more")


if __name__ == "__main__":
    main()

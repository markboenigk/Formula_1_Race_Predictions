"""
Run the inference Lambda handler locally.

Uses test_payload.json (event_partition_key). Ensure DynamoDB has a race (R)
row for that event so date/circuit can be resolved.

Run from repo root: PYTHONPATH=. python src/inference/run_local.py
Or from here: cd src/inference && PYTHONPATH=../.. python run_local.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from lambda_function import lambda_handler


# Resolve test_payload.json relative to this script's directory
SCRIPT_DIR = Path(__file__).parent


class MockContext:
    def __init__(self):
        self.function_name = "inference_local"
        self.memory_limit_in_mb = 256
        self.invoked_function_arn = "arn:aws:lambda:local:0:function:inference"
        self.aws_request_id = "local-inference-request-id"


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("DYNAMODB_TABLE_NAME"):
        print("ℹ️  DYNAMODB_TABLE_NAME not set, using default: f1_session_tracking")
        print()

    payload_file = SCRIPT_DIR / "test_payload.json"
    if payload_file.exists():
        with open(payload_file, "r") as f:
            payload_data = json.load(f)
    else:
        payload_data = {"event_partition_key": "2025_monaco"}
        print("ℹ️  No test_payload.json; using example event_partition_key.")
        print()

    event = {
        "httpMethod": "POST",
        "path": "/inference",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload_data) if payload_data else None,
    }
    context = MockContext()

    print("=" * 60)
    print("Inference Lambda (local)")
    print("=" * 60)
    if payload_data:
        print(f"Payload: {json.dumps(payload_data, indent=2)}")
    print("=" * 60)

    response = lambda_handler(event, context)
    print("\nResponse:")
    print(json.dumps(response, indent=2))

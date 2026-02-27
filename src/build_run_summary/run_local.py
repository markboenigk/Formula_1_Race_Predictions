"""
Run the build_run_summary Lambda handler locally.

Reads payload from test_payload.json and invokes lambda_handler.
"""
import json
import os

from dotenv import load_dotenv

from lambda_function import lambda_handler


class MockContext:
    def __init__(self):
        self.function_name = "local_build_run_summary"
        self.memory_limit_in_mb = 256
        self.invoked_function_arn = "arn:aws:lambda:local:0:function:build_run_summary"
        self.aws_request_id = "local-request-id"


if __name__ == "__main__":
    load_dotenv()

    payload_file = "test_payload.json"
    payload_data = {}
    if os.path.exists(payload_file):
        with open(payload_file, "r", encoding="utf-8") as f:
            payload_data = json.load(f)

    event = {
        "httpMethod": "POST",
        "path": "/build-run-summary",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload_data) if payload_data else None,
    }

    response = lambda_handler(event, MockContext())
    print(json.dumps(response, indent=2))

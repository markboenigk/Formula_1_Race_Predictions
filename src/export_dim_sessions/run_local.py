import json
import os

from dotenv import load_dotenv

from lambda_function import lambda_handler


class MockContext:
    def __init__(self):
        self.function_name = "local_lambda"
        self.memory_limit_in_mb = 128
        self.invoked_function_arn = "arn:aws:lambda:local:0:function:local_lambda"
        self.aws_request_id = "local-request-id"


if __name__ == "__main__":
    load_dotenv()

    payload_data = {}
    payload_file = "test_payload.json"
    if os.path.exists(payload_file):
        with open(payload_file, "r", encoding="utf-8") as f:
            payload_data = json.load(f)

    event = {
        "httpMethod": "POST",
        "path": "/export",
        "headers": {"Content-Type": "application/json", "x-api-key": "test-api-key"},
        "body": json.dumps(payload_data) if payload_data else None,
    }

    response = lambda_handler(event, MockContext())
    print(json.dumps(response, indent=2))


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
    # Load environment variables from .env file
    load_dotenv()
    
    # Set environment variables for local testing if not already set
    if not os.getenv("S3_BUCKET_NAME"):
        # Use the same bucket as other Lambda functions

        print(f"ℹ️  S3_BUCKET_NAME not set")
        print()
    
    if not os.getenv("S3_WL_BRONZE_PATH"):
        print(f"ℹ️  S3_WL_BRONZE_PATH not set")
        print()
    
    # Read the JSON payload from file (try test_payload.json first, fallback to test_event.json)
    payload_file = 'test_payload.json'
    if not os.path.exists(payload_file):
        payload_file = 'test_event.json'
    
    if os.path.exists(payload_file):
        with open(payload_file, 'r') as f:
            payload_data = json.load(f)
    else:
        # Use empty payload if no test file exists
        payload_data = {}
        print(f"ℹ️  No test file found (test_payload.json or test_event.json). Using empty payload.")
        print()
    
    # Mock event that simulates API Gateway POST request
    event = {
        "httpMethod": "POST",
        "path": "/sync",
        "headers": {
            "Content-Type": "application/json",
            "x-api-key": "test-api-key"
        },
        "body": json.dumps(payload_data) if payload_data else None  # API Gateway sends body as JSON string
    }
    
    context = MockContext()
    
    print("=" * 60)
    print("Testing Lambda Function Locally")
    print("=" * 60)
    if payload_data:
        print(f"📦 Payload loaded from {payload_file}")
        print()
    
    print("=" * 60)

    response = lambda_handler(event, context)

    print("\nLambda response:")
    print(json.dumps(response, indent=2))

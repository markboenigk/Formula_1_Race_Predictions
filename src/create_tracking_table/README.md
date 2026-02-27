# Create Tracking Table

Creates the DynamoDB tracking table for F1 session data.

## Files

- `lambda_function.py`: AWS Lambda handler
- `run_local.py`: local launcher using `test_payload.json`
- `test_payload.json`: example payload
- `check_tracking_status.py`: utility to check session status

## Purpose

Creates and manages the `f1_session_tracking` DynamoDB table that tracks F1 session data lifecycle (bronze → silver → gold).

## Local run

```bash
source .venv/bin/activate
python src/create_tracking_table/run_local.py
```

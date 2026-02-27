# Download Session Data

Downloads F1 session data using the FastF1 library.

## Files

- `lambda_function.py`: AWS Lambda handler (supports API Gateway and orchestrator invocation)
- `run_local.py`: local launcher using `test_payload.json`
- `test_payload.json`: example payload

## Purpose

Fetches qualifying and race session data from FastF1 and writes to bronze layer in S3.

## Local run

```bash
source .venv/bin/activate
python src/download_session_data/run_local.py
```

# Fetch Historical Weather

Fetches historical weather data from Open-Meteo archive for F1 race days.

## Files

- `lambda_function.py`: AWS Lambda handler
- `run_local.py`: local launcher
- `run_bronze_ingest_few.py`: ingest a few events for testing
- `test_payload.json`: example payload
- `verify_qualifying_hour_alignment.py`: utility to verify data alignment

## Purpose

Fetches archived weather data for past F1 races and writes to bronze layer in S3.

## Local run

```bash
source .venv/bin/activate
python src/fetch_historical_weather/run_local.py
```

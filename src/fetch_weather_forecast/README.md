# Fetch Weather Forecast

Fetches weather forecasts from Open-Meteo for upcoming F1 race weekends.

## Files

- `lambda_function.py`: AWS Lambda handler
- `run_local.py`: local launcher
- `run_bronze_ingest_few.py`: ingest a few events for testing
- `run_10_races_check_alignment.py`: verify data alignment
- `test_payload.json`: example payload

## Purpose

Fetches weather forecast data for upcoming F1 races and writes to bronze layer in S3.

## Local run

```bash
source .venv/bin/activate
python src/fetch_weather_forecast/run_local.py
```

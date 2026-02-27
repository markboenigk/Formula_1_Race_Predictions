# Weather to Silver

Transforms weather data from bronze to silver layer.

## Files

- `lambda_function.py`: AWS Lambda handler
- `run_local.py`: local launcher
- `test_payload.json`: example payload

## Purpose

Transforms raw weather data (bronze) into processed weather data (silver) for the F1 pipeline.

## Local run

```bash
source .venv/bin/activate
python src/weather_to_silver/run_local.py
```

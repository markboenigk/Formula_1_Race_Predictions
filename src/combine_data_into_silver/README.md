# Combine Data into Silver

Combines bronze layer data into silver layer for the F1 data pipeline.

## Files

- `lambda_function.py`: AWS Lambda handler (supports API Gateway and direct invocation)
- `run_local.py`: local launcher using `test_payload.json`
- `test_payload.json`: example payload

## Purpose

Transforms raw bronze layer data into cleaned, processed silver layer data.

## Local run

```bash
source .venv/bin/activate
python src/combine_data_into_silver/run_local.py
```

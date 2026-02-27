# Inference

Runs prediction inference using the trained XGBoost model.

## Files

- `lambda_function.py`: AWS Lambda handler (invoked after qualifying by Step Functions)
- `predict.py`: prediction logic
- `run_local.py`: local launcher
- `test_payload.json`: example payload

## Purpose

Fetches live weather, runs prediction using ONNX model, and writes results to DynamoDB + S3.

## Trigger

Invoked by Step Functions after qualifying session completes.

## Local run

```bash
source .venv/bin/activate
python src/inference/run_local.py
```

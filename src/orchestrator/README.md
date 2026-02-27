# Orchestrator

Pipeline orchestration using AWS Step Functions.

## Files

- `lambda_function.py`: AWS Lambda entry point
- `orchestrator.py`: main orchestration logic

## Purpose

Coordinates the F1 data pipeline: downloads session data → transforms to silver → builds gold features → runs inference.

## Trigger

EventBridge schedule or manual invocation.

## Note

Vendored dependencies moved to `vendor/orchestrator/`

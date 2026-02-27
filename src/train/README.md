# Training (Phase 3 baseline)

This module trains the baseline `grid_only_v1` model and writes model artifacts to S3.

## Files

- `train.py`: training entrypoint.
- `run_local.py`: local launcher reading `test_payload.json`.
- `test_payload.json`: local run configuration.
- `requirements.txt`: training dependencies.

## Artifact layout

Artifacts are written to:

- `s3://<bucket>/models/run_id=<timestamp>/xgb_model.ubj`
- `s3://<bucket>/models/run_id=<timestamp>/preprocessor.joblib`
- `s3://<bucket>/models/run_id=<timestamp>/model_metadata.json`
- `s3://<bucket>/models/run_id=<timestamp>/_SUCCESS`
- `s3://<bucket>/models/run_id=<timestamp>/predictions_vs_actuals/event_year=<YYYY>/event=<slug>/predictions_vs_actuals.parquet`
- `s3://<bucket>/models/_LATEST`

## Local run

```bash
source .venv/bin/activate
python src/train/run_local.py
```

## Notes

- Gold dataset source: `gold/gold_driver_event_features/run_id=<gold_run_id>/training_dataset.parquet`
- If `gold_run_id` is omitted, `train.py` resolves the latest run from S3.
- Baseline feature is `qualifying_position` (fallback: `grid_starting_position`).

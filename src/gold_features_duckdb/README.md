# Gold Features (DuckDB)

Builds gold layer features using DuckDB for ML training.

## Files

- `build_gold_features.py`: main script to build gold features
- `gold_driver_event_features.sql`: SQL for driver/event features
- `gold_training_dataset.sql`: SQL for training dataset
- `validate_rolling_features.py`: validation utility

## Purpose

Transforms silver layer data into gold layer features for model training using DuckDB with S3 integration.

## Local run

```bash
source .venv/bin/activate
python src/gold_features_duckdb/build_gold_features.py
```

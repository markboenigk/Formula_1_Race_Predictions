"""
Weather utilities for the F1 pipeline.

- forecast: inference-time live forecast (fetch_race_forecast). Used by the
  inference Lambda; not for storage. Package this with the inference Lambda.
- historical_backfill: local script for ingestion backfill (bronze/silver).
  Run as: python -m src.weather.historical_backfill. Not a Lambda.
"""

# Weather Utilities

Shared weather utilities for the F1 pipeline.

## Files

- `historical_backfill.py`: Script to fetch historical weather from Open-Meteo archive
- `forecast.py`: Weather forecast utilities
- `__init__.py`: Package init

## Purpose

Provides weather data fetching and processing utilities used by other modules.

## Local run

```bash
source .venv/bin/activate
python -m src.weather.historical_backfill
```

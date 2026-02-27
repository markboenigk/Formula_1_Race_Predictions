# Common Utilities

Shared utilities used across the F1 pipeline.

## Files

- `circuit_coordinates.py`: Circuit lat/lng coordinates and event name normalization
- `delete_table.py`: Utility to delete DynamoDB tables
- `recreate_table.py`: Utility to recreate DynamoDB tables
- `__init__.py`: Package init

## Purpose

Provides single source of truth for:
- `normalize_event_name()` - consistent event slug creation for S3 paths
- `CIRCUIT_COORDINATES` - lat/lng for all 2022-2025 calendar circuits

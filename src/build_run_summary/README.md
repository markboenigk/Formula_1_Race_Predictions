# Build Run Summary (serverless dashboard)

This Lambda discovers all training runs under `s3://<bucket>/<models_prefix>/run_id=*/`, reads each `model_metadata.json`, and writes a single **run summary** as JSON and HTML to S3. View the HTML in a browser (e.g. S3 Object URL or behind CloudFront) for a serverless “dashboard” with no extra server.

## Files

- `lambda_function.py`: handler and helpers
- `run_local.py`: local launcher using `test_payload.json`
- `test_payload.json`: example payload (optional bucket/prefix overrides)

## Output

- `s3://<bucket>/<reports_prefix>/run_summary.json` – machine-readable list of runs and metrics
- `s3://<bucket>/<reports_prefix>/run_summary.html` – simple HTML table (run_id, trained_at_utc, run_type, baseline_name, metrics, train_rows, season_range, git commit)

## Env (optional)

- `S3_BUCKET_NAME` – default `f1-race-prediction`
- `S3_MODEL_PATH` – default `models`
- `S3_REPORTS_PATH` – default `reports`

## Trigger

- **EventBridge**: after the Step Functions state machine completes (training + inference), trigger this Lambda so the report is updated automatically.
- **S3 event**: optional notification on `s3://<bucket>/<models_prefix>/` when `model_metadata.json` or `_SUCCESS` is written.
- **On-demand**: invoke with an empty body or with `{"bucket","models_prefix","reports_prefix"}` overrides.

## Local run

```bash
source .venv/bin/activate
python src/build_run_summary/run_local.py
```

Uses `test_payload.json` in the same directory; override bucket/prefixes there if needed.

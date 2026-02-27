"""
Build Run Summary Lambda: discover model runs from S3, aggregate model_metadata.json
into a summary, and write JSON + HTML report to S3 for a serverless dashboard.

Trigger: EventBridge (after training) or S3 event on models/ prefix, or on-demand.
Output: s3://<bucket>/<reports_prefix>/run_summary.json, run_summary.html
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3


S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_MODELS_PREFIX = os.environ.get("S3_MODEL_PATH", "models").rstrip("/")
S3_REPORTS_PREFIX = os.environ.get("S3_REPORTS_PATH", "reports").rstrip("/")
RUN_SUMMARY_JSON_KEY = f"{S3_REPORTS_PREFIX}/run_summary.json"
RUN_SUMMARY_HTML_KEY = f"{S3_REPORTS_PREFIX}/run_summary.html"


def list_run_ids(s3_client, bucket: str, models_prefix: str) -> List[str]:
    """List all run_id values under s3://bucket/models_prefix/run_id=*."""
    prefix = f"{models_prefix}/"
    marker = "run_id="
    run_ids: set[str] = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if marker not in key:
                continue
            rest = key.split(marker, 1)[1]
            run_id = rest.split("/", 1)[0]
            if run_id:
                run_ids.add(run_id)
    return sorted(run_ids, reverse=True)


def fetch_metadata(s3_client, bucket: str, models_prefix: str, run_id: str) -> Dict[str, Any] | None:
    """Load model_metadata.json for a run. Returns None if missing or invalid."""
    key = f"{models_prefix}/run_id={run_id}/model_metadata.json"
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read().decode("utf-8")
        return json.loads(body)
    except (s3_client.exceptions.NoSuchKey, json.JSONDecodeError, KeyError):
        return None


def to_summary_row(meta: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Extract a flat row for the summary table."""
    metrics = meta.get("metrics") or {}
    return {
        "run_id": run_id,
        "trained_at_utc": meta.get("trained_at_utc", ""),
        "run_type": meta.get("run_type", ""),
        "baseline_name": meta.get("baseline_name", ""),
        "feature_set_name": meta.get("feature_set_name", ""),
        "dataset_ref": meta.get("dataset_ref", ""),
        "precision_at_3": metrics.get("precision_at_3"),
        "recall_at_3": metrics.get("recall_at_3"),
        "spearman": metrics.get("spearman"),
        "train_rows": meta.get("train_rows"),
        "season_range": meta.get("season_range"),
        "git_commit": (meta.get("git_commit") or "")[:8] if meta.get("git_commit") else "",
        "xgboost_version": meta.get("xgboost_version", ""),
    }


def build_summary_json(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Structure for run_summary.json."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "count": len(runs),
    }


def build_summary_html(runs: List[Dict[str, Any]], bucket: str, reports_prefix: str) -> str:
    """Simple HTML table for viewing in browser (e.g. S3 Object URL or CloudFront)."""
    rows_html = []
    for r in runs:
        precision = r.get("precision_at_3")
        recall = r.get("recall_at_3")
        spearman = r.get("spearman")
        rows_html.append(
            f"""
    <tr>
      <td>{_h(r.get("run_id", ""))}</td>
      <td>{_h(r.get("trained_at_utc", ""))}</td>
      <td>{_h(r.get("run_type", ""))}</td>
      <td>{_h(r.get("baseline_name", ""))}</td>
      <td>{_h(r.get("feature_set_name", ""))}</td>
      <td>{_fmt(precision)}</td>
      <td>{_fmt(recall)}</td>
      <td>{_fmt(spearman)}</td>
      <td>{_h(r.get("train_rows"))}</td>
      <td>{_h(r.get("season_range"))}</td>
      <td>{_h(r.get("git_commit", ""))}</td>
    </tr>"""
        )

    table_body = "\n".join(rows_html) if rows_html else "<tr><td colspan=\"11\">No runs found.</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>F1 Model Run Summary</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #0f0f0f; color: #e0e0e0; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 0.5rem; }}
    p.meta {{ color: #888; font-size: 0.875rem; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.8125rem; }}
    th, td {{ border: 1px solid #333; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #1a1a1a; color: #aaa; font-weight: 600; }}
    tr:nth-child(even) {{ background: #161616; }}
    a {{ color: #6af; }}
    .num {{ text-align: right; }}
  </style>
</head>
<body>
  <h1>F1 Model Run Summary</h1>
  <p class="meta">Bucket: {_h(bucket)} · Prefix: {_h(reports_prefix)} · {len(runs)} run(s)</p>
  <table>
    <thead>
      <tr>
        <th>run_id</th>
        <th>trained_at_utc</th>
        <th>run_type</th>
        <th>baseline_name</th>
        <th>feature_set</th>
        <th class="num">precision@3</th>
        <th class="num">recall@3</th>
        <th class="num">spearman</th>
        <th class="num">train_rows</th>
        <th>season_range</th>
        <th>git</th>
      </tr>
    </thead>
    <tbody>
{table_body}
    </tbody>
  </table>
</body>
</html>"""


def _h(x: Any) -> str:
    """Escape and stringify for HTML."""
    if x is None:
        return ""
    s = str(x)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(x: Any) -> str:
    """Format number for display; empty if None."""
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return _h(x)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Build run summary from S3 model metadata and write JSON + HTML to S3.

    Optional event body: { "bucket": "...", "models_prefix": "...", "reports_prefix": "..." }
    """
    try:
        body = event.get("body")
        if isinstance(body, str) and body.strip():
            payload = json.loads(body)
        elif isinstance(body, dict):
            payload = body
        else:
            payload = {}

        bucket = payload.get("bucket") or S3_BUCKET_NAME
        models_prefix = payload.get("models_prefix") or S3_MODELS_PREFIX
        reports_prefix = payload.get("reports_prefix") or S3_REPORTS_PREFIX
        json_key = f"{reports_prefix}/run_summary.json"
        html_key = f"{reports_prefix}/run_summary.html"

        s3 = boto3.client("s3")
        run_ids = list_run_ids(s3, bucket, models_prefix)
        runs: List[Dict[str, Any]] = []
        for run_id in run_ids:
            meta = fetch_metadata(s3, bucket, models_prefix, run_id)
            if meta:
                runs.append(to_summary_row(meta, run_id))

        summary = build_summary_json(runs)

        s3.put_object(
            Bucket=bucket,
            Key=json_key,
            Body=json.dumps(summary, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        html = build_summary_html(runs, bucket, reports_prefix)
        s3.put_object(
            Bucket=bucket,
            Key=html_key,
            Body=html.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Run summary written to S3",
                    "bucket": bucket,
                    "json_key": json_key,
                    "html_key": html_key,
                    "runs_count": len(runs),
                },
                indent=2,
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }

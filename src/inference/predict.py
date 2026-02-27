#!/usr/bin/env python3
"""
Fargate Inference Script

Loads ONNX model from S3, reads qualifying results, runs predictions,
writes to DynamoDB and S3.

Usage:
    python -m src.inference.predict --event-key "2025_Australian_Grand_Prix"

Or with environment variables:
    S3_BUCKET=f1-race-prediction python -m src.inference.predict --event-key "2025_Australian_Grand_Prix"
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import boto3
import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

METRICS_NAMESPACE = os.environ.get("METRICS_NAMESPACE", "F1/RacePrediction")


def log_json(level: str, event: str, **kwargs):
    """Structured JSON logging for CloudWatch Logs."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "service": "f1-inference-fargate",
        "event": event,
    }
    log_entry.update(kwargs)
    print(json.dumps(log_entry))


def put_metric(metric_name: str, value: float, unit: str = "Milliseconds", session_type: str = None):
    """Emit CloudWatch metric."""
    import json
    cloudwatch = boto3.client("cloudwatch")
    dimensions = []
    if session_type:
        dimensions.append({"Name": "SessionType", "Value": session_type})
    try:
        cloudwatch.put_metric_data(
            Namespace=METRICS_NAMESPACE,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": dimensions,
                }
            ],
        )
    except Exception as e:
        log_json("WARNING", "metric_emit_failed", error=str(e))

S3_BUCKET = os.environ.get("S3_BUCKET", "f1-race-prediction")
S3_MODELS_PREFIX = os.environ.get("S3_MODELS_PREFIX", "models")
S3_SILVER_PREFIX = os.environ.get("S3_SILVER_PREFIX", "silver")
PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "f1_predictions")
PREDICTIONS_S3_PREFIX = os.environ.get("PREDICTIONS_S3_PREFIX", "predictions")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "f1_session_tracking")

QUALIFYING_ABR = "Q"
SPRINT_QUALIFYING_ABR = "SQ"  # 2024+ format
SPRINT_SHOOTOUT_ABR = "SS"    # 2023 format
SPRINT_ABR = "S"  # Sprint race
RACE_ABR = "R"
PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"


def load_model_from_s3() -> Tuple[ort.InferenceSession, Any]:
    """Load ONNX model and preprocessor from S3."""
    s3 = boto3.client("s3")

    # Read _LATEST pointer
    latest_key = f"{S3_MODELS_PREFIX}/_LATEST"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=latest_key)
    run_id = resp["Body"].read().decode("utf-8").strip()
    print(f"Loading model from run_id={run_id}")

    # Check _SUCCESS sentinel
    success_key = f"{S3_MODELS_PREFIX}/run_id={run_id}/_SUCCESS"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=success_key)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise RuntimeError(f"Model run {run_id} missing _SUCCESS sentinel")
        raise

    # Download ONNX model
    model_key = f"{S3_MODELS_PREFIX}/run_id={run_id}/model.onnx"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=model_key)
    model_bytes = resp["Body"].read()
    model = ort.InferenceSession(model_bytes)

    # Download preprocessor
    preprocessor_key = f"{S3_MODELS_PREFIX}/run_id={run_id}/preprocessor.joblib"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=preprocessor_key)
    preprocessor = joblib.load(io.BytesIO(resp["Body"].read()))

    return model, preprocessor


def get_session_results(year: int, event_slug: str, session_type: str = "qualifying") -> pd.DataFrame:
    """Fetch session results from silver S3. session_type: 'qualifying', 'sprint_qualifying', 'sprint_shootout', or 'sprint'."""
    # Map session type to abbreviation
    if session_type == "sprint_qualifying":
        session_abr = SPRINT_QUALIFYING_ABR  # SQ
    elif session_type == "sprint_shootout":
        session_abr = SPRINT_SHOOTOUT_ABR    # SS
    elif session_type == "sprint":
        session_abr = SPRINT_ABR              # S (sprint race)
    else:
        session_abr = QUALIFYING_ABR          # Q
    prefix = f"{S3_SILVER_PREFIX}/results/event_year={year}/event={event_slug}/session={session_abr}/"
    s3 = boto3.client("s3")

    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    if "Contents" not in resp:
        raise ValueError(f"No qualifying results at s3://{S3_BUCKET}/{prefix}")

    dfs = []
    for obj in resp["Contents"]:
        key = obj["Key"]
        if not key.endswith(".parquet"):
            continue
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No parquet files at s3://{S3_BUCKET}/{prefix}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.rename(columns={"finishing_position": "qualifying_position"})
    return combined[["driver_number", "driver_id", "full_name", "qualifying_position"]].copy()


def run_prediction(model: ort.InferenceSession, preprocessor: Any, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """Run ONNX model prediction."""
    df = qualifying_df.dropna(subset=["qualifying_position"]).copy()
    df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")
    df = df.dropna(subset=["qualifying_position"])

    if df.empty:
        raise ValueError("No valid qualifying positions")

    X = df[["qualifying_position"]].values.astype(np.float32)
    X_transformed = preprocessor.transform(pd.DataFrame(X, columns=["qualifying_position"]))

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    predictions = model.run([output_name], {input_name: X_transformed})[0]

    df["predicted_position"] = np.round(predictions).astype(int)
    df["predicted_score"] = predictions
    df["is_podium_prediction"] = df["predicted_position"] <= 3
    df["predicted_rank"] = df["predicted_score"].rank(method="first", ascending=True).astype(int)

    return df


def write_to_dynamodb(predictions_df: pd.DataFrame, event_partition_key: str, session_type: str = "qualifying") -> None:
    """Write predictions to DynamoDB."""
    from decimal import Decimal
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(PREDICTIONS_TABLE)

    # Include session type in partition key for sprint weekends
    full_key = f"{event_partition_key}_{session_type}" if session_type != "qualifying" else event_partition_key

    with table.batch_writer() as batch:
        for _, row in predictions_df.iterrows():
            item = {
                "event_partition_key": full_key,
                "driver_id": str(row.get("driver_id", row.get("driver_number", ""))),
                "predicted_position": int(row["predicted_position"]),
                "predicted_rank": int(row["predicted_rank"]),
                "predicted_score": Decimal(str(row["predicted_score"])),
                "is_podium_prediction": bool(row["is_podium_prediction"]),
                "qualifying_position": int(row["qualifying_position"]),
                "session_type": session_type,
            }
            batch.put_item(Item=item)


def write_to_s3(predictions_df: pd.DataFrame, year: int, event_slug: str, session_type: str = "qualifying") -> str:
    """Write predictions to S3 as Parquet."""
    s3 = boto3.client("s3")
    session_folder = f"session_{session_type}" if session_type != "qualifying" else ""
    key = f"{PREDICTIONS_S3_PREFIX}/event_year={year}/event={event_slug}/{session_folder}predictions.parquet".replace("//", "/")

    buffer = io.BytesIO()
    predictions_df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3.upload_fileobj(buffer, S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"


def parse_event_key(partition_key: str) -> Tuple[int, str]:
    """Parse event_partition_key like '2025_Bahrain_Grand_Prix' into (year, event_slug)."""
    parts = partition_key.split("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid event_partition_key: {partition_key}")
    year = int(parts[0])
    # Event names in DynamoDB are like "Australian_Grand_Prix"
    # S3 paths use just the race name: "australian"
    event_name = parts[1].replace("_", " ")
    # Map full names to short names for S3 path
    event_short_names = {
        "australian grand prix": "australian",
        "bahrain grand prix": "bahrain",
        "chinese grand prix": "chinese",
        "dutch grand prix": "dutch",
        "emilia romagna grand prix": "emilia_romagna",
        "monaco grand prix": "monaco",
        "spanish grand prix": "spanish",
        "canadian grand prix": "canadian",
        "british grand prix": "british",
        "austrian grand prix": "austrian",
        "hungarian grand prix": "hungarian",
        "belgian grand prix": "belgian",
        "italian grand prix": "italian",
        "singapore grand prix": "singapore",
        "japanese grand prix": "japanese",
        "qatar grand prix": "qatar",
        "saudi arabian grand prix": "saudi_arabian",
        "abu dhabi grand prix": "abu_dhabi",
        "miami grand prix": "miami",
        "las vegas grand prix": "las_vegas",
        "são paulo grand prix": "sao_paulo",
    }
    event_lower = event_name.lower()
    event_slug = event_short_names.get(event_lower, event_lower.replace(" ", "_"))
    return year, event_slug


def get_race_date(event_partition_key: str) -> str:
    """Get race date from DynamoDB."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(DYNAMODB_TABLE)
    response = table.get_item(
        Key={PARTITION_KEY: event_partition_key, SORT_KEY: RACE_ABR}
    )
    item = response.get("Item")
    if not item:
        raise ValueError(f"No race row for {event_partition_key}")
    
    utc = item.get("session_date_utc")
    if not utc:
        raise ValueError("Race row missing session_date_utc")
    if isinstance(utc, str):
        return utc[:10]
    return str(utc)[:10]


def main(event_key: str, session_type: str = "qualifying") -> Dict[str, Any]:
    """Main inference entry point. session_type: 'qualifying' or 'sprint'."""
    start_time = time.time()
    
    log_json("INFO", "inference_started", event_partition_key=event_key, session_type=session_type)
    
    try:
        # Parse event key
        year, event_slug = parse_event_key(event_key)
        race_date = get_race_date(event_key)
        
        # Load model
        load_start = time.time()
        model, preprocessor = load_model_from_s3()
        load_time_ms = (time.time() - load_start) * 1000
        put_metric("Model.LoadTime", load_time_ms, unit="Milliseconds", session_type=session_type)
        log_json("INFO", "model_loaded", event_partition_key=event_key, load_time_ms=load_time_ms)
        
        # Get session data (qualifying or sprint)
        data_start = time.time()
        session_df = get_session_results(year, event_slug, session_type)
        data_time_ms = (time.time() - data_start) * 1000
        put_metric("Data.LoadTime", data_time_ms, unit="Milliseconds", session_type=session_type)
        log_json("INFO", "data_loaded", event_partition_key=event_key, session_type=session_type, 
                 drivers_count=len(session_df), load_time_ms=data_time_ms)
        
        # Run predictions
        predict_start = time.time()
        predictions_df = run_prediction(model, preprocessor, session_df)
        predict_time_ms = (time.time() - predict_start) * 1000
        put_metric("Prediction.RunTime", predict_time_ms, unit="Milliseconds", session_type=session_type)
        log_json("INFO", "predictions_computed", event_partition_key=event_key, 
                 drivers_predicted=len(predictions_df), predict_time_ms=predict_time_ms)
        
        # Write results (include session_type in storage key)
        storage_start = time.time()
        write_to_dynamodb(predictions_df, event_key, session_type)
        s3_uri = write_to_s3(predictions_df, year, event_slug, session_type)
        storage_time_ms = (time.time() - storage_start) * 1000
        put_metric("Storage.WriteTime", storage_time_ms, unit="Milliseconds", session_type=session_type)
        
        total_time_ms = (time.time() - start_time) * 1000
        put_metric("Inference.TotalTime", total_time_ms, unit="Milliseconds", session_type=session_type)
        put_metric("Prediction.Count", len(predictions_df), unit="Count", session_type=session_type)
        
        result = {
            "event_partition_key": event_key,
            "session_type": session_type,
            "event_year": year,
            "event_slug": event_slug,
            "race_date": race_date,
            "predictions": {
                "total_drivers": len(predictions_df),
                "podium_predictions": int(predictions_df["is_podium_prediction"].sum()),
            },
            "storage": {
                "dynamodb": PREDICTIONS_TABLE,
                "s3": s3_uri,
            },
            "timing_ms": {
                "model_load": load_time_ms,
                "data_load": data_time_ms,
                "prediction": predict_time_ms,
                "storage": storage_time_ms,
                "total": total_time_ms,
            }
        }
        
        log_json("INFO", "inference_completed", event_partition_key=event_key, session_type=session_type,
                  total_time_ms=total_time_ms, drivers_predicted=len(predictions_df))
        
        return result
        
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        put_metric("Inference.Errors", 1, unit="Count", session_type=session_type)
        log_json("ERROR", "inference_failed", event_partition_key=event_key, session_type=session_type,
                 error=str(e), total_time_ms=total_time_ms)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fargate Inference")
    parser.add_argument("--event-key", required=True, help="Event partition key (e.g., 2025_Australian_Grand_Prix)")
    parser.add_argument("--session-type", default="qualifying", choices=["qualifying", "sprint"], 
                        help="Session type: 'qualifying' or 'sprint' (default: qualifying)")
    args = parser.parse_args()
    
    try:
        main(args.event_key, args.session_type)
    except Exception as e:
        log_json("ERROR", "inference_exited_with_error", event_partition_key=args.event_key, 
                 session_type=args.session_type, error=str(e))
        sys.exit(1)

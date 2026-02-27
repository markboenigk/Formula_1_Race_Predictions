"""
Inference Lambda: run at prediction time (e.g. after qualifying).

Resolves race date + circuit from event/DynamoDB, fetches live weather via
weather.forecast.fetch_race_forecast(), builds the feature vector, runs the
model, and writes predictions. Weather is fetched in-process (one HTTP call
to Open-Meteo); no separate weather Lambda or S3 read for this path.

Deploy this Lambda with src/weather/forecast.py and src/common (and requests)
in its package or layer.
"""
import io
import json
import os
from typing import Any, Dict, List, Tuple

import boto3
import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from boto3.dynamodb.conditions import Key

# Resolve race date + circuit; fetch live forecast in-process
from src.common.circuit_coordinates import get_circuit_coords, normalize_event_name
from src.weather.forecast import fetch_race_forecast

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_MODELS_PREFIX = os.environ.get("S3_MODEL_PATH", "models")
S3_SILVER_PREFIX = os.environ.get("S3_SILVER_PATH", "silver")
PREDICTIONS_TABLE_NAME = os.environ.get("PREDICTIONS_TABLE_NAME", "f1_predictions")
PREDICTIONS_S3_PREFIX = os.environ.get("PREDICTIONS_S3_PREFIX", "predictions")

PARTITION_KEY = "event_partition_key"
SORT_KEY = "session_name_abr"
RACE_ABR = "R"
QUALIFYING_ABR = "Q"
SESSION_DATE_UTC = "session_date_utc"
EVENT_NAME = "EventName"
CIRCUIT_LAT = "circuit_lat"
CIRCUIT_LNG = "circuit_lng"

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")


def _load_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        return {}
    if isinstance(body, dict):
        return body
    if isinstance(body, str) and body.strip():
        return json.loads(body)
    return {}


def _get_race_row(event_partition_key: str) -> Dict[str, Any]:
    """Fetch race (R) row from DynamoDB for event."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    response = table.get_item(
        Key={PARTITION_KEY: event_partition_key, SORT_KEY: RACE_ABR}
    )
    item = response.get("Item")
    if not item:
        raise ValueError(f"No race row for {event_partition_key}")
    return item


def _race_date_from_row(row: Dict[str, Any]) -> str:
    """Return race date YYYY-MM-DD from session_date_utc."""
    utc = row.get(SESSION_DATE_UTC) or row.get("session_date_utc")
    if not utc:
        raise ValueError("Race row missing session_date_utc")
    if isinstance(utc, str):
        return utc[:10]
    return str(utc)[:10]


def _parse_event_partition_key(partition_key: str) -> Tuple[int, str]:
    """Parse event_partition_key like '2025_Bahrain_Grand_Prix' into (year, event_slug)."""
    parts = partition_key.split("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid event_partition_key format: {partition_key}")
    year = int(parts[0])
    event_name = parts[1].replace("_", " ")  # DynamoDB uses underscores, S3 uses spaces
    event_slug = normalize_event_name(event_name)
    return year, event_slug


def _load_model_from_s3() -> Tuple[ort.InferenceSession, Any]:
    """Load ONNX model and preprocessor from S3 via _LATEST pointer. Checks _SUCCESS sentinel."""
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
            raise RuntimeError(f"Model run {run_id} missing _SUCCESS sentinel - not ready for inference")
        raise

    # Download ONNX model
    model_key = f"{S3_MODELS_PREFIX}/run_id={run_id}/model.onnx"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=model_key)
    model_bytes = resp["Body"].read()
    model = ort.InferenceSession(model_bytes)

    # Download preprocessor (still needed for feature transformation)
    preprocessor_key = f"{S3_MODELS_PREFIX}/run_id={run_id}/preprocessor.joblib"
    resp = s3.get_object(Bucket=S3_BUCKET, Key=preprocessor_key)
    preprocessor = joblib.load(io.BytesIO(resp["Body"].read()))

    return model, preprocessor


def _get_qualifying_results(year: int, event_slug: str) -> pd.DataFrame:
    """Fetch qualifying results from silver S3 for a given event."""
    prefix = f"{S3_SILVER_PREFIX}/results/event_year={year}/event={event_slug}/session={QUALIFYING_ABR}/"
    s3 = boto3.client("s3")

    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    if "Contents" not in resp:
        raise ValueError(f"No qualifying results found at s3://{S3_BUCKET}/{prefix}")

    dfs = []
    for obj in resp["Contents"]:
        key = obj["Key"]
        if not key.endswith(".parquet"):
            continue
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No parquet files found at s3://{S3_BUCKET}/{prefix}")

    combined = pd.concat(dfs, ignore_index=True)
    # Use finishing_position as qualifying position
    combined = combined.rename(columns={"finishing_position": "qualifying_position"})
    return combined[["driver_number", "driver_id", "full_name", "qualifying_position"]].copy()


def _run_prediction(model: ort.InferenceSession, preprocessor: Any, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """Run ONNX model prediction on qualifying data."""
    df = qualifying_df.dropna(subset=["qualifying_position"]).copy()
    df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")
    df = df.dropna(subset=["qualifying_position"])

    if df.empty:
        raise ValueError("No valid qualifying positions to predict")

    feature_col = model.get_inputs()[0].name
    X = df[["qualifying_position"]].values.astype(np.float32)
    X_transformed = preprocessor.transform(pd.DataFrame(X, columns=["qualifying_position"]))

    # ONNX prediction
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    predictions = model.run([output_name], {input_name: X_transformed})[0]

    df["predicted_position"] = np.round(predictions).astype(int)
    df["predicted_score"] = predictions
    df["is_podium_prediction"] = df["predicted_position"] <= 3

    # Rank within the race
    df["predicted_rank"] = df["predicted_score"].rank(method="first", ascending=True).astype(int)

    return df


def _write_predictions_to_dynamodb(predictions_df: pd.DataFrame, event_partition_key: str) -> None:
    """Write predictions to DynamoDB f1_predictions table."""
    from decimal import Decimal
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(PREDICTIONS_TABLE_NAME)

    with table.batch_writer() as batch:
        for _, row in predictions_df.iterrows():
            item = {
                "event_partition_key": event_partition_key,
                "driver_id": str(row.get("driver_id", row.get("driver_number", ""))),
                "predicted_position": int(row["predicted_position"]),
                "predicted_rank": int(row["predicted_rank"]),
                "predicted_score": Decimal(str(row["predicted_score"])),
                "is_podium_prediction": bool(row["is_podium_prediction"]),
                "qualifying_position": int(row["qualifying_position"]),
            }
            batch.put_item(Item=item)


def _write_predictions_to_s3(predictions_df: pd.DataFrame, year: int, event_slug: str) -> str:
    """Write predictions to S3 as Parquet. Returns the S3 URI."""
    s3 = boto3.client("s3")
    key = f"{PREDICTIONS_S3_PREFIX}/event_year={year}/event={event_slug}/predictions.parquet"

    buffer = io.BytesIO()
    predictions_df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3.upload_fileobj(buffer, S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Invoked after qualifying (e.g. by Step Functions). Fetches live weather
    in-process, runs prediction, and writes results to DynamoDB + S3.
    """
    try:
        body = _load_body(event)
        event_partition_key = body.get("event_partition_key")
        if not event_partition_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "event_partition_key required"}),
            }

        # Get race info from DynamoDB
        race_row = _get_race_row(event_partition_key)
        race_date = _race_date_from_row(race_row)
        event_name = race_row.get(EVENT_NAME) or race_row.get("EventName")
        if not event_name:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Race row missing event name"}),
            }

        # Parse event key
        year, event_slug = _parse_event_partition_key(event_partition_key)

        # Fetch live weather (optional for baseline - model uses qualifying_position only)
        try:
            lat, lng = get_circuit_coords(event_name)
            weather = fetch_race_forecast(lat, lng, race_date)
        except Exception as e:
            print(f"Warning: Could not fetch weather: {e}")
            weather = {"error": str(e), "note": "Weather fetch failed - continuing without weather"}

        # Load model and preprocessor from S3
        model, preprocessor = _load_model_from_s3()

        # Get qualifying results from silver S3
        qualifying_df = _get_qualifying_results(year, event_slug)

        # Run predictions
        predictions_df = _run_prediction(model, preprocessor, qualifying_df)

        # Write predictions to DynamoDB and S3
        _write_predictions_to_dynamodb(predictions_df, event_partition_key)
        s3_uri = _write_predictions_to_s3(predictions_df, year, event_slug)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "event_partition_key": event_partition_key,
                "event_year": year,
                "event_slug": event_slug,
                "race_date": race_date,
                "weather": weather,
                "predictions": {
                    "total_drivers": len(predictions_df),
                    "podium_predictions": int(predictions_df["is_podium_prediction"].sum()),
                },
                "storage": {
                    "dynamodb": PREDICTIONS_TABLE_NAME,
                    "s3": s3_uri,
                },
            }, default=str),
        }
    except ValueError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }

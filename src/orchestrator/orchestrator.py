"""
Pipeline Orchestrator: Runs every 2 hours to:
1. Check for new session data (ingest if available)
2. Determine race context from DynamoDB
3. Check if predictions should run (qualifying complete, race pending)
4. Run inference if conditions met
"""
import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration
# ============================================================================
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "f1-race-prediction")
S3_GOLD_PREFIX = os.environ.get("S3_GOLD_PATH", "gold")
S3_MODELS_PREFIX = os.environ.get("S3_MODEL_PATH", "models")
S3_SILVER_PREFIX = os.environ.get("S3_SILVER_PATH", "silver")
PREDICTIONS_S3_PREFIX = os.environ.get("PREDICTIONS_S3_PREFIX", "predictions")
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "f1_session_tracking")

# Lambda function names (can be overridden via environment variables)
DATA_INGESTION_LAMBDA = os.environ.get("DATA_INGESTION_LAMBDA", "f1-data-ingestion")
INFERENCE_LAMBDA = os.environ.get("INFERENCE_LAMBDA", "f1-inference")

# ============================================================================
# DynamoDB Helpers
# ============================================================================

def get_dynamodb_table():
    """Get DynamoDB table resource."""
    dynamodb = boto3.resource("dynamodb")
    return dynamodb.Table(DYNAMODB_TABLE_NAME)


def get_session_by_key(partition_key: str, sort_key: str) -> Optional[Dict]:
    """Get a single session from DynamoDB."""
    table = get_dynamodb_table()
    response = table.get_item(Key={"event_partition_key": partition_key, "session_name_abr": sort_key})
    return response.get("Item")


def get_event_sessions(partition_key: str) -> Dict[str, Dict]:
    """Get all sessions for an event."""
    table = get_dynamodb_table()
    response = table.query(
        KeyConditionExpression=Key("event_partition_key").eq(partition_key)
    )
    sessions = {}
    for item in response.get("Items", []):
        sessions[item["session_name_abr"]] = item
    return sessions


def get_upcoming_race_weekend(now: datetime) -> Optional[Dict]:
    """
    Find the next race weekend that hasn't started yet.
    Returns the race session info and event context.
    """
    table = get_dynamodb_table()
    
    # Find the next race session that's in the future
    # We query for races (R) and check if they're upcoming
    # For simplicity, get recent events and filter
    
    # Get all future qualifying sessions
    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Scan with filter is expensive but DynamoDB doesn't support future queries well
    # In production, consider using a GSI on session_date_utc
    response = table.scan(
        FilterExpression="session_name_abr = :q AND session_date_utc > :now",
        ExpressionAttributeValues={":q": "Q", ":now": now_str}
    )
    
    qualifying_sessions = sorted(
        response.get("Items", []),
        key=lambda x: x.get("session_date_utc", "")
    )
    
    if not qualifying_sessions:
        return None
    
    # Get the first upcoming qualifying
    qual_session = qualifying_sessions[0]
    event_key = qual_session["event_partition_key"]
    
    # Get all sessions for this event
    event_sessions = get_event_sessions(event_key)
    
    # Get race session
    race_session = event_sessions.get("R")
    
    return {
        "event_partition_key": event_key,
        "event_year": qual_session.get("event_year"),
        "event_name": qual_session.get("EventName"),
        "qualifying_session": qual_session,
        "race_session": race_session,
        "all_sessions": event_sessions,
    }


def get_current_race_context(now: datetime) -> Optional[Dict]:
    """
    Find the current race weekend context.
    Returns info if we're in a race weekend where:
    - Qualifying has completed
    - Race hasn't started yet (prediction window)
    """
    table = get_dynamodb_table()
    
    # Get all events that have qualifying in the past and race in the future
    # This is the "prediction window"
    
    # First, get recent qualifying sessions (last 7 days to handle timezones)
    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    response = table.scan(
        FilterExpression="session_name_abr = :q AND session_date_utc > :week_ago AND session_date_utc < :now",
        ExpressionAttributeValues={
            ":q": "Q",
            ":week_ago": week_ago,
            ":now": now_str
        }
    )
    
    for qual_session in response.get("Items", []):
        event_key = qual_session["event_partition_key"]
        event_sessions = get_event_sessions(event_key)
        
        race_session = event_sessions.get("R")
        if not race_session:
            continue
            
        race_time = datetime.fromisoformat(
            race_session["session_date_utc"].replace("Z", "+00:00")
        )
        
        # If race is in the future, we're in the prediction window
        if race_time > now:
            # Check if qualifying is actually complete (has data been loaded?)
            # For now, use time-based check - qualifying time has passed
            qual_time = datetime.fromisoformat(
                qual_session["session_date_utc"].replace("Z", "+00:00")
            )
            
            if qual_time <= now:
                return {
                    "event_partition_key": event_key,
                    "event_year": qual_session.get("event_year"),
                    "event_name": qual_session.get("EventName"),
                    "qualifying_time": qual_session["session_date_utc"],
                    "race_time": race_session["session_date_utc"],
                    "qualifying_session": qual_session,
                    "race_session": race_session,
                    "all_sessions": event_sessions,
                }
    
    return None


# ============================================================================
# Data Ingestion Check
# ============================================================================

def check_new_session_data_available(partition_key: str, session_abr: str) -> bool:
    """
    Check if new data is available for a session.
    Returns True if data has been loaded (via fastf1 ingestion).
    """
    session = get_session_by_key(partition_key, session_abr)
    if not session:
        return False
    return session.get("data_loaded", False)


# ============================================================================
# Idempotency Check
# ============================================================================

def prediction_exists(event_partition_key: str, event_year: int) -> bool:
    """
    Check if predictions already exist for this event.
    Uses S3 as the source of truth for predictions.
    """
    s3 = boto3.client("s3")
    
    # Try to find predictions for this event
    prefix = f"{PREDICTIONS_S3_PREFIX}/event_year={event_year}/"
    
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=1
        )
        return "Contents" in response and len(response["Contents"]) > 0
    except Exception as e:
        logger.warning(f"Error checking prediction existence: {e}")
        return False


def get_latest_prediction_run_id(event_partition_key: str) -> Optional[str]:
    """Get the run_id of the latest prediction if exists."""
    # Could store this in DynamoDB or S3 metadata
    # For now, check S3
    s3 = boto3.client("s3")
    
    prefix = f"{PREDICTIONS_S3_PREFIX}/"
    
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
        )
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith("predictions.parquet"):
                # Extract event from key
                return key
    except:
        pass
    
    return None


# ============================================================================
# Model Readiness Check
# ============================================================================

def is_model_ready() -> Tuple[bool, Optional[str]]:
    """
    Check if a trained model is ready for inference.
    Returns (is_ready, run_id).
    """
    s3 = boto3.client("s3")
    
    try:
        # Read _LATEST pointer
        latest_key = f"{S3_MODELS_PREFIX}/_LATEST"
        response = s3.get_object(Bucket=S3_BUCKET, Key=latest_key)
        run_id = response["Body"].read().decode("utf-8").strip()
        
        # Check _SUCCESS sentinel
        success_key = f"{S3_MODELS_PREFIX}/run_id={run_id}/_SUCCESS"
        s3.head_object(Bucket=S3_BUCKET, Key=success_key)
        
        return True, run_id
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.info("No trained model available yet")
            return False, None
        raise
    except Exception as e:
        logger.warning(f"Error checking model readiness: {e}")
        return False, None


# ============================================================================
# Main Orchestrator Logic
# ============================================================================

def should_ingest_data() -> Dict[str, Any]:
    """
    Determine if we should check for new data to ingest.
    Returns reason and action.
    """
    now = datetime.now(timezone.utc)
    
    # Check upcoming race weekend
    upcoming = get_upcoming_race_weekend(now)
    if not upcoming:
        return {
            "should_ingest": False,
            "reason": "no upcoming race weekend found",
            "context": None
        }
    
    event_key = upcoming["event_partition_key"]
    qual_session = upcoming["qualifying_session"]
    
    # Check if qualifying data is loaded
    qual_loaded = qual_session.get("data_loaded", False)
    
    # If qualifying time has passed, we should check for data
    qual_time = datetime.fromisoformat(
        qual_session["session_date_utc"].replace("Z", "+00:00")
    )
    
    if qual_time <= now and not qual_loaded:
        return {
            "should_ingest": True,
            "reason": "qualifying time passed, data not yet loaded",
            "context": upcoming
        }
    
    return {
        "should_ingest": False,
        "reason": f"qualifying data status: {qual_loaded}",
        "context": upcoming
    }


def should_run_prediction() -> Dict[str, Any]:
    """
    Determine if we should run predictions now.
    Returns reason and context.
    """
    now = datetime.now(timezone.utc)
    
    # Check if model is ready
    model_ready, run_id = is_model_ready()
    if not model_ready:
        return {
            "should_predict": False,
            "reason": "no trained model available",
            "model_run_id": None,
            "context": None
        }
    
    # Get current race context (qualifying complete, race pending)
    context = get_current_race_context(now)
    if not context:
        return {
            "should_predict": False,
            "reason": "not in prediction window (no qualifying-complete race pending)",
            "model_run_id": run_id,
            "context": None
        }
    
    # Add model info to context
    context["model_run_id"] = run_id
    
    # Check idempotency - don't re-predict
    event_key = context["event_partition_key"]
    event_year = context["event_year"]
    
    if prediction_exists(event_key, event_year):
        return {
            "should_predict": False,
            "reason": f"predictions already exist for {event_key}",
            "model_run_id": run_id,
            "context": context
        }
    
    # Check if qualifying data is loaded
    qual_session = context.get("qualifying_session", {})
    if not qual_session.get("data_loaded", False):
        return {
            "should_predict": False,
            "reason": "qualifying data not yet loaded",
            "model_run_id": run_id,
            "context": context
        }
    
    return {
        "should_predict": True,
        "reason": "all conditions met",
        "model_run_id": run_id,
        "context": context
    }


def run_pipeline() -> Dict[str, Any]:
    """
    Main pipeline orchestrator.
    Runs every 2 hours to:
    1. Check if new data needs ingestion
    2. Check if predictions should run
    """
    now = datetime.now(timezone.utc)
    
    results = {
        "timestamp": now.isoformat(),
        "data_ingestion": None,
        "prediction": None,
        "actions_taken": [],
    }
    
    # Step 1: Check data ingestion
    ingest_check = should_ingest_data()
    results["data_ingestion"] = ingest_check
    
    if ingest_check["should_ingest"]:
        context = ingest_check["context"]
        action_result = trigger_data_ingestion(
            partition_key=context["event_partition_key"],
            session_abr="Q"
        )
        results["actions_taken"].append({
            "action": "trigger_data_ingestion",
            "event": context["event_partition_key"],
            "session": "Q",
            "invocation_result": action_result
        })
        logger.info(f"Triggered data ingestion for {context['event_partition_key']}")
    
    # Step 2: Check prediction
    predict_check = should_run_prediction()
    results["prediction"] = predict_check
    
    if predict_check["should_predict"]:
        context = predict_check["context"]
        action_result = trigger_inference(context)
        results["actions_taken"].append({
            "action": "run_prediction",
            "event": context["event_partition_key"],
            "model_run_id": predict_check["model_run_id"],
            "invocation_result": action_result
        })
        logger.info(f"Triggered prediction for {context['event_partition_key']} with model {predict_check['model_run_id']}")
    else:
        logger.info(f"Skipping prediction: {predict_check['reason']}")
    
    return results


# ============================================================================
# Lambda Invocation
# ============================================================================

def invoke_lambda(lambda_name: str, payload: dict) -> dict:
    """
    Invoke another Lambda function asynchronously.
    Returns the invocation response.
    """
    try:
        lambda_client = boto3.client("lambda")
        response = lambda_client.invoke(
            FunctionName=lambda_name,
            InvocationType="Event",  # Async invocation
            Payload=json.dumps(payload)
        )
        logger.info(f"Invoked {lambda_name}: {response.get('StatusCode')}")
        return {
            "status": "invoked",
            "lambda": lambda_name,
            "status_code": response.get("StatusCode")
        }
    except Exception as e:
        logger.error(f"Failed to invoke {lambda_name}: {e}")
        return {
            "status": "failed",
            "lambda": lambda_name,
            "error": str(e)
        }


def trigger_data_ingestion(partition_key: str, session_abr: str) -> dict:
    """
    Trigger the data ingestion Lambda for a specific session.
    """
    payload = {
        "action": "ingest_session",
        "event_partition_key": partition_key,
        "session_abr": session_abr
    }
    logger.info(f"Triggering data ingestion for {partition_key}/{session_abr}")
    return invoke_lambda(DATA_INGESTION_LAMBDA, payload)


def trigger_inference(context: dict) -> dict:
    """
    Trigger the inference Lambda for a specific event.
    """
    payload = {
        "action": "run_prediction",
        "event_partition_key": context["event_partition_key"],
        "event_year": context["event_year"],
        "event_name": context["event_name"],
        "model_run_id": context.get("model_run_id"),
        "race_time": context.get("race_time"),
    }
    logger.info(f"Triggering inference for {context['event_partition_key']}")
    return invoke_lambda(INFERENCE_LAMBDA, payload)


# ============================================================================
# Lambda Handler
# ============================================================================

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler - runs every 2 hours via EventBridge or Step Functions.
    Supports two modes:
    - Standard mode: Run full pipeline (EventBridge trigger)
    - Step Functions mode: Individual actions (check_race_window, store_prediction)
    """
    try:
        action = event.get("action")
        logger.info(f"Starting pipeline orchestration - action: {action}")
        
        # Step Functions mode: return race window status for decision
        if action == "check_race_window":
            return handle_check_race_window()
        
        # Step Functions mode: store prediction result
        if action == "store_prediction":
            return handle_store_prediction(event)
        
        # Standard mode: Run full pipeline (EventBridge trigger)
        results = run_pipeline()
        
        logger.info(f"Pipeline results: {json.dumps(results, default=str)}")
        
        # Determine if any actions were taken
        actions = results.get("actions_taken", [])
        
        if not actions:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No actions needed",
                    "timestamp": results["timestamp"],
                    "data_ingestion": results["data_ingestion"]["reason"],
                    "prediction": results["prediction"]["reason"],
                })
            }
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Actions taken: {len(actions)}",
                "timestamp": results["timestamp"],
                "actions": actions,
                "details": results,
            })
        }
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def handle_check_race_window() -> Dict[str, Any]:
    """
    Handle check_race_window action from Step Functions.
    Returns the race window status for decision making.
    Note: Returns data directly (not API Gateway format) for Step Functions.
    """
    now = datetime.now(timezone.utc)
    
    # Check data ingestion
    ingest_check = should_ingest_data()
    
    # Check prediction
    predict_check = should_run_prediction()
    
    # Return directly for Step Functions (not wrapped in statusCode/body)
    return {
        "should_ingest": ingest_check["should_ingest"],
        "should_predict": predict_check["should_predict"],
        "ingest_reason": ingest_check["reason"],
        "predict_reason": predict_check["reason"],
        "context": predict_check.get("context") or ingest_check.get("context"),
        "model_run_id": predict_check.get("model_run_id"),
    }


def handle_store_prediction(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle store_prediction action from Step Functions.
    Stores prediction results to S3 and DynamoDB.
    Note: Returns data directly (not API Gateway format) for Step Functions.
    """
    prediction_result = event.get("prediction_result", {})
    
    if not prediction_result:
        return {"message": "No prediction result to store"}
    
    logger.info(f"Storing prediction result: {json.dumps(prediction_result, default=str)}")
    
    # Return directly for Step Functions
    return {
        "message": "Prediction stored successfully",
        "result": prediction_result
    }

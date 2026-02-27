import json
import os
import io
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import fastf1
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key, Attr

# ============================================================================
# DynamoDB Schema Constants
# ============================================================================
# Primary Key Fields
PARTITION_KEY = 'event_partition_key'  # e.g., "2025_Australian_Grand_Prix"
SORT_KEY = 'session_name_abr'  # e.g., "P1", "Q1", "R"

# Event Fields
EVENT_YEAR = 'event_year'  # Int: Year of the event (e.g., 2025)
EVENT_NUMBER = 'event_number'  # Int: Round number (1-24)
EVENT_NAME = 'EventName'  # String: Full event name (e.g., "Australian Grand Prix")
EVENT_FORMAT = 'event_format'  # String: "conventional" or "sprint"

# Session Fields
SESSION_NAME = 'session_name'  # String: Full session name (e.g., "Practice 1", "Race")
SESSION_NAME_ABR = 'session_name_abr'  # String: Abbreviation (e.g., "P1", "R")
SESSION_DATE = 'session_date'  # String: Local session time
SESSION_DATE_UTC = 'session_date_utc'  # String: ISO 8601 UTC format (e.g., "2025-03-16T01:00:00Z")
IS_BEFORE_RACE = 'is_before_race'  # Boolean: True for all sessions except Race

# Data Loading Status Fields
DATA_LOADED = 'data_loaded'  # Boolean: Marks if session data has been downloaded and stored
S3_RESULTS_LOCATION = 's3_results_location'  # String: S3 path to results Parquet file
S3_LAPS_LOCATION = 's3_laps_location'  # String: S3 path to laps Parquet file

# Required Fields (must exist for a valid session record)
REQUIRED_FIELDS = [
    PARTITION_KEY,
    SORT_KEY,
    EVENT_YEAR,
    EVENT_NAME,
    SESSION_NAME,
    SESSION_NAME_ABR,
]


def get_table_name() -> str:
    """Get the DynamoDB table name from environment variable or use default."""
    return os.environ.get('DYNAMODB_TABLE_NAME', 'f1_session_tracking')


def validate_session_record(session: dict) -> tuple[bool, list[str]]:
    """
    Validate that a session record has all required fields.
    
    Args:
        session: Dictionary representing a session record
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing_fields = []
    for field in REQUIRED_FIELDS:
        if field not in session or session[field] is None:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


# ============================================================================
# Configuration
# ============================================================================
# DynamoDB Configuration
DYNAMODB_TABLE_NAME = get_table_name()

# S3 Configuration
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'f1-race-prediction')
# Backward-compatible env var naming: prefer S3_BRONZE_PATH but support existing S3_WL_BRONZE_PATH.
S3_WL_BRONZE_PATH = os.environ.get('S3_BRONZE_PATH') or os.environ.get('S3_WL_BRONZE_PATH', 'bronze')

# FastF1 Cache Configuration
if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
    # Running in AWS Lambda
    F1_CACHE_DIR = '/tmp/f1cache'
else:
    # Running locally - use project root f1cache directory
    F1_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'f1cache')
    F1_CACHE_DIR = os.path.abspath(F1_CACHE_DIR)

# Configure FastF1 cache directory
os.makedirs(F1_CACHE_DIR, exist_ok=True)

# Enable FastF1 cache
# This should be done at module level (outside handler) for efficiency
fastf1.Cache.enable_cache(F1_CACHE_DIR)
print(f"FastF1 cache enabled at: {F1_CACHE_DIR}")


def get_sessions_needing_data_load(
    table_name: Optional[str] = None,
    include_future_sessions: bool = False,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Efficiently query DynamoDB to find sessions that haven't loaded their data yet.
    
    For ~800 items, uses a scan with filter expression which is efficient enough.
    Filters for sessions where:
    - data_loaded is False or doesn't exist
    - Optionally filters out future sessions (only past sessions)
    
    Args:
        table_name: Name of the DynamoDB table (defaults to config value)
        include_future_sessions: If True, include future sessions. If False, only past sessions.
        limit: Optional limit on number of sessions to return
    
    Returns:
        List of dictionaries containing session data that needs data loading
    """
    if table_name is None:
        table_name = get_table_name()
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        # Build filter expression: data_loaded is False or doesn't exist
        # In DynamoDB, we need to check both cases:
        # 1. Attribute doesn't exist (new sessions that haven't been processed)
        # 2. Attribute exists but is False (sessions marked as not loaded)
        filter_expression = Attr(DATA_LOADED).not_exists() | Attr(DATA_LOADED).eq(False)
        
        # Optionally filter out future sessions (uses system UTC time)
        if not include_future_sessions:
            current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            print(f"Filtering to past sessions only: session_date_utc < {current_time_utc}")
            # Only include sessions where session_date_utc exists and is in the past
            filter_expression = filter_expression & (
                Attr(SESSION_DATE_UTC).exists() & 
                Attr(SESSION_DATE_UTC).lt(current_time_utc)
            )
        
        # Scan with filter expression. Must fetch ALL matching items first, then sort and
        # apply limit - otherwise we only get the first chunk (arbitrary order) and miss
        # e.g. 2025 sessions if DynamoDB returns 2024/2023 first.
        items = []
        scan_kwargs = {
            'FilterExpression': filter_expression
        }
        response = table.scan(**scan_kwargs)
        items.extend(response.get('Items', []))
        while 'LastEvaluatedKey' in response:
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = table.scan(**scan_kwargs)
            items.extend(response.get('Items', []))
        
        # Sort so we prioritize by year (2025 first, then 2024, then 2023), then latest
        # session date within that year. Items without year/date go last.
        def _year_desc(x):
            try:
                return -(int(x.get(EVENT_YEAR)) or 0)
            except (TypeError, ValueError):
                return 0

        items.sort(key=lambda x: x.get(SESSION_DATE_UTC) or "", reverse=True)  # latest date first
        items.sort(key=_year_desc)  # 2025 first, then 2024, then 2023 (stable sort keeps date order)
        items.sort(key=lambda x: (x.get(EVENT_YEAR) is None))  # missing year to end
        # Apply limit after sort so we process the N most recent sessions first
        if limit:
            items = items[:limit]

        print(f"Found {len(items)} sessions needing data load")
        return items
    
    except Exception as e:
        print(f"Error querying DynamoDB table: {str(e)}")
        raise


from src.common.circuit_coordinates import normalize_event_name


def store_results_to_s3(
    results_df: pd.DataFrame,
    event_year: int,
    event_name: str,
    session_name_abr: str,
    bucket_name: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    data_type: str = 'Results'
) -> Dict[str, Any]:
    """
    Store session results DataFrame to S3 as parquet file.
    
    Path structure: s3://f1-race-prediction/raw/season=2024/event=monaco/session=R/results/2024_Monaco_Grand_Prix_Results.parquet
    
    Args:
        results_df: DataFrame containing session results
        event_year: Year of the event (e.g., 2024)
        event_name: Full event name (e.g., 'Monaco Grand Prix')
        session_name_abr: Session abbreviation (e.g., 'R', 'P1', 'Q1')
        bucket_name: S3 bucket name (defaults to config value)
        s3_prefix: S3 prefix/path (defaults to config value)
    
    Returns:
        Dictionary with upload status and S3 path
    """
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    if s3_prefix is None:
        s3_prefix = S3_WL_BRONZE_PATH
    
    s3_client = boto3.client('s3')
    
    try:
        # Normalize event name for path
        event_normalized = normalize_event_name(event_name)
        
        # Create filename: {year}_{EventName}_Results.parquet
        # Replace spaces with underscores in event name for filename
        filename_event_name = event_name.replace(' ', '_')
        filename = f"{event_year}_{filename_event_name}_{data_type}.parquet"
        
        # Build S3 key path
        s3_key = f"{s3_prefix}/season={event_year}/event={event_normalized}/session={session_name_abr}/{data_type}/{filename}"
        
        # Convert DataFrame to parquet bytes
        parquet_buffer = io.BytesIO()
        results_df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
        parquet_buffer.seek(0)
        
        # Upload to S3 with server-side encryption
        s3_client.upload_fileobj(
            parquet_buffer,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': 'application/octet-stream',
                'ServerSideEncryption': 'AES256'
            }
        )
        
        s3_path = f"s3://{bucket_name}/{s3_key}"
        print(f"Successfully uploaded results to {s3_path}")
        
        return {
            'success': True,
            's3_path': s3_path,
            'bucket': bucket_name,
            'key': s3_key,
            'filename': filename,
            'rows': len(results_df)
        }
    
    except Exception as e:
        error_msg = f"Failed to upload results to S3: {str(e)}"
        print(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'bucket': bucket_name,
            'event_year': event_year,
            'event_name': event_name,
            'session_name_abr': session_name_abr
        }


def convert_timedelta_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Timedelta columns in DataFrame to seconds (as new columns).
    
    FastF1 returns lap times and other time-related data as Python Timedelta objects.
    These can cause issues with Parquet readers and DynamoDB. This function creates
    new columns with "_seconds" suffix containing the numeric seconds values.
    
    Args:
        df: DataFrame that may contain Timedelta columns
    
    Returns:
        DataFrame with additional "_seconds" columns for each Timedelta column
    """
    df = df.copy()
    
    # Find all columns that are Timedelta type
    timedelta_columns = []
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            timedelta_columns.append(col)
    
    # Convert each Timedelta column to seconds
    for col in timedelta_columns:
        seconds_col_name = f"{col}_seconds"
        # Convert to total seconds, handling NaT (Not a Time) values
        df[seconds_col_name] = df[col].dt.total_seconds()
    
    return df


def process_session_data(event_year: int, event_name: str, session_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load session data once and extract both results and laps DataFrames.
    
    This function optimizes performance by loading the session only once instead of
    calling session.load() separately for results and laps, which would double
    execution time and AWS costs.
    
    Args:
        event_year: Year of the event (e.g., 2018)
        event_name: Name of the event (e.g., 'Australian Grand Prix')
        session_name: Name of the session (e.g., 'Practice 1', 'Race', 'Qualifying')
    
    Returns:
        Tuple of (results_df, laps_df) with added columns for session_name, event_name, 
        event_year, and "_seconds" columns for all Timedelta fields
    """
    # Initialize session
    session = fastf1.get_session(event_year, event_name, session_name)
    
    # Load data once (This fetches Results, Laps, and Telemetry all at once)
    # Setting telemetry=True is vital if you want speed/brake data later
    session.load(telemetry=True, weather=True)
    
    # Extract DataFrames
    results_df = session.results.copy()
    laps_df = session.laps.copy()
    
    # Convert Timedelta columns to seconds for Parquet/DynamoDB compatibility
    results_df = convert_timedelta_to_seconds(results_df)
    laps_df = convert_timedelta_to_seconds(laps_df)
    
    # Add metadata columns to both DataFrames
    results_df['session_name'] = session_name
    results_df['event_name'] = event_name
    results_df['event_year'] = event_year
    
    laps_df['session_name'] = session_name
    laps_df['event_name'] = event_name
    laps_df['event_year'] = event_year
    
    return results_df, laps_df


def get_session_results(event_year: int, event_name: str, session_name: str) -> pd.DataFrame:
    """
    Get session results for a specific F1 session.
    
    DEPRECATED: Use process_session_data() instead to avoid loading session twice.
    This function is kept for backward compatibility.
    
    Args:
        event_year: Year of the event (e.g., 2018)
        event_name: Name of the event (e.g., 'Australian Grand Prix')
        session_name: Name of the session (e.g., 'Practice 1', 'Race', 'Qualifying')
    
    Returns:
        pandas.DataFrame: Session results with added columns for session_name, event_name, and event_year
    """
    session = fastf1.get_session(event_year, event_name, session_name)
    session.load()
    session_results = session.results
    session_results['session_name'] = session_name
    session_results['event_name'] = event_name
    session_results['event_year'] = event_year
    return session_results


def get_laps(event_year: int, event_name: str, session_name: str) -> pd.DataFrame:
    """
    Get lap data for a specific F1 session.
    
    DEPRECATED: Use process_session_data() instead to avoid loading session twice.
    This function is kept for backward compatibility.
    
    Args:
        event_year: Year of the event (e.g., 2018)
        event_name: Name of the event (e.g., 'Australian Grand Prix')
        session_name: Name of the session (e.g., 'Practice 1', 'Race', 'Qualifying')
    
    Returns:
        pandas.DataFrame: Lap data with added columns for session_name, event_name, and event_year
    """
    session = fastf1.get_session(event_year, event_name, session_name)
    session.load()
    laps = session.laps
    laps['session_name'] = session_name
    laps['event_name'] = event_name
    laps['event_year'] = event_year
    return laps


def update_tracking_table(
    event_partition_key: str,
    session_name_abr: str,
    attribute_name: str,
    attribute_value: Any,
    table_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update a specific attribute in the DynamoDB tracking table.
    
    Args:
        event_partition_key: Partition key (event identifier)
        session_name_abr: Sort key (session abbreviation)
        attribute_name: Name of the attribute to update (e.g., 'results_loaded', 'data_loaded')
        attribute_value: Value to set for the attribute
        table_name: Name of the DynamoDB table (defaults to config value)
    
    Returns:
        Dictionary with update status
    """
    if table_name is None:
        table_name = get_table_name()
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        table.update_item(
            Key={
                PARTITION_KEY: event_partition_key,
                SORT_KEY: session_name_abr
            },
            UpdateExpression=f'SET {attribute_name} = :value',
            ExpressionAttributeValues={
                ':value': attribute_value
            }
        )
        print(f"Updated {event_partition_key}/{session_name_abr} - {attribute_name}={attribute_value}")
        return {
            'success': True,
            'event_partition_key': event_partition_key,
            'session_name_abr': session_name_abr,
            'attribute_name': attribute_name,
            'attribute_value': attribute_value
        }
    except Exception as e:
        error_msg = f"Failed to update DynamoDB: {str(e)}"
        print(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'event_partition_key': event_partition_key,
            'session_name_abr': session_name_abr
        }


def read_tracking_table(
    table_name: Optional[str] = None,
    event_partition_key: Optional[str] = None,
    session_name_abr: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read sessions from DynamoDB tracking table.
    
    Args:
        table_name: Name of the DynamoDB table (defaults to config value)
        event_partition_key: Optional partition key to filter by specific event
        session_name_abr: Optional sort key to filter by specific session
    
    Returns:
        List of dictionaries containing session data
    """
    if table_name is None:
        table_name = get_table_name()
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        if event_partition_key and session_name_abr:
            # Get specific item by partition key and sort key
            response = table.get_item(
                Key={
                    PARTITION_KEY: event_partition_key,
                    SORT_KEY: session_name_abr
                }
            )
            if 'Item' in response:
                return [response['Item']]
            else:
                return []
        
        elif event_partition_key:
            # Query all sessions for a specific event (partition key)
            response = table.query(
                KeyConditionExpression=Key(PARTITION_KEY).eq(event_partition_key)
            )
            return response.get('Items', [])
        
        else:
            # Scan entire table (use with caution - can be expensive)
            items = []
            response = table.scan()
            items.extend(response.get('Items', []))
            
            # Handle pagination if there are more items
            while 'LastEvaluatedKey' in response:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items.extend(response.get('Items', []))
            
            return items
    
    except Exception as e:
        print(f"Error reading from DynamoDB table: {str(e)}")
        raise


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Supports two invocation patterns:
    1. API Gateway (HTTP): Has httpMethod, path, body
    2. Direct Lambda (from orchestrator): Has action, event_partition_key, session_abr
    
    Args:
        event: Lambda event object
        context: Lambda context object
    
    Returns:
        API Gateway response dictionary or Lambda response
    """
    try:
        # Check if invoked by orchestrator (direct Lambda invocation)
        if "action" in event:
            return handle_orchestrator_invocation(event)
        
        # Otherwise, handle as API Gateway request
        return handle_api_gateway_request(event)
        


        # Extract optional parameters from request body
        include_future_sessions = body.get('include_future_sessions', False) if body else False
        limit = body.get('limit') if body else None
        
        # Get sessions needing data load
        not_loaded_sessions = get_sessions_needing_data_load(
            include_future_sessions=include_future_sessions,
            limit=limit
        )
        
        if not not_loaded_sessions:
            result = {
                "message": "No sessions found that need data loading",
                "sessions_processed": 0
            }
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
                },
                'body': json.dumps(result)
            }
        
        # Process each session
        processed_count = 0
        success_count = 0
        failed_count = 0
        
        for session in not_loaded_sessions:
            # Validate session record has required fields
            is_valid, missing_fields = validate_session_record(session)
            if not is_valid:
                print(f"Skipping session {session.get(PARTITION_KEY, 'unknown')}: missing required fields {missing_fields}")
                failed_count += 1
                continue
            
            # Get session data using schema constants
            event_year = int(session.get(EVENT_YEAR))
            event_name = session.get(EVENT_NAME)
            session_name = session.get(SESSION_NAME)
            session_name_abr = session.get(SESSION_NAME_ABR)
            event_partition_key = session.get(PARTITION_KEY)
            
            # Validate required fields are not None
            if not all([event_year, event_name, session_name, session_name_abr, event_partition_key]):
                print(f"Skipping session {event_partition_key}: missing required field values")
                failed_count += 1
                continue
            
            print(f"Processing session: {event_year} {event_name} - {session_name} ({session_name_abr})")
            
            # Track success for this session
            session_results_success = False
            session_laps_success = False
            
            try:
                # Load session once and extract both results and laps DataFrames
                # This optimizes performance by avoiding duplicate session.load() calls
                session_results, laps_df = process_session_data(event_year, event_name, session_name)
                
                # Store results to S3 (needs session_name_abr for path)
                s3_session_result = store_results_to_s3(
                    session_results, event_year, event_name, session_name_abr, data_type='Results'
                )
                
                if s3_session_result['success']:
                    # Update tracking table with S3 location
                    s3_results_path = s3_session_result['s3_path']
                    update_tracking_table(
                        event_partition_key, session_name_abr,
                        attribute_name=S3_RESULTS_LOCATION,
                        attribute_value=s3_results_path
                    )
                    session_results_success = True
                else:
                    print(f"Failed to store session results to S3: {s3_session_result.get('error', 'Unknown error')}")
                
                # Store laps to S3
                s3_session_laps = store_results_to_s3(
                    laps_df, event_year, event_name, session_name_abr, data_type='Laps'
                )
                
                if s3_session_laps['success']:
                    # Update tracking table with S3 location
                    s3_laps_path = s3_session_laps['s3_path']
                    update_tracking_table(
                        event_partition_key, session_name_abr,
                        attribute_name=S3_LAPS_LOCATION,
                        attribute_value=s3_laps_path
                    )
                    session_laps_success = True
                else:
                    print(f"Failed to store session laps to S3: {s3_session_laps.get('error', 'Unknown error')}")
                
                # Mark data_loaded as True only if both results and laps were successfully uploaded
                if session_results_success and session_laps_success:
                    update_tracking_table(
                        event_partition_key, session_name_abr,
                        attribute_name=DATA_LOADED,
                        attribute_value=True
                    )
                    success_count += 1
                    print(f"Successfully processed session: {event_year} {event_name} - {session_name}")
                else:
                    failed_count += 1
                    print(f"Failed to complete processing for session: {event_year} {event_name} - {session_name}")
                
                processed_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"Error processing session {event_partition_key}/{session_name_abr}: {str(e)}")
                # Log the failure in DynamoDB so you know to check it later
                continue
        
def handle_orchestrator_invocation(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle direct invocation from the pipeline orchestrator.
    
    Expected payload:
    {
        "action": "ingest_session",
        "event_partition_key": "2025_Australian_Grand_Prix",
        "session_abr": "Q"
    }
    """
    action = event.get("action")
    
    if action == "ingest_session":
        partition_key = event.get("event_partition_key")
        session_abr = event.get("session_abr")
        
        if not partition_key or not session_abr:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing event_partition_key or session_abr"})
            }
        
        # Get session from DynamoDB
        session = read_tracking_table(
            event_partition_key=partition_key,
            session_name_abr=session_abr
        )
        
        if not session:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": f"Session not found: {partition_key}/{session_abr}"})
            }
        
        session = session[0]
        
        # Check if already loaded
        if session.get(DATA_LOADED, False):
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Session already loaded",
                    "event_partition_key": partition_key,
                    "session_abr": session_abr
                })
            }
        
        # Process this specific session
        try:
            event_year = int(session[EVENT_YEAR])
            event_name = session[EVENT_NAME]
            session_name = session[SESSION_NAME]
            session_name_abr = session[SESSION_NAME_ABR]
            
            print(f"Orchestrator: Processing {event_year} {event_name} - {session_name}")
            
            # Load session data
            session_results, laps_df = process_session_data(event_year, event_name, session_name)
            
            # Store results to S3
            s3_results = store_results_to_s3(
                session_results, event_year, event_name, session_name_abr, data_type='Results'
            )
            s3_laps = store_results_to_s3(
                laps_df, event_year, event_name, session_name_abr, data_type='Laps'
            )
            
            # Update tracking table
            if s3_results['success']:
                update_tracking_table(
                    partition_key, session_name_abr,
                    attribute_name=S3_RESULTS_LOCATION,
                    attribute_value=s3_results['s3_path']
                )
            
            if s3_laps['success']:
                update_tracking_table(
                    partition_key, session_name_abr,
                    attribute_name=S3_LAPS_LOCATION,
                    attribute_value=s3_laps['s3_path']
                )
            
            if s3_results['success'] and s3_laps['success']:
                update_tracking_table(
                    partition_key, session_name_abr,
                    attribute_name=DATA_LOADED,
                    attribute_value=True
                )
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "message": "Session data loaded successfully",
                        "event_partition_key": partition_key,
                        "session_abr": session_abr,
                        "results_s3": s3_results.get('s3_path'),
                        "laps_s3": s3_laps.get('s3_path')
                    })
                }
            else:
                return {
                    "statusCode": 500,
                    "body": json.dumps({
                        "error": "Failed to upload to S3",
                        "results_success": s3_results['success'],
                        "laps_success": s3_laps['success']
                    })
                }
                
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }
    
    return {
        "statusCode": 400,
        "body": json.dumps({"error": f"Unknown action: {action}"})
    }


def handle_api_gateway_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle traditional API Gateway invocation."""
    # Extract HTTP method and path
    http_method = event.get('httpMethod', '')
    path = event.get('path', '')
    
    # Parse request body if present
    body = None
    if event.get('body'):
        try:
            body = json.loads(event['body'])
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps({"error": "Invalid JSON in request body"})
            }
    
    # Extract headers
    headers = event.get('headers', {})
    
    # Log the incoming request (for debugging)
    print(f"Received {http_method} request to {path}")
    print(f"Headers: {headers}")
    print(f"Body: {body}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        # Log the error
        print(f"Error processing request: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Return error response
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
            },
            'body': json.dumps({"error": "Internal server error", "message": str(e)})
        }

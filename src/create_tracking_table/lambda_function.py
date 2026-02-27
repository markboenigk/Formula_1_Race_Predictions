import json
import os
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import fastf1
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key

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

# Circuit location (for weather and other consumers)
CIRCUIT_LAT = 'circuit_lat'  # Number: Circuit latitude
CIRCUIT_LNG = 'circuit_lng'  # Number: Circuit longitude


from src.common.circuit_coordinates import (
    CIRCUIT_COORDINATES,
    normalize_event_name as _normalize_event_name,
)


def _get_circuit_coords(event_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (lat, lng) for a known circuit; (None, None) for unknown (e.g. new calendar)."""
    slug = _normalize_event_name(event_name)
    if slug not in CIRCUIT_COORDINATES:
        return (None, None)
    return CIRCUIT_COORDINATES[slug]


def get_table_name() -> str:
    """Get the DynamoDB table name from environment variable or use default."""
    return os.environ.get('DYNAMODB_TABLE_NAME', 'f1_session_tracking')


# ============================================================================
# Configuration
# ============================================================================
# DynamoDB Configuration
DYNAMODB_TABLE_NAME = get_table_name()

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

def get_schedules(years: list) -> pd.DataFrame:
    """
    Gets event schedules for the specified years and returns them as a pandas DataFrame.

    Args:
        years: List of years (integers) to retrieve schedules for

    Returns:
        pandas.DataFrame: A DataFrame containing the event schedules for all specified years.
    """
    schedule_list = []
    for year in years:
        event_schedule = fastf1.get_event_schedule(year)
        event_schedule['year'] = year
        event_schedule['event_year_key'] = str(year) + '_' + event_schedule['EventName'].str.strip().str.replace(' ', '_')
        schedule_list.append(event_schedule)
    return pd.concat(schedule_list)


def transform_schedules_to_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the wide format schedule data into a long format with one row per session.
    
    Args:
        df: DataFrame with wide format (one row per event, multiple session columns)
    
    Returns:
        DataFrame with columns: event_partition_key, event_year, event_number, EventName, session_name, 
        session_name_abr, is_before_race, session_date, session_date_utc, event_format
    
    Note:
        - Partition Key: event_partition_key (e.g., "2025_Australian_Grand_Prix")
        - Sort Key: session_name_abr (e.g., "P1", "P2", "Q1", "R")
        - is_before_race: Boolean flag (True for all sessions except Race)
    """

    session_type_dict = {
        'Practice 1': 'P1',
        'Practice 2': 'P2',
        'Practice 3': 'P3',
        'Qualifying': 'Q',
        'Sprint Qualifying': 'SQ',
        'Sprint Shootout': 'SS',  # Sprint Shootout (new format from 2023)
        'Sprint': 'S',
        'Race': 'R'
    }
    sessions_list = []
    
    # Iterate through each event
    for _, row in df.iterrows():
        partition_key = row.get('event_year_key', row.get('event_name', ''))  # Partition key for the table
        event_year = row.get('year', None)
        event_format = row['EventFormat']
        event_number = row.get('RoundNumber', None)  # Get RoundNumber (event number 1-24)
        event_name_original = row.get('EventName', '')  # Get original EventName
        
        # Process each session (Session1 through Session5)
        for session_num in range(1, 6):
            session_name_col = f'Session{session_num}'
            session_date_col = f'Session{session_num}Date'
            session_date_utc_col = f'Session{session_num}DateUtc'
            
            # Check if session exists (not None)
            session_name_raw = row.get(session_name_col)
            # Check for None, NaN, or the string "None"
            if (pd.notna(session_name_raw) and 
                session_name_raw is not None and 
                str(session_name_raw).strip().lower() != 'none' and
                str(session_name_raw).strip() != ''):
                # Normalize session_name - convert to string and strip whitespace
                session_name = str(session_name_raw).strip()
                
                session_date = row.get(session_date_col)
                session_date_utc_raw = row.get(session_date_utc_col)
                
                # Normalize session_date_utc to ISO 8601 format for better DynamoDB querying
                # Convert to ISO 8601 format (e.g., "2018-03-23T01:00:00Z")
                session_date_utc = None
                if pd.notna(session_date_utc_raw):
                    try:
                        # If it's already a datetime object, format it
                        if isinstance(session_date_utc_raw, pd.Timestamp):
                            session_date_utc = session_date_utc_raw.strftime('%Y-%m-%dT%H:%M:%SZ')
                        elif isinstance(session_date_utc_raw, datetime):
                            session_date_utc = session_date_utc_raw.strftime('%Y-%m-%dT%H:%M:%SZ')
                        elif isinstance(session_date_utc_raw, str):
                            # Parse string and convert to ISO 8601
                            # Handle formats like "2018-03-23 01:00:00"
                            dt = pd.to_datetime(session_date_utc_raw)
                            session_date_utc = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                        else:
                            session_date_utc = str(session_date_utc_raw)
                    except Exception as e:
                        # Fallback to original value if conversion fails
                        print(f"Warning: Could not convert session_date_utc to ISO 8601: {e}")
                        session_date_utc = str(session_date_utc_raw) if session_date_utc_raw else None
                
                # Determine if session is before race (True for all sessions except Race)
                is_before_race = session_name != 'Race'
                
                # Get session abbreviation - normalize session_name first, then lookup
                # This ensures exact match with dictionary keys
                session_name_abr = session_type_dict.get(session_name, '')
                
                # If still empty, use first letter of session_name as fallback
                # This should never happen if dictionary is complete, but provides safety
                if not session_name_abr:
                    if session_name:
                        session_name_abr = session_name[0].upper()
                        # Log warning for debugging - this indicates a missing dictionary entry
                        print(f"Warning: session_name '{session_name}' not in dictionary, using fallback '{session_name_abr}' for event {partition_key}")
                    else:
                        # This should never happen since we check for None/NaN above
                        print(f"Warning: session_name is empty for event {partition_key}")
                        session_name_abr = 'UNK'  # Unknown as last resort
                
                circuit_lat, circuit_lng = _get_circuit_coords(event_name_original)
                session_record = {
                    PARTITION_KEY: partition_key,  # Partition Key
                    EVENT_YEAR: event_year,
                    EVENT_NUMBER: event_number,
                    EVENT_NAME: event_name_original,
                    SESSION_NAME: session_name,
                    SESSION_NAME_ABR: session_name_abr,  # Sort Key (guaranteed non-empty)
                    IS_BEFORE_RACE: is_before_race,  # Boolean: True if session is before race
                    SESSION_DATE: session_date,
                    SESSION_DATE_UTC: session_date_utc,
                    EVENT_FORMAT: event_format,
                    DATA_LOADED: False  # Initialize data_loaded field
                }
                if circuit_lat is not None and circuit_lng is not None:
                    session_record[CIRCUIT_LAT] = circuit_lat
                    session_record[CIRCUIT_LNG] = circuit_lng
                sessions_list.append(session_record)
    
    return pd.DataFrame(sessions_list)


def upsert_sessions_to_dynamodb(sessions_df: pd.DataFrame, table_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Upsert sessions data to DynamoDB table.
    
    Args:
        sessions_df: DataFrame containing session data to upsert
        table_name: Name of the DynamoDB table (defaults to config value)
    
    Returns:
        Dictionary with upsert statistics (successful, failed, errors)
    """
    if table_name is None:
        table_name = get_table_name()
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    successful = 0
    failed = 0
    errors = []
    
    # Convert DataFrame to list of dictionaries
    sessions_list = sessions_df.to_dict('records')
    
    # Deduplicate sessions by (event_partition_key, session_name_abr) - keep the last occurrence
    # This prevents duplicate key errors in DynamoDB
    seen_keys = {}
    deduplicated_sessions = []
    for session in sessions_list:
        partition_key = str(session.get(PARTITION_KEY, ''))
        session_name_abr = str(session.get(SESSION_NAME_ABR, '')).strip()
        
        # Skip if either key is empty
        if not partition_key or not session_name_abr:
            continue
            
        key = (partition_key, session_name_abr)
        if key in seen_keys:
            # Replace with newer session (in case data was updated)
            idx = seen_keys[key]
            deduplicated_sessions[idx] = session
        else:
            seen_keys[key] = len(deduplicated_sessions)
            deduplicated_sessions.append(session)
    
    print(f"Deduplicated {len(sessions_list)} sessions to {len(deduplicated_sessions)} unique sessions")

    # Upsert each session: preserve existing data_loaded and S3 pointers when the item already has them
    for session in deduplicated_sessions:
        try:
            session_name_abr = session.get(SESSION_NAME_ABR, '')
            session_name = session.get(SESSION_NAME, '')

            if not session_name_abr or pd.isna(session_name_abr) or str(session_name_abr).strip() == '':
                if session_name and pd.notna(session_name) and str(session_name).strip():
                    session_name_abr = str(session_name)[0].upper()
                else:
                    failed += 1
                    errors.append(f"Skipping session {session.get(PARTITION_KEY, 'unknown')}: empty session_name_abr and session_name")
                    continue

            session_name_abr = str(session_name_abr).strip()
            if not session_name_abr:
                failed += 1
                errors.append(f"Skipping session {session.get(PARTITION_KEY, 'unknown')}: empty session_name_abr after processing")
                continue

            partition_key = str(session[PARTITION_KEY])

            # Preserve existing data_loaded and S3 locations if this session was already ingested
            existing = table.get_item(
                Key={PARTITION_KEY: partition_key, SORT_KEY: session_name_abr}
            ).get('Item')

            data_loaded = False
            s3_results_location = None
            s3_laps_location = None
            if existing:
                if existing.get(DATA_LOADED) is True:
                    data_loaded = True
                if existing.get(S3_RESULTS_LOCATION):
                    s3_results_location = existing.get(S3_RESULTS_LOCATION)
                if existing.get(S3_LAPS_LOCATION):
                    s3_laps_location = existing.get(S3_LAPS_LOCATION)

            item = {
                PARTITION_KEY: partition_key,
                SORT_KEY: session_name_abr,
                EVENT_YEAR: int(session[EVENT_YEAR]) if pd.notna(session.get(EVENT_YEAR)) else None,
                EVENT_NUMBER: int(session[EVENT_NUMBER]) if pd.notna(session.get(EVENT_NUMBER)) else None,
                EVENT_NAME: str(session[EVENT_NAME]) if pd.notna(session.get(EVENT_NAME)) else '',
                SESSION_NAME: str(session[SESSION_NAME]) if pd.notna(session.get(SESSION_NAME)) else '',
                IS_BEFORE_RACE: bool(session[IS_BEFORE_RACE]) if pd.notna(session.get(IS_BEFORE_RACE)) else False,
                SESSION_DATE: str(session[SESSION_DATE]) if pd.notna(session.get(SESSION_DATE)) else None,
                SESSION_DATE_UTC: str(session[SESSION_DATE_UTC]) if pd.notna(session.get(SESSION_DATE_UTC)) else None,
                EVENT_FORMAT: str(session[EVENT_FORMAT]) if pd.notna(session.get(EVENT_FORMAT)) else '',
                DATA_LOADED: data_loaded,
            }
            if s3_results_location is not None:
                item[S3_RESULTS_LOCATION] = s3_results_location
            if s3_laps_location is not None:
                item[S3_LAPS_LOCATION] = s3_laps_location
            lat, lng = session.get(CIRCUIT_LAT), session.get(CIRCUIT_LNG)
            if pd.notna(lat) and pd.notna(lng) and lat is not None and lng is not None:
                item[CIRCUIT_LAT] = Decimal(str(lat))
                item[CIRCUIT_LNG] = Decimal(str(lng))

            item = {k: v for k, v in item.items() if v is not None}

            table.put_item(Item=item)
            successful += 1

        except Exception as e:
            failed += 1
            error_msg = f"Failed to upsert session {session.get(PARTITION_KEY, 'unknown')}/{session.get(SESSION_NAME_ABR, 'unknown')}: {str(e)}"
            errors.append(error_msg)
            print(error_msg)
    
    result = {
        'successful': successful,
        'failed': failed,
        'total': len(sessions_list),
        'deduplicated_total': len(deduplicated_sessions),
        'errors': errors[:10]  # Limit to first 10 errors to avoid huge response
    }
    
    print(f"Upsert complete: {successful} successful, {failed} failed out of {len(deduplicated_sessions)} deduplicated sessions (from {len(sessions_list)} total)")
    
    return result



def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event object (API Gateway event in this case)
        context: Lambda context object
    
    Returns:
        API Gateway response dictionary
    """
    try:
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
                        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
                    },
                    'body': json.dumps({"error": "Invalid JSON in request body"})
                }
        
        # Extract headers
        headers = event.get('headers', {})
        
        # Log the incoming request (for debugging)
        print(f"Received {http_method} request to {path}")
        print(f"Headers: {headers}")
        print(f"Body: {body}")
        
        # Extract years from request body, default to [2025] if not provided
        years = body.get('years', [2025]) if body else [2025]
        if not isinstance(years, list):
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
                },
                'body': json.dumps({"error": "Invalid request: 'years' must be a list"})
            }
        
        print(f"Processing years: {years}")
        
        # Get schedules and transform to session format
        all_schedules = get_schedules(years)
        sessions_df = transform_schedules_to_sessions(all_schedules)
        #all_schedules.to_csv('all_schedules.csv', index=False)
        print(f"Transformed {len(sessions_df)} sessions")
        print(sessions_df.head())
        
        # Upsert sessions to DynamoDB
        upsert_result = upsert_sessions_to_dynamodb(sessions_df)
        
        # Save the transformed data to CSV (for local testing/debugging)
        sessions_df.to_csv('all_schedules.csv', index=False)

        if path == '/sync' and http_method == 'POST':
            # Your processing logic here
            result = {
                "message": "Lambda function executed successfully",
                "path": path,
                "method": http_method,
                "years_processed": years,
                "sessions_count": len(sessions_df),
                "dynamodb_upsert": upsert_result,
                "processed_data": body
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
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
                    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
                },
                'body': json.dumps({"error": f"Path {path} with method {http_method} not found"})
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



import json
import os
import io
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Attr
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================================
# DynamoDB Schema Constants
# ============================================================================
# Primary Key Fields
PARTITION_KEY = 'event_partition_key'  # e.g., "2025_Australian_Grand_Prix"
SORT_KEY = 'session_name_abr'  # e.g., "P1", "Q1", "R"

# Data Loading Status Fields
S3_LAPS_LOCATION = 's3_laps_location'  # String: S3 path to laps Parquet file
S3_RESULTS_LOCATION = 's3_results_location'  # String: S3 path to results Parquet file

# ============================================================================
# Configuration
# ============================================================================
# DynamoDB Configuration
def get_table_name() -> str:
    """Get the DynamoDB table name from environment variable or use default."""
    return os.environ.get('DYNAMODB_TABLE_NAME', 'f1_session_tracking')

DYNAMODB_TABLE_NAME = get_table_name()

# S3 Configuration
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'f1-race-prediction')
S3_BRONZE_PATH = os.environ.get('S3_BRONZE_PATH') or os.environ.get('S3_WL_BRONZE_PATH', 'bronze')
S3_SILVER_PATH = os.environ.get('S3_SILVER_PATH', 'silver')
S3_SILVER_OVERWRITE = os.environ.get('S3_SILVER_OVERWRITE', 'true').strip().lower() in ('1', 'true', 'yes', 'y')


def _delete_s3_prefix(
    bucket: str,
    prefix: str,
    s3_client: Optional[Any] = None
) -> int:
    """
    Delete all objects under an S3 prefix.

    Returns number of objects deleted.
    """
    if s3_client is None:
        s3_client = boto3.client('s3')

    deleted = 0

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get('Contents', [])
        if not contents:
            continue

        # Delete in batches of 1000 (S3 API limit)
        keys = [{'Key': obj['Key']} for obj in contents]
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            resp = s3_client.delete_objects(Bucket=bucket, Delete={'Objects': batch})
            deleted += len(resp.get('Deleted', []))

    return deleted


def write_partitioned_parquet_dataset_to_s3(
    df: pd.DataFrame,
    dataset_name: str,
    partition_cols: List[str],
    bucket_name: Optional[str] = None,
    silver_prefix: Optional[str] = None,
    overwrite: Optional[bool] = None,
    s3_client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Write a Hive-partitioned Parquet dataset to S3 by grouping a DataFrame on partition columns.

    This avoids requiring direct filesystem-backed S3 writers by writing per-partition Parquet
    bytes and uploading with boto3.
    """
    if df.empty:
        return {"success": True, "dataset": dataset_name, "files_written": 0, "rows": 0}

    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    if silver_prefix is None:
        silver_prefix = S3_SILVER_PATH
    if overwrite is None:
        overwrite = S3_SILVER_OVERWRITE
    if s3_client is None:
        s3_client = boto3.client('s3')

    missing = [c for c in partition_cols if c not in df.columns]
    if missing:
        return {"success": False, "dataset": dataset_name, "error": f"Missing partition columns: {missing}"}

    base_prefix = f"{silver_prefix.rstrip('/')}/{dataset_name}/"

    deleted = 0
    if overwrite:
        deleted = _delete_s3_prefix(bucket=bucket_name, prefix=base_prefix, s3_client=s3_client)
        print(f"Deleted {deleted} existing objects under s3://{bucket_name}/{base_prefix}")

    files_written = 0
    errors: List[str] = []

    # Group by partition columns; write one parquet file per partition group
    grouped = df.groupby(partition_cols, dropna=False, sort=False)
    for part_values, part_df in grouped:
        # Normalize scalar vs tuple
        if not isinstance(part_values, tuple):
            part_values = (part_values,)

        # Build Hive partition path pieces like col=value
        partition_parts: List[str] = []
        for col, val in zip(partition_cols, part_values):
            if pd.isna(val):
                val_str = "__NULL__"
            else:
                val_str = str(val)
            # Keep paths safe-ish
            val_str = val_str.strip().replace("/", "_")
            partition_parts.append(f"{col}={val_str}")

        partition_prefix = base_prefix + "/".join(partition_parts) + "/"
        filename = f"part-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}.parquet"
        key = partition_prefix + filename

        try:
            # Do not store partition columns inside the parquet file itself.
            # When reading Hive-partitioned datasets, engines derive partition columns from the path.
            # Keeping them in-file can cause schema merge conflicts (e.g., int vs dictionary encodings).
            data_df = part_df.drop(columns=partition_cols, errors="ignore")
            table = pa.Table.from_pandas(data_df, preserve_index=False)
            buf = io.BytesIO()
            pq.write_table(table, buf, compression="snappy")
            buf.seek(0)

            s3_client.upload_fileobj(
                buf,
                bucket_name,
                key,
                ExtraArgs={
                    "ContentType": "application/octet-stream",
                    "ServerSideEncryption": "AES256"
                }
            )
            files_written += 1
        except Exception as e:
            errors.append(f"Failed writing partition {partition_parts}: {str(e)}")

    return {
        "success": len(errors) == 0,
        "dataset": dataset_name,
        "rows": len(df),
        "files_written": files_written,
        "deleted_existing_objects": deleted,
        "errors": errors[:10]
    }


def get_sessions_with_s3_locations(
    s3_location_field: str = S3_LAPS_LOCATION,
    table_name: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Query DynamoDB to find all sessions that have a specified S3 location attribute.
    
    Args:
        s3_location_field: The S3 location field to filter on (e.g., S3_LAPS_LOCATION or S3_RESULTS_LOCATION)
        table_name: Name of the DynamoDB table (defaults to config value)
        limit: Optional limit on number of sessions to return
    
    Returns:
        List of dictionaries containing session data with the specified S3 location field
    """
    if table_name is None:
        table_name = get_table_name()
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        # Filter for sessions where the specified S3 location field exists.
        # Fetch all matching sessions (full pagination); at ~400 sessions this is fine.
        filter_expression = Attr(s3_location_field).exists()
        items = []
        scan_kwargs = {'FilterExpression': filter_expression}
        response = table.scan(**scan_kwargs)
        items.extend(response.get('Items', []))
        while 'LastEvaluatedKey' in response:
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = table.scan(**scan_kwargs)
            items.extend(response.get('Items', []))
        if limit is not None and limit > 0:
            items = items[:limit]
        print(f"Found {len(items)} sessions with {s3_location_field}")
        return items
    
    except Exception as e:
        print(f"Error querying DynamoDB table: {str(e)}")
        raise



def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse an S3 path (s3://bucket/key) into bucket and key.
    
    Args:
        s3_path: Full S3 path (e.g., "s3://bucket-name/path/to/file.parquet")
    
    Returns:
        Tuple of (bucket_name, key)
    """
    # Remove s3:// prefix if present
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    
    # Split into bucket and key
    parts = s3_path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket, key


def list_parquet_files_in_s3_folder(
    bucket: str,
    prefix: str,
    s3_client: Optional[Any] = None
) -> List[str]:
    """
    List all parquet files in an S3 folder and its subfolders.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix/folder path (e.g., "bronze/season=2024/event=monaco/session=R/Laps/")
        s3_client: Optional boto3 S3 client (creates one if not provided)
    
    Returns:
        List of S3 keys (full paths) to parquet files
    """
    if s3_client is None:
        s3_client = boto3.client('s3')
    
    parquet_files = []
    
    try:
        # Ensure prefix ends with / if it's a folder
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if file is a parquet file
                    if key.endswith('.parquet'):
                        s3_path = f"s3://{bucket}/{key}"
                        parquet_files.append(s3_path)
        
        print(f"Found {len(parquet_files)} parquet files in s3://{bucket}/{prefix}")
        return parquet_files
    
    except Exception as e:
        print(f"Error listing parquet files in S3: {str(e)}")
        raise


def read_parquet_from_s3(
    s3_path: str,
    s3_client: Optional[Any] = None
) -> pd.DataFrame:
    """
    Read a parquet file from S3 into a pandas DataFrame.
    
    Args:
        s3_path: Full S3 path (e.g., "s3://bucket-name/path/to/file.parquet")
        s3_client: Optional boto3 S3 client (creates one if not provided)
    
    Returns:
        pandas DataFrame
    """
    if s3_client is None:
        s3_client = boto3.client('s3')
    
    try:
        bucket, key = parse_s3_path(s3_path)
        
        # Read parquet file directly from S3
        df = pd.read_parquet(f"s3://{bucket}/{key}", engine='pyarrow')
        print(f"Read {len(df)} rows from {s3_path}")
        return df
    
    except Exception as e:
        print(f"Error reading parquet file from S3 {s3_path}: {str(e)}")
        raise


def _read_parquet_file_wrapper(s3_path: str) -> Optional[pd.DataFrame]:
    """
    Wrapper function for parallel parquet file reading.
    
    Args:
        s3_path: Full S3 path to parquet file
    
    Returns:
        DataFrame or None if reading failed
    """
    try:
        return read_parquet_from_s3(s3_path)
    except Exception as e:
        print(f"Failed to read {s3_path}: {str(e)}")
        return None


def _list_files_for_session(
    session: Dict[str, Any],
    s3_location_field: str,
    s3_client: Any
) -> tuple[List[str], str]:
    """
    Helper function to list parquet files for a single session (for parallel execution).
    
    Args:
        session: Session dictionary from DynamoDB
        s3_location_field: The S3 location field to use
        s3_client: boto3 S3 client
    
    Returns:
        Tuple of (list of parquet file paths, session_key for error reporting)
    """
    session_key = session.get(PARTITION_KEY, 'unknown')
    s3_location = session.get(s3_location_field)
    
    if not s3_location:
        return [], session_key
    
    try:
        bucket, key = parse_s3_path(s3_location)
        folder_path = '/'.join(key.split('/')[:-1])  # Remove filename, keep folder path
        parquet_files = list_parquet_files_in_s3_folder(bucket, folder_path, s3_client)
        return parquet_files, session_key
    except Exception as e:
        print(f"Error listing files for session {session_key}: {str(e)}")
        return [], session_key


def read_all_data_from_s3_locations(
    sessions: List[Dict[str, Any]],
    s3_location_field: str = S3_LAPS_LOCATION,
    max_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Read all parquet files from S3 locations stored in DynamoDB sessions.
    
    Since the S3 location field points to a specific file, this function extracts the folder path
    from the file path and reads all parquet files in that folder and subfolders.
    
    This function uses parallel processing to read multiple files concurrently, significantly
    improving performance when processing many sessions.
    
    Args:
        sessions: List of session dictionaries from DynamoDB with the specified S3 location attribute
        s3_location_field: The S3 location field to use (e.g., S3_LAPS_LOCATION or S3_RESULTS_LOCATION)
        max_workers: Maximum number of parallel workers for file reading (default: min(32, num_files + 4))
    
    Returns:
        Combined pandas DataFrame with all data
    """
    s3_client = boto3.client('s3')
    
    # Step 1: Collect all parquet file paths
    # Parallelize file listing if we have many sessions (threshold: 10)
    all_parquet_files = []
    session_file_map = {}  # Track which files belong to which session for error reporting
    
    print(f"Collecting parquet file paths from {len(sessions)} sessions...")
    
    if len(sessions) > 10:
        # Parallelize file listing for many sessions
        print(f"Using parallel file listing for {len(sessions)} sessions...")
        with ThreadPoolExecutor(max_workers=min(16, len(sessions))) as executor:
            future_to_session = {
                executor.submit(_list_files_for_session, session, s3_location_field, s3_client): session
                for session in sessions
            }
            
            for future in as_completed(future_to_session):
                session = future_to_session[future]
                try:
                    parquet_files, session_key = future.result()
                    if parquet_files:
                        all_parquet_files.extend(parquet_files)
                        for parquet_file in parquet_files:
                            session_file_map[parquet_file] = session_key
                except Exception as e:
                    print(f"Error processing session {session.get(PARTITION_KEY, 'unknown')}: {str(e)}")
    else:
        # Sequential file listing for small number of sessions
        for session in sessions:
            s3_location = session.get(s3_location_field)
            
            if not s3_location:
                print(f"Skipping session {session.get(PARTITION_KEY, 'unknown')}: no {s3_location_field}")
                continue
            
            try:
                bucket, key = parse_s3_path(s3_location)
                
                # Extract folder path from file path (remove filename)
                # e.g., "bronze/season=2024/event=monaco/session=R/Laps/file.parquet" 
                # -> "bronze/season=2024/event=monaco/session=R/Laps/"
                folder_path = '/'.join(key.split('/')[:-1])  # Remove filename, keep folder path
                
                # List all parquet files in the folder and subfolders
                parquet_files = list_parquet_files_in_s3_folder(bucket, folder_path, s3_client)
                
                if not parquet_files:
                    print(f"No parquet files found in folder: s3://{bucket}/{folder_path}")
                    continue
                
                all_parquet_files.extend(parquet_files)
                # Track files per session for better error reporting
                for parquet_file in parquet_files:
                    session_file_map[parquet_file] = session.get(PARTITION_KEY, 'unknown')
            
            except Exception as e:
                print(f"Error processing session {session.get(PARTITION_KEY, 'unknown')}: {str(e)}")
                continue
    
    if not all_parquet_files:
        print("No parquet files found to read")
        return pd.DataFrame()
    
    print(f"Found {len(all_parquet_files)} parquet files to read. Starting parallel reading...")
    
    # Step 2: Read all files in parallel
    # Use ThreadPoolExecutor for I/O-bound operations (S3 reads)
    # Default max_workers: min(32, num_files + 4) to avoid overwhelming S3
    if max_workers is None:
        max_workers = min(32, len(all_parquet_files) + 4)
    
    dataframes_list = []
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all read tasks
        future_to_file = {
            executor.submit(_read_parquet_file_wrapper, parquet_file): parquet_file
            for parquet_file in all_parquet_files
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            parquet_file = future_to_file[future]
            completed += 1
            
            try:
                df = future.result()
                if df is not None and not df.empty:
                    dataframes_list.append(df)
                elif df is None:
                    failed_files.append(parquet_file)
            except Exception as e:
                print(f"Unexpected error processing {parquet_file}: {str(e)}")
                failed_files.append(parquet_file)
            
            # Progress logging for large batches
            if completed % 10 == 0 or completed == len(all_parquet_files):
                print(f"Progress: {completed}/{len(all_parquet_files)} files read "
                      f"({len(dataframes_list)} successful, {len(failed_files)} failed)")
    
    if failed_files:
        print(f"Warning: {len(failed_files)} files failed to read")
        # Optionally log which sessions had failures
        failed_sessions = set(session_file_map.get(f, 'unknown') for f in failed_files)
        if len(failed_sessions) <= 10:  # Only log if not too many
            print(f"Failed sessions: {', '.join(failed_sessions)}")
    
    if not dataframes_list:
        print("No dataframes successfully loaded")
        return pd.DataFrame()
    
    # Step 3: Combine all dataframes
    print(f"Combining {len(dataframes_list)} dataframes...")
    combined_df = pd.concat(dataframes_list, ignore_index=True)
    print(f"Combined {len(dataframes_list)} dataframes into one with {len(combined_df)} total rows")
    
    return combined_df


def convert_to_snake_case(name: str) -> str:
    """
    Convert column name to snake_case.
    
    Examples:
        DriverNumber -> driver_number
        IsPersonalBest -> is_personal_best
        Time_seconds -> time_seconds
    """
    import re
    # Insert underscore before uppercase letters (except at start)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letters that follow lowercase or numbers
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def transform_laps_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform combined laps DataFrame from bronze to silver layer.
    
    Transformations:
    1. Convert column names to snake_case
    2. Remove Timedelta columns, keep only _seconds versions
    3. Standardize data types (integers, booleans, floats)
    4. Remove redundant columns
    5. Standardize categorical values
    6. Handle missing values appropriately
    
    Args:
        df: Combined laps DataFrame from bronze layer
    
    Returns:
        Transformed DataFrame ready for silver layer
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 0. Convert empty strings to NaN for better null handling
    # This ensures empty strings are treated as missing values throughout
    df = df.replace(['', 'nan', 'None', 'null', 'NULL'], pd.NA)
    
    # 1. Convert column names to snake_case
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    
    # 2. Remove Timedelta columns (keep only _seconds versions)
    timedelta_cols_to_remove = [
        'time', 'lap_time', 'pit_out_time', 'pit_in_time',
        'sector1_time', 'sector2_time', 'sector3_time',
        'sector1_session_time', 'sector2_session_time', 'sector3_session_time',
        'lap_start_time'
    ]
    for col in timedelta_cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Rename _seconds columns to remove suffix (they're now the primary time columns)
    time_rename_map = {
        'time_seconds': 'time_seconds',
        'lap_time_seconds': 'lap_time_seconds',
        'pit_out_time_seconds': 'pit_out_time_seconds',
        'pit_in_time_seconds': 'pit_in_time_seconds',
        'sector1_time_seconds': 'sector1_time_seconds',
        'sector2_time_seconds': 'sector2_time_seconds',
        'sector3_time_seconds': 'sector3_time_seconds',
        'sector1_session_time_seconds': 'sector1_session_time_seconds',
        'sector2_session_time_seconds': 'sector2_session_time_seconds',
        'sector3_session_time_seconds': 'sector3_session_time_seconds',
        'lap_start_time_seconds': 'lap_start_time_seconds'
    }
    # Actually, let's keep the _seconds suffix for clarity, or remove it - user preference
    # For now, keeping _seconds suffix to be explicit
    
    # 3. Standardize data types
    # Integer columns - handle empty strings properly
    int_columns = ['driver_number', 'lap_number', 'stint', 'tyre_life', 'position']
    for col in int_columns:
        if col in df.columns:
            # Convert empty strings to NaN before conversion
            if df[col].dtype == 'object':
                df[col] = df[col].replace(['', 'nan', 'None', None], pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Nullable integer
    
    # Boolean columns - handle both naming variations
    # Use nullable boolean dtype to preserve NULL values
    bool_columns = ['is_personal_best', 'fresh_tyre', 'deleted', 'fastf1_generated', 'fast_f1_generated', 'is_accurate']
    # Standardize fast_f1_generated naming
    if 'fast_f1_generated' in df.columns and 'fastf1_generated' not in df.columns:
        df['fastf1_generated'] = df['fast_f1_generated']
        df = df.drop(columns=['fast_f1_generated'])
    elif 'fast_f1_generated' in df.columns and 'fastf1_generated' in df.columns:
        # Merge them (prefer fastf1_generated)
        df['fastf1_generated'] = df['fastf1_generated'].fillna(df['fast_f1_generated'])
        df = df.drop(columns=['fast_f1_generated'])
    
    for col in bool_columns:
        if col in df.columns:
            # Convert string representations to proper boolean/null values
            if df[col].dtype == 'object':
                # Replace string representations
                df[col] = df[col].replace(['False', 'false', '0', 0], False)
                df[col] = df[col].replace(['True', 'true', '1', 1], True)
                # Empty strings and other null-like values become pd.NA (already handled in step 0)
            # Convert to nullable boolean dtype
            df[col] = df[col].astype('boolean')  # Pandas nullable boolean dtype
    
    # Float columns (speed columns) - use nullable float
    float_columns = ['speed_i1', 'speed_i2', 'speed_fl', 'speed_st']
    for col in float_columns:
        if col in df.columns:
            # Convert empty strings to NaN before conversion
            if df[col].dtype == 'object':
                df[col] = df[col].replace(['', 'nan', 'None', None], pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')  # Nullable float
    
    # 4. Remove redundant columns
    # Keep `event` for partitioning/joining downstream (it is a stable, path-safe identifier).
    redundant_cols = ['season']  # event_year already exists
    if 'session' in df.columns and 'session_name' in df.columns:
        # session is abbreviation, session_name is full name - keep both for now
        # But if you want to remove one, remove 'session' (abbreviation can be derived)
        pass
    for col in redundant_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 5. Standardize categorical values
    # Compound: Ensure uppercase (already appears to be)
    if 'compound' in df.columns:
        df['compound'] = df['compound'].str.upper()
    
    # 6. Handle missing values
    # TrackStatus: Integer status codes representing track conditions
    # Common values observed: 1 (clear/green), 12, 15 (yellow flags), 21, 152 (various conditions)
    # NULLs are preserved for handling in gold layer
    if 'track_status' in df.columns:
        # Convert to numeric, preserving NULLs
        df['track_status'] = pd.to_numeric(df['track_status'], errors='coerce').astype('Int64')  # Preserves NULLs
    
    # 7. Standardize date column
    if 'lap_start_date' in df.columns:
        # Convert to datetime if it's a string
        if df['lap_start_date'].dtype == 'object':
            df['lap_start_date'] = pd.to_datetime(df['lap_start_date'], errors='coerce')
    
    # 8. Add derived columns (optional but useful)
    # is_pit_lap: True if pit_in_time or pit_out_time exists
    if 'pit_in_time_seconds' in df.columns and 'pit_out_time_seconds' in df.columns:
        df['is_pit_lap'] = (
            df['pit_in_time_seconds'].notna() | 
            df['pit_out_time_seconds'].notna()
        )
    
    # Round time columns to 3 decimal places (milliseconds precision)
    time_cols = [
        'time_seconds', 'lap_time_seconds', 'lap_start_time_seconds',
        'sector1_time_seconds', 'sector2_time_seconds', 'sector3_time_seconds',
        'pit_in_time_seconds', 'pit_out_time_seconds',
        'sector1_session_time_seconds', 'sector2_session_time_seconds', 'sector3_session_time_seconds'
    ]
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].round(3)
    
    # total_sector_time: Sum of all three sectors (for validation)
    sector_cols = ['sector1_time_seconds', 'sector2_time_seconds', 'sector3_time_seconds']
    if all(col in df.columns for col in sector_cols):
        df['total_sector_time_seconds'] = (
            df['sector1_time_seconds'].fillna(0) + 
            df['sector2_time_seconds'].fillna(0) + 
            df['sector3_time_seconds'].fillna(0)
        )
        # Replace 0 with NaN if all sectors were missing
        mask = (
            df['sector1_time_seconds'].isna() & 
            df['sector2_time_seconds'].isna() & 
            df['sector3_time_seconds'].isna()
        )
        df.loc[mask, 'total_sector_time_seconds'] = pd.NA
        # Round the total
        df['total_sector_time_seconds'] = df['total_sector_time_seconds'].round(3)
        
        # Add validation flag: sector sum should match lap_time when all sectors exist
        if 'lap_time_seconds' in df.columns:
            all_sectors_present = (
                df['sector1_time_seconds'].notna() & 
                df['sector2_time_seconds'].notna() & 
                df['sector3_time_seconds'].notna() &
                df['lap_time_seconds'].notna()
            )
            if all_sectors_present.any():
                time_diff = (df.loc[all_sectors_present, 'total_sector_time_seconds'] - 
                           df.loc[all_sectors_present, 'lap_time_seconds']).abs()
                # Flag discrepancies > 0.1 seconds (100ms tolerance for rounding)
                df['sector_time_mismatch'] = False
                mismatch_mask = all_sectors_present & (time_diff > 0.1)
                df.loc[mismatch_mask, 'sector_time_mismatch'] = True
                if mismatch_mask.any():
                    print(f"Warning: {mismatch_mask.sum()} laps have sector time mismatches > 0.1s")
    
    # session_type: Extract type from session_name (Practice, Qualifying, Race, Sprint)
    # Note: Order matters - more specific checks (shootout) must come before general (sprint)
    if 'session_name' in df.columns:
        def extract_session_type(session_name: str) -> str:
            if pd.isna(session_name):
                return None
            session_lower = str(session_name).lower()
            # More specific checks first
            if 'shootout' in session_lower:
                return 'sprint_shootout'  # SS - 2023 format
            elif 'sprint qualifying' in session_lower:
                return 'sprint_qualifying'  # SQ - 2024+ format
            elif 'practice' in session_lower:
                return 'practice'
            elif 'race' in session_lower or session_lower == 'r':
                return 'race'
            elif 'qualifying' in session_lower or session_lower.startswith('q'):
                return 'qualifying'  # Q - conventional
            elif 'sprint' in session_lower or session_lower == 's':
                return 'sprint'  # S - Sprint race
            return 'unknown'
        
        df['session_type'] = df['session_name'].apply(extract_session_type)
    
    # Sort columns for better organization (optional)
    # Group: identifiers, times, speeds, flags, metadata
    priority_cols = [
        'event_year', 'event_name', 'session_name', 'session_type', 'session',
        'driver', 'driver_number', 'team', 'lap_number', 'stint',
        'time_seconds', 'lap_time_seconds', 'lap_start_time_seconds',
        'sector1_time_seconds', 'sector2_time_seconds', 'sector3_time_seconds',
        'total_sector_time_seconds', 'sector_time_mismatch',
        'pit_in_time_seconds', 'pit_out_time_seconds', 'is_pit_lap',
        'speed_i1', 'speed_i2', 'speed_fl', 'speed_st',
        'compound', 'tyre_life', 'fresh_tyre',
        'position', 'track_status',
        'is_personal_best', 'is_accurate', 'deleted', 'fastf1_generated',
        'lap_start_date',
        'sector1_session_time_seconds', 'sector2_session_time_seconds', 'sector3_session_time_seconds',
        'deleted_reason'
    ]
    
    # Reorder columns: priority first (only if they exist), then remaining
    existing_priority_cols = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority_cols]
    df = df[existing_priority_cols + remaining_cols]
    
    return df


def transform_results_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform combined results DataFrame from bronze to silver layer.
    
    Transformations:
    1. Convert column names to snake_case
    2. Remove Timedelta columns, keep only _seconds versions
    3. Standardize data types (integers, booleans, floats)
    4. Remove redundant columns
    5. Standardize categorical values
    6. Handle missing values appropriately
    7. Create derived columns (is_retired, session_type, etc.)
    
    Args:
        df: Combined results DataFrame from bronze layer
    
    Returns:
        Transformed DataFrame ready for silver layer
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 0. Convert empty strings to NaN for better null handling
    df = df.replace(['', 'nan', 'None', 'null', 'NULL'], pd.NA)
    
    # 1. Convert column names to snake_case
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    
    # 2. Remove Timedelta columns (keep only _seconds versions)
    timedelta_cols_to_remove = ['q1', 'q2', 'q3', 'time']
    for col in timedelta_cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 3. Handle position columns with "R" (retired) values
    # Create is_retired flag before converting positions
    if 'position' in df.columns:
        df['is_retired'] = df['position'].astype(str).str.upper() == 'R'
        # Convert "R" to NULL in position column
        df['position'] = df['position'].replace(['R', 'r'], pd.NA)
    
    if 'classified_position' in df.columns:
        # classified_position might also have "R"
        df['classified_position'] = df['classified_position'].replace(['R', 'r'], pd.NA)
        # Update is_retired if classified_position is "R" but position wasn't
        if 'is_retired' in df.columns:
            df['is_retired'] = df['is_retired'] | (df['classified_position'].astype(str).str.upper() == 'R')
    
    # Create is_classified flag (has a numeric classified position)
    if 'classified_position' in df.columns:
        df['is_classified'] = pd.to_numeric(df['classified_position'], errors='coerce').notna()
    
    # Rename position columns for clarity
    if 'position' in df.columns:
        df = df.rename(columns={'position': 'finishing_position'})
    if 'grid_position' in df.columns:
        df = df.rename(columns={'grid_position': 'grid_starting_position'})
    
    # 4. Standardize data types
    # Integer columns - handle empty strings properly
    int_columns = ['driver_number', 'finishing_position', 'classified_position', 'grid_starting_position', 'laps', 'event_year']
    for col in int_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace(['', 'nan', 'None', None], pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Nullable integer
    
    # Float columns (time and points) - use nullable float
    float_columns = ['q1_seconds', 'q2_seconds', 'q3_seconds', 'time_seconds', 'points']
    for col in float_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace(['', 'nan', 'None', None], pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')  # Nullable float
    
    # Round time columns to 3 decimal places (milliseconds precision)
    time_cols = ['q1_seconds', 'q2_seconds', 'q3_seconds', 'time_seconds']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].round(3)
    
    # String columns - convert to pandas string dtype
    string_columns = [
        'broadcast_name', 'abbreviation', 'driver_id', 'first_name', 'last_name', 
        'full_name', 'headshot_url', 'country_code', 'team_name', 'team_color', 
        'team_id', 'status', 'session_name', 'event_name', 'event', 'session'
    ]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype('string')
    
    # 5. Remove redundant columns
    redundant_cols = ['season']  # event_year already exists
    for col in redundant_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 6. Standardize categorical values
    # Status: Keep as-is but ensure consistency
    if 'status' in df.columns:
        # Standardize status values (already appear to be standardized)
        df['status'] = df['status'].str.strip() if df['status'].dtype == 'string' else df['status']
    
    # 7. Create derived columns
    # session_type: Extract type from session_name
    if 'session_name' in df.columns:
        def extract_session_type(session_name: str) -> str:
            if pd.isna(session_name):
                return None
            session_lower = str(session_name).lower()
            # More specific checks first - order matters!
            if 'shootout' in session_lower:
                return 'sprint_shootout'  # SS - 2023 format
            elif 'sprint qualifying' in session_lower:
                return 'sprint_qualifying'  # SQ - 2024+ format
            elif 'practice' in session_lower:
                return 'practice'
            elif 'race' in session_lower or session_lower == 'r':
                return 'race'
            elif 'qualifying' in session_lower or session_lower.startswith('q'):
                return 'qualifying'  # Q - conventional
            elif 'sprint' in session_lower or session_lower == 's':
                return 'sprint'  # S - Sprint race
            return 'unknown'
        
        df['session_type'] = df['session_name'].apply(extract_session_type)
        df['session_type'] = df['session_type'].astype('string')
    
    # Boolean flags for session types
    if 'session_type' in df.columns:
        df['is_practice'] = df['session_type'] == 'practice'
        df['is_qualifying'] = df['session_type'] == 'qualifying'
        df['is_sprint_qualifying'] = df['session_type'] == 'sprint_qualifying'
        df['is_sprint_shootout'] = df['session_type'] == 'sprint_shootout'
        df['is_race'] = df['session_type'] == 'race'
        df['is_sprint'] = df['session_type'].isin(['sprint', 'sprint_qualifying', 'sprint_shootout'])
    
    # Data completeness flags
    if all(col in df.columns for col in ['q1_seconds', 'q2_seconds', 'q3_seconds']):
        df['has_complete_qualifying_data'] = (
            df['q1_seconds'].notna() & 
            df['q2_seconds'].notna() & 
            df['q3_seconds'].notna()
        )
        df['has_qualifying_data'] = (
            df['q1_seconds'].notna() | 
            df['q2_seconds'].notna() | 
            df['q3_seconds'].notna()
        )
    
    if all(col in df.columns for col in ['time_seconds', 'status', 'laps']):
        df['has_complete_race_data'] = (
            df['time_seconds'].notna() & 
            df['status'].notna() & 
            df['laps'].notna()
        )
        df['has_race_data'] = (
            df['time_seconds'].notna() | 
            df['status'].notna() | 
            (df['laps'].notna() & (df['laps'] > 0))
        )
    
    # Position change (race only): grid_starting_position - finishing_position
    if all(col in df.columns for col in ['grid_starting_position', 'finishing_position', 'is_race']):
        # Only calculate for race sessions where both positions exist
        race_mask = df['is_race'] & df['grid_starting_position'].notna() & df['finishing_position'].notna()
        df['position_change'] = pd.NA
        df.loc[race_mask, 'position_change'] = (
            df.loc[race_mask, 'grid_starting_position'] - df.loc[race_mask, 'finishing_position']
        ).astype('Int64')
    
    # 8. Sort columns for better organization
    priority_cols = [
        # Event/Session identifiers
        'event_year', 'event_name', 'event', 'session_name', 'session', 'session_type',
        'is_practice', 'is_qualifying', 'is_race', 'is_sprint',
        # Driver identifiers
        'driver_number', 'driver_id', 'full_name', 'first_name', 'last_name', 
        'broadcast_name', 'abbreviation', 'country_code',
        # Team identifiers
        'team_id', 'team_name', 'team_color',
        # Position data
        'finishing_position', 'classified_position', 'grid_starting_position', 'position_change',
        'is_retired', 'is_classified',
        # Qualifying data
        'q1_seconds', 'q2_seconds', 'q3_seconds',
        'has_qualifying_data', 'has_complete_qualifying_data',
        # Race data
        'time_seconds', 'status', 'laps', 'points',
        'has_race_data', 'has_complete_race_data',
        # Metadata
        'headshot_url'
    ]
    
    # Reorder columns: priority first (only if they exist), then remaining
    existing_priority_cols = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority_cols]
    df = df[existing_priority_cols + remaining_cols]
    
    return df


def create_api_response(
    status_code: int,
    body: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a standardized Lambda response.
    
    Args:
        status_code: HTTP status code (e.g., 200, 400, 500)
        body: Response body as a dictionary (will be JSON serialized)
    
    Returns:
        Lambda response dictionary
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(body)
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for combining bronze layer data into silver layer.
    

    
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
                return create_api_response(
                    400,
                    {"error": "Invalid JSON in request body"}
                )
        
        # Extract headers
        headers = event.get('headers', {})
        
        # Log the incoming request (for debugging)
        print(f"Received {http_method} request to {path}")
        print(f"Headers: {headers}")
        print(f"Body: {body}")
        

        # Extract optional parameters from request body
        #include_future_sessions = body.get('include_future_sessions', False) if body else False
        limit = body.get('limit') if body else None
        
        # Get sessions with laps location
        sessions_with_laps_location = get_sessions_with_s3_locations(
            s3_location_field=S3_LAPS_LOCATION,
            limit=limit
        )

        # Read all laps from S3 locations
        all_laps = read_all_data_from_s3_locations(
            sessions_with_laps_location,
            s3_location_field=S3_LAPS_LOCATION
        )
        
        # Initialize transformed_laps
        transformed_laps = pd.DataFrame()
        
        # Log summary before transformation
        if not all_laps.empty:
            print(f"Successfully loaded {len(all_laps)} total lap records from bronze layer")
           
            # Transform to silver layer
            transformed_laps = transform_laps_to_silver(all_laps)
            transformed_laps.head(500).to_csv("transformed_laps.csv", index=False, na_rep='')
        else:
            print("No lap data loaded")

        # Write transformed laps to silver layer (S3)
        laps_write_result = {}
        if not transformed_laps.empty:
            laps_write_result = write_partitioned_parquet_dataset_to_s3(
                transformed_laps,
                dataset_name="laps",
                partition_cols=["event_year", "event", "session"]
            )
            print(f"Silver laps write result: {laps_write_result}")
        
        # Get sessions with results location
        sessions_with_results_location = get_sessions_with_s3_locations(
            s3_location_field=S3_RESULTS_LOCATION,
            limit=limit
        )

        all_results = read_all_data_from_s3_locations(
            sessions_with_results_location,
            s3_location_field=S3_RESULTS_LOCATION
        )
        
        # Initialize transformed_results
        transformed_results = pd.DataFrame()
        
        if not all_results.empty:
            print(f"Successfully loaded {len(all_results)} total result records from bronze layer")
            # Transform to silver layer
            transformed_results = transform_results_to_silver(all_results)
            transformed_results.head(250).to_csv("transformed_results.csv", index=False, na_rep='')
        else:
            print("No result data loaded")

        # Write transformed results to silver layer (S3)
        results_write_result = {}
        if not transformed_results.empty:
            results_write_result = write_partitioned_parquet_dataset_to_s3(
                transformed_results,
                dataset_name="results",
                partition_cols=["event_year", "event", "session"]
            )
            print(f"Silver results write result: {results_write_result}")

        # Response with summary
        result = {
            "message": "Lambda function executed successfully",
            "sessions_processed": len(sessions_with_laps_location),
            "total_lap_records_bronze": len(all_laps) if not all_laps.empty else 0,
            "total_lap_records_silver": len(transformed_laps) if not transformed_laps.empty else 0,
            "total_results_records_bronze": len(all_results) if not all_results.empty else 0,
            "total_results_records_silver": len(transformed_results) if not transformed_results.empty else 0,
            "s3_bucket": S3_BUCKET_NAME,
            "bronze_path": S3_BRONZE_PATH,
            "silver_path": S3_SILVER_PATH,
            "silver_write": {
                "laps": laps_write_result,
                "results": results_write_result
            },
            "status": "transformed" if not transformed_laps.empty else "no_data"
        }
        
        return create_api_response(200, result)
    
    except Exception as e:
        # Log the error
        print(f"Error processing request: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Return error response
        return create_api_response(
            500,
            {"error": "Internal server error", "message": str(e)}
        )

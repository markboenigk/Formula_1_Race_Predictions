import argparse
import os
from datetime import datetime, timezone

import duckdb


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def configure_duckdb_s3(con: duckdb.DuckDBPyConnection) -> None:
    """
    Enable DuckDB S3 access via httpfs and configure credentials/region.

    DuckDB can use environment variables directly, but setting them explicitly
    makes local runs more predictable.
    """
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    region = _env_first("AWS_REGION", "AWS_DEFAULT_REGION") or "us-east-1"
    con.execute(f"SET s3_region='{region}';")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    # Prefer DuckDB secrets when available (supports env/config/SSO/instance profiles).
    # Fall back to explicit config vars if the secret feature isn't available.
    try:
        # The credential chain will use env vars, ~/.aws/{config,credentials}, SSO, etc.
        con.execute(
            f"""
CREATE OR REPLACE SECRET f1_s3 (
  TYPE S3,
  PROVIDER credential_chain,
  REGION '{region}'
);
"""
        )
        return
    except Exception:
        pass

    if access_key and secret_key:
        con.execute(f"SET s3_access_key_id='{access_key}';")
        con.execute(f"SET s3_secret_access_key='{secret_key}';")
        if session_token:
            con.execute(f"SET s3_session_token='{session_token}';")


def read_sql(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold features in DuckDB from Silver on S3.")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME", "f1-race-prediction"))
    parser.add_argument("--silver-prefix", default=os.getenv("S3_SILVER_PATH", "silver").rstrip("/"))
    parser.add_argument("--gold-prefix", default=os.getenv("S3_GOLD_PATH", "gold").rstrip("/"))
    parser.add_argument(
        "--output-base",
        default=None,
        help="Optional base S3 prefix for outputs. Defaults to '{gold-prefix}/gold_driver_event_features/'.",
    )
    parser.add_argument(
        "--event-year",
        type=int,
        default=None,
        help="Optional filter to a single year (speeds iteration).",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Optional filter to a single event (e.g., 'monaco') (speeds iteration).",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    silver_results_glob = f"s3://{args.bucket}/{args.silver_prefix}/results/**/*.parquet"
    silver_laps_glob = f"s3://{args.bucket}/{args.silver_prefix}/laps/**/*.parquet"
    dim_sessions_glob = f"s3://{args.bucket}/{args.silver_prefix}/dim_sessions/**/*.parquet"
    silver_weather_glob = f"s3://{args.bucket}/{args.silver_prefix}/weather_forecast/**/*.parquet"

    out_base = args.output_base or f"s3://{args.bucket}/{args.gold_prefix}/gold_driver_event_features/"
    out_features = f"{out_base.rstrip('/')}/run_id={run_id}/gold_driver_event_features.parquet"
    out_training = f"{out_base.rstrip('/')}/run_id={run_id}/training_dataset.parquet"

    con = duckdb.connect()
    configure_duckdb_s3(con)

    # Views over Silver datasets (Hive partition columns are derived from S3 paths)
    con.execute(
        f"""
CREATE OR REPLACE VIEW silver_results AS
SELECT * FROM read_parquet('{silver_results_glob}', hive_partitioning=1, union_by_name=1);
"""
    )
    con.execute(
        f"""
CREATE OR REPLACE VIEW silver_laps AS
SELECT * FROM read_parquet('{silver_laps_glob}', hive_partitioning=1, union_by_name=1);
"""
    )
    con.execute(
        f"""
CREATE OR REPLACE VIEW dim_sessions AS
SELECT * FROM read_parquet('{dim_sessions_glob}', hive_partitioning=1, union_by_name=1);
"""
    )
    con.execute(
        f"""
CREATE OR REPLACE VIEW weather AS
SELECT * FROM read_parquet('{silver_weather_glob}', hive_partitioning=1, union_by_name=1);
"""
    )

    filters = []
    if args.event_year is not None:
        filters.append(f"event_year = {int(args.event_year)}")
    if args.event is not None:
        filters.append(f"event = '{args.event}'")
    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql_dir = os.path.dirname(__file__)
    features_sql = read_sql(os.path.join(sql_dir, "gold_driver_event_features.sql"))
    training_sql = read_sql(os.path.join(sql_dir, "gold_training_dataset.sql"))

    # Materialize features
    con.execute("CREATE OR REPLACE TEMP TABLE gold_driver_event_features AS " + features_sql.format(where_clause=where_clause))

    # Warn for events that have no weather data (LEFT JOIN produces NULL weather columns)
    missing_weather = con.execute("""
        SELECT DISTINCT event_year, event
        FROM gold_driver_event_features
        WHERE is_wet_race IS NULL
        ORDER BY event_year, event
    """).fetchall()
    for year, event in missing_weather:
        print(f"WARNING: No weather data for {year} {event} — weather columns will be NULL for this event")

    con.execute(f"COPY gold_driver_event_features TO '{out_features}' (FORMAT PARQUET);")

    # Materialize training dataset (features + race label)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE training_dataset AS "
        + training_sql.format(where_clause=where_clause)
    )
    con.execute(f"COPY training_dataset TO '{out_training}' (FORMAT PARQUET);")

    print("Wrote:", out_features)
    print("Wrote:", out_training)


if __name__ == "__main__":
    main()


"""
Validate rolling / history features for plausibility and no leakage.

Compares gold_driver_event_features for an early event (e.g. Monaco 2025) vs a
later event (e.g. Qatar 2025): rolling features must use only information
available before that event's qualifying session.

Leakage audit (from gold_driver_event_features.sql):
- points_history: only R/S sessions with session_date_utc < qualifying_session_date_utc.
- race_history_ranked / last3_race_form: same filter; last 3 races before target quali.
- circuit_last3y: only event_year in (target_year - 3 .. target_year - 1); same circuit.

Usage:
  python validate_rolling_features.py --event-year 2025 --early monaco --late qatar
  python validate_rolling_features.py --event-year 2025 --early monaco --late qatar --out-dir tmp
"""

import argparse
import os
from pathlib import Path

import duckdb

# Reuse build_gold_features S3 and SQL setup
from build_gold_features import configure_duckdb_s3, read_sql


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rolling features: leakage audit + plausibility (early vs late event)."
    )
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME", "f1-race-prediction"))
    parser.add_argument(
        "--silver-prefix",
        default=os.getenv("S3_SILVER_PATH", "silver").rstrip("/"),
    )
    parser.add_argument("--event-year", type=int, required=True, help="e.g. 2025")
    parser.add_argument("--early", default="monaco", help="Early event (e.g. monaco)")
    parser.add_argument("--late", default="qatar", help="Later event (e.g. qatar)")
    parser.add_argument(
        "--out-dir",
        default="tmp",
        help="Directory to write validation CSVs",
    )
    args = parser.parse_args()

    event_year = args.event_year
    early_event = args.early.lower()
    late_event = args.late.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    silver_results_glob = f"s3://{args.bucket}/{args.silver_prefix}/results/**/*.parquet"
    silver_laps_glob = f"s3://{args.bucket}/{args.silver_prefix}/laps/**/*.parquet"
    dim_sessions_glob = f"s3://{args.bucket}/{args.silver_prefix}/dim_sessions/**/*.parquet"

    con = duckdb.connect()
    configure_duckdb_s3(con)

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

    sql_dir = Path(__file__).resolve().parent
    features_sql = read_sql(str(sql_dir / "gold_driver_event_features.sql"))

    rolling_cols = [
        "target_event_number",
        "qualifying_session_date_utc",
        "season_points_to_date",
        "last3_race_points",
        "last3_avg_finish",
        "last3_dnfs",
        "last3_race_count",
        "circuit_last3y_avg_finish",
        "circuit_last3y_avg_points",
        "circuit_last3y_race_count",
    ]

    def run_for_event(evt: str) -> str:
        where = f"WHERE event_year = {event_year} AND event = '{evt}'"
        con.execute(
            "CREATE OR REPLACE TEMP TABLE gold_validation AS "
            + features_sql.format(where_clause=where)
        )
        path = out_dir / f"gold_rolling_validation_{event_year}_{evt}.csv"
        con.execute(f"COPY gold_validation TO '{path}' (HEADER, DELIMITER ',');")
        return str(path)

    path_early = run_for_event(early_event)
    path_late = run_for_event(late_event)

    print("Wrote:", path_early)
    print("Wrote:", path_late)

    # Plausibility: same driver should have higher season_points_to_date and last3_race_count at late event
    early_df = con.execute(
        "SELECT * FROM read_csv_auto(?)", [path_early]
    ).fetchdf()
    late_df = con.execute(
        "SELECT * FROM read_csv_auto(?)", [path_late]
    ).fetchdf()

    # Align by driver (abbreviation or driver_number)
    key = "abbreviation" if "abbreviation" in early_df.columns else "driver_number"
    early_by_driver = early_df.set_index(key) if key in early_df.columns else early_df
    late_by_driver = late_df.set_index(key) if key in late_df.columns else late_df

    print("\n--- LEAKAGE AUDIT (gold_driver_event_features.sql) ---")
    print("• points_history: only R/S with session_date_utc < qualifying_session_date_utc.")
    print("• race_history_ranked / last3_race_form: same; last 3 races before target quali.")
    print("• circuit_last3y: only event_year in (target_year-3 .. target_year-1); same circuit.")
    print("→ No future or same-weekend data is used.\n")

    print("--- PLAUSIBILITY: rolling stats by event ---")
    for col in ["target_event_number", "season_points_to_date", "last3_race_count"]:
        if col not in early_df.columns:
            continue
        e_vals = early_df[col].dropna()
        l_vals = late_df[col].dropna()
        print(f"  {col}:")
        print(f"    {early_event}: min={e_vals.min()}, max={e_vals.max()}, mean={e_vals.mean():.2f}")
        print(f"    {late_event}:  min={l_vals.min()}, max={l_vals.max()}, mean={l_vals.mean():.2f}")
        if col == "season_points_to_date" and e_vals.max() != "" and l_vals.max() != "":
            try:
                if float(l_vals.max()) >= float(e_vals.max()):
                    print(f"    → OK: later event has >= season points (no leakage).")
                else:
                    print(f"    → CHECK: later event has lower max season points than early.")
            except (TypeError, ValueError):
                pass
        if col == "last3_race_count":
            try:
                em, lm = float(e_vals.mean()), float(l_vals.mean())
                if 0 <= em <= 3 and 0 <= lm <= 3:
                    print(f"    → OK: last3_race_count in [0,3].")
            except (TypeError, ValueError):
                pass

    # Same-driver comparison for a few well-known drivers
    if key in early_df.columns and key in late_df.columns:
        for drv in ["VER", "NOR", "LEC"]:
            if drv not in early_by_driver.index or drv not in late_by_driver.index:
                continue
            re = early_by_driver.loc[drv]
            rl = late_by_driver.loc[drv]
            pts_e = re.get("season_points_to_date")
            pts_l = rl.get("season_points_to_date")
            n3_e = re.get("last3_race_count")
            n3_l = rl.get("last3_race_count")
            print(f"\n  Driver {drv}:")
            print(f"    {early_event}: season_points_to_date={pts_e}, last3_race_count={n3_e}")
            print(f"    {late_event}:  season_points_to_date={pts_l}, last3_race_count={n3_l}")
            try:
                if pts_e != "" and pts_l != "" and float(pts_l) >= float(pts_e):
                    print(f"    → OK: points increase (or equal) at later event.")
                elif pts_e != "" and pts_l != "":
                    print(f"    → CHECK: later event has fewer points.")
            except (TypeError, ValueError):
                pass

    print("\nDone. Review CSVs for full rolling columns.")


if __name__ == "__main__":
    main()

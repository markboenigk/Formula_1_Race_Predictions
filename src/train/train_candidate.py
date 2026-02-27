import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO

import boto3
import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from src.train.onnx_export import export_xgboost_to_onnx
from src.train.config import load_config_by_name, ModelSettingsConfig


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def configure_duckdb_s3(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    region = _env_first("AWS_REGION", "AWS_DEFAULT_REGION") or "us-east-1"
    con.execute(f"SET s3_region='{region}';")
    try:
        con.execute(
            f"""
CREATE OR REPLACE SECRET f1_s3_train (
  TYPE S3,
  PROVIDER credential_chain,
  REGION '{region}'
);
"""
        )
    except Exception:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        if access_key and secret_key:
            con.execute(f"SET s3_access_key_id='{access_key}';")
            con.execute(f"SET s3_secret_access_key='{secret_key}';")
            if session_token:
                con.execute(f"SET s3_session_token='{session_token}';")


def resolve_latest_gold_run_id(bucket: str, gold_prefix: str) -> str:
    s3 = boto3.client("s3")
    prefix = f"{gold_prefix.rstrip('/')}/gold_driver_event_features/"
    paginator = s3.get_paginator("list_objects_v2")
    run_ids: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            marker = "run_id="
            if marker not in key:
                continue
            rest = key.split(marker, 1)[1]
            run_id = rest.split("/", 1)[0]
            if run_id:
                run_ids.add(run_id)
    if not run_ids:
        raise RuntimeError(f"No run_id found under s3://{bucket}/{prefix}")
    return sorted(run_ids)[-1]


def load_training_dataset(bucket: str, gold_prefix: str, run_id: str) -> pd.DataFrame:
    dataset_uri = (
        f"s3://{bucket}/{gold_prefix.rstrip('/')}/gold_driver_event_features/"
        f"run_id={run_id}/training_dataset.parquet"
    )
    con = duckdb.connect()
    configure_duckdb_s3(con)
    return con.execute(f"SELECT * FROM read_parquet('{dataset_uri}', hive_partitioning=1);").df()


def precision_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple[float, float]:
    actual_top = y_true <= k
    predicted_top = y_pred <= k
    tp = int(np.logical_and(actual_top, predicted_top).sum())
    predicted_count = int(predicted_top.sum())
    actual_count = int(actual_top.sum())
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / actual_count if actual_count else 0.0
    return precision, recall


def spearman_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank(method="average").to_numpy()
    b = pd.Series(y_pred).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


@dataclass
class FoldMetric:
    held_out_season: int
    rows: int
    precision_at_3: float
    recall_at_3: float
    precision_at_10: float
    recall_at_10: float
    spearman: float


def run_loso_cv(df: pd.DataFrame, feature_cols: list[str], cat_features: list[str], use_ranker: bool = False, config: ModelSettingsConfig = None) -> list[FoldMetric]:
    seasons = sorted(df["event_year"].dropna().astype(int).unique().tolist())
    if len(seasons) < 2:
        raise RuntimeError("Need at least 2 seasons for leave-one-season-out CV.")

    num_features = [c for c in feature_cols if c not in cat_features]
    
    # Build preprocessor
    transformers = []
    if num_features:
        transformers.append(("num", SimpleImputer(strategy="median"), num_features))
    if cat_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Get hyperparameters from config or use defaults
    hp = config.hyperparameters if config else None

    metrics: list[FoldMetric] = []
    for held_out in seasons:
        train_df = df[df["event_year"].astype(int) != held_out].copy()
        test_df = df[df["event_year"].astype(int) == held_out].copy()
        if train_df.empty or test_df.empty:
            continue

        if use_ranker:
            # XGBRanker needs group IDs (number of samples per event)
            # Sort by event to ensure proper grouping
            train_df = train_df.sort_values(["event_year", "event"]).reset_index(drop=True)
            test_df = test_df.sort_values(["event_year", "event"]).reset_index(drop=True)
            
            # Calculate group sizes (drivers per event)
            train_group_sizes = train_df.groupby(["event_year", "event"]).size().values.tolist()
            test_group_sizes = test_df.groupby(["event_year", "event"]).size().values.tolist()
            
            model = xgb.XGBRanker(
                n_estimators=hp.n_estimators if hp else 200,
                max_depth=hp.max_depth if hp else 3,
                learning_rate=hp.learning_rate if hp else 0.05,
                subsample=hp.subsample if hp else 0.9,
                colsample_bytree=hp.colsample_bytree if hp else 0.9,
                objective="rank:pairwise",
                random_state=hp.random_state if hp else 42,
                n_jobs=hp.n_jobs if hp else 1,
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            # Fit with group parameter for ranker
            pipeline.fit(train_df[feature_cols], train_df["race_finishing_position"], model__group=train_group_sizes)
            preds = pipeline.predict(test_df[feature_cols])
        else:
            model = xgb.XGBRegressor(
                n_estimators=hp.n_estimators if hp else 200,
                max_depth=hp.max_depth if hp else 3,
                learning_rate=hp.learning_rate if hp else 0.05,
                subsample=hp.subsample if hp else 0.9,
                colsample_bytree=hp.colsample_bytree if hp else 0.9,
                objective=hp.objective if hp else "reg:squarederror",
                random_state=hp.random_state if hp else 42,
                n_jobs=hp.n_jobs if hp else 1,
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(train_df[feature_cols], train_df["race_finishing_position"])
            preds = pipeline.predict(test_df[feature_cols])
        
        y_true = test_df["race_finishing_position"].to_numpy(dtype=float)
        
        p3, r3 = precision_recall_at_k(y_true, preds, 3)
        p10, r10 = precision_recall_at_k(y_true, preds, 10)
        spearman = spearman_rank(y_true, preds)
        
        metrics.append(
            FoldMetric(
                held_out_season=int(held_out),
                rows=int(len(test_df)),
                precision_at_3=float(p3),
                recall_at_3=float(r3),
                precision_at_10=float(p10),
                recall_at_10=float(r10),
                spearman=float(spearman),
            )
        )
    if not metrics:
        raise RuntimeError("No valid LOSO folds produced.")
    return metrics


def aggregate_metrics(folds: list[FoldMetric]) -> dict[str, float]:
    return {
        "precision_at_3": float(np.mean([f.precision_at_3 for f in folds])),
        "recall_at_3": float(np.mean([f.recall_at_3 for f in folds])),
        "precision_at_10": float(np.mean([f.precision_at_10 for f in folds])),
        "recall_at_10": float(np.mean([f.recall_at_10 for f in folds])),
        "spearman": float(np.mean([f.spearman for f in folds])),
    }


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def export_predictions_vs_actuals(
    df: pd.DataFrame,
    pipeline: Pipeline,
    s3_client: boto3.client,
    bucket: str,
    model_key_prefix: str,
    run_id: str,
    trained_at_utc: str,
    feature_cols: list[str],
) -> int:
    scored = df.copy()
    scored["predicted_score"] = pipeline.predict(scored[feature_cols])
    scored["predicted_rank"] = (
        scored.groupby(["event_year", "event"])["predicted_score"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    scored["abs_error"] = (scored["predicted_rank"] - scored["race_finishing_position"]).abs()
    scored["run_id"] = run_id
    scored["trained_at_utc"] = trained_at_utc

    base_cols = [
        "event_year",
        "event",
        "driver_number",
        "full_name",
        "qualifying_position",
        "predicted_score",
        "predicted_rank",
        "race_finishing_position",
        "abs_error",
        "run_id",
        "trained_at_utc",
    ]
    export_cols = [col for col in base_cols if col in scored.columns]
    written = 0
    for (event_year, event), grp in scored.groupby(["event_year", "event"], dropna=False):
        out = grp[export_cols].sort_values("predicted_rank").reset_index(drop=True)
        buffer = BytesIO()
        out.to_parquet(buffer, index=False)
        buffer.seek(0)
        key = (
            f"{model_key_prefix}/predictions_vs_actuals/"
            f"event_year={int(event_year)}/event={event}/predictions_vs_actuals.parquet"
        )
        s3_client.upload_fileobj(buffer, bucket, key)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate model with advanced features.")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME", "f1-race-prediction"))
    parser.add_argument("--gold-prefix", default=os.getenv("S3_GOLD_PATH", "gold").rstrip("/"))
    parser.add_argument("--models-prefix", default=os.getenv("S3_MODEL_PATH", "models").rstrip("/"))
    parser.add_argument("--gold-run-id", default=None, help="Optional gold run_id to train from.")
    parser.add_argument("--run-type", default="candidate", choices=["baseline", "candidate"])
    parser.add_argument("--config", default=os.getenv("MODEL_CONFIG", "grid_plus_quali_v1"), help="Model config name (e.g., grid_plus_quali_v1)")
    parser.add_argument("--min-precision-at-3", type=float, default=None, help="Override min precision from config")
    parser.add_argument("--use-ranker", action="store_true", help="Use XGBRanker instead of XGBRegressor (overrides config)")
    args = parser.parse_args()

    # Load model configuration from YAML
    config = load_config_by_name(args.config)
    print(f"Loaded config: {config.model.name} (v{config.model.version})")
    print(f"Model type: {config.model.type}")
    print(f"Features: {config.features.numerical}")
    
    # Use ranker from CLI flag if provided, otherwise from config
    use_ranker = args.use_ranker or config.training.use_ranker

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gold_run_id = args.gold_run_id or resolve_latest_gold_run_id(args.bucket, args.gold_prefix)
    print(f"Using gold run_id={gold_run_id}")
    df = load_training_dataset(args.bucket, args.gold_prefix, gold_run_id)

    required = {"event_year", "race_finishing_position"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise RuntimeError(f"Missing required columns in training dataset: {missing_required}")

    # Define feature set: V0.6 + Qualifying Gap Features
    # v0.6.2: Testing if qualifying gaps help
    
    # Calculate qualifying gap features per event
    # 1. Gap to pole (min q time for the driver - min q time in session)
    # 2. Gap within top 10 (spread of times in top 10)
    
    def add_qualifying_gaps(df):
        df = df.copy()
        
        # Best qualifying time per event (pole position time)
        event_pole_time = df.groupby(["event_year", "event"])["q3_seconds"].min()
        
        # Gap to pole for each driver
        df["gap_to_pole"] = df.apply(
            lambda row: row["q3_seconds"] - event_pole_time.get((row["event_year"], row["event"]), None)
            if pd.notna(row["q3_seconds"]) else None,
            axis=1
        )
        
        # Top 10 gap spread (time between P1 and P10) - competitive indicator
        def calc_top10_spread(group):
            top10 = group.nsmallest(10, "q3_seconds")["q3_seconds"]
            if len(top10) >= 2 and top10.notna().sum() >= 2:
                return top10.max() - top10.min()
            return None
        
        top10_spread = df.groupby(["event_year", "event"]).apply(calc_top10_spread)
        df["top10_gap_spread"] = df.apply(
            lambda row: top10_spread.get((row["event_year"], row["event"]), None),
            axis=1
        )
        
        return df
    
    df = add_qualifying_gaps(df)
    
    # Calculate overtake difficulty features
    # Average position change from grid to finish per track (historical)
    def add_overtake_features(df):
        df = df.copy()
        
        # Calculate avg position change per event (positive = gained positions)
        # Using qualifying_position as proxy for grid position
        df["position_change"] = df["qualifying_position"] - df["race_finishing_position"]
        
        # Avg position gain/loss per track (lower = harder to overtake)
        avg_position_change = df.groupby(["event_year", "event"])["position_change"].mean()
        df["track_overtake_difficulty"] = df.apply(
            lambda row: avg_position_change.get((row["event_year"], row["event"]), None),
            axis=1
        )
        
        # Negative values = drivers typically lose positions (hard to overtake)
        # Positive values = drivers typically gain positions (easy to overtake)
        
        df = df.drop(columns=["position_change"], errors="ignore")
        return df
    
    df = add_overtake_features(df)
    
    # Race pace features (historical)
    race_pace_features = [
        "hist_avg_race_lap_time",
        "hist_race_lap_time_std",
        "hist_race_laps_count",
    ]
    
    feature_cols = [
        "qualifying_position",
        "q1_seconds",
        "q2_seconds", 
        "q3_seconds",
        "gap_to_pole",
        "top10_gap_spread",
        "practice_best_lap_time_seconds",
        "practice_median_lap_time_seconds",
    ]
    # Add race pace features
    race_pace_available = [f for f in race_pace_features if f in df.columns]
    feature_cols = feature_cols + race_pace_available
    cat_features = []
    
    # Filter to only available columns
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = set(feature_cols) - set(available_features)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    feature_cols = available_features
    
    if "event" not in df.columns:
        raise RuntimeError("'event' column required for track-aware model")
    # Ensure event is string
    df["event"] = df["event"].astype(str)

    # Clean data
    df = df.dropna(subset=["race_finishing_position"]).copy()
    df["race_finishing_position"] = pd.to_numeric(df["race_finishing_position"], errors="coerce")
    df = df.dropna(subset=["race_finishing_position"]).copy()
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Run CV
    folds = run_loso_cv(df, feature_cols, cat_features, use_ranker=use_ranker, config=config)
    metrics = aggregate_metrics(folds)
    print("CV Metrics:", json.dumps(metrics))
    
    min_precision = args.min_precision_at_3 if args.min_precision_at_3 is not None else config.cv.min_precision_at_3
    if metrics["precision_at_3"] < min_precision:
        raise RuntimeError(
            f"precision_at_3={metrics['precision_at_3']:.4f} below minimum {args.min_precision_at_3:.4f}"
        )

    # Train final model
    num_features = [c for c in feature_cols if c not in cat_features]
    transformers = []
    if num_features:
        transformers.append(("num", SimpleImputer(strategy="median"), num_features))
    if cat_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    if args.use_ranker:
        # Sort by event for proper group IDs
        df = df.sort_values(["event_year", "event"]).reset_index(drop=True)
        group_sizes = df.groupby(["event_year", "event"]).size().values.tolist()
        
        hp = config.hyperparameters
        model = xgb.XGBRanker(
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            learning_rate=hp.learning_rate,
            subsample=hp.subsample,
            colsample_bytree=hp.colsample_bytree,
            objective="rank:pairwise",
            random_state=hp.random_state,
            n_jobs=hp.n_jobs,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(df[feature_cols], df["race_finishing_position"], model__group=group_sizes)
    else:
        hp = config.hyperparameters
        model = xgb.XGBRegressor(
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            learning_rate=hp.learning_rate,
            subsample=hp.subsample,
            colsample_bytree=hp.colsample_bytree,
            objective=hp.objective,
            random_state=hp.random_state,
            n_jobs=hp.n_jobs,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(df[feature_cols], df["race_finishing_position"])

    # Export
    model_key_prefix = f"{args.models_prefix}/run_id={run_id}"
    s3 = boto3.client("s3")

    model_local_path = "/tmp/xgb_model.ubj"
    pipeline.named_steps["model"].get_booster().save_model(model_local_path)
    s3.upload_file(model_local_path, args.bucket, f"{model_key_prefix}/xgb_model.ubj")

    preprocessor_buffer = BytesIO()
    joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_buffer)
    preprocessor_buffer.seek(0)
    s3.upload_fileobj(preprocessor_buffer, args.bucket, f"{model_key_prefix}/preprocessor.joblib")

    # ONNX export (optional - may fail if onnxmltools not installed)
    try:
        onnx_model_path = "/tmp/model.onnx"
        booster = pipeline.named_steps["model"].get_booster()
        export_xgboost_to_onnx(
            booster=booster,
            feature_name="features",
            output_path=onnx_model_path,
            num_features=len(feature_cols),
        )
        s3.upload_file(onnx_model_path, args.bucket, f"{model_key_prefix}/model.onnx")
    except ImportError:
        print("Warning: onnxmltools not installed, skipping ONNX export")

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    exported_prediction_events = export_predictions_vs_actuals(
        df=df,
        pipeline=pipeline,
        s3_client=s3,
        bucket=args.bucket,
        model_key_prefix=model_key_prefix,
        run_id=run_id,
        trained_at_utc=trained_at_utc,
        feature_cols=feature_cols,
    )

    metadata = {
        "run_id": run_id,
        "run_type": args.run_type,
        "model_config": config.model.name,
        "config_version": config.model.version,
        "dataset_ref": f"gold_driver_event_features/run_id={gold_run_id}",
        "feature_set_name": config.model.name,
        "use_ranker": use_ranker,
        "feature_columns": feature_cols,
        "categorical_features": cat_features,
        "target_definition": "race_finishing_position_regression",
        "cv_strategy": config.cv.strategy,
        "metrics": metrics,
        "fold_metrics": [f.__dict__ for f in folds],
        "train_rows": int(len(df)),
        "season_range": [int(df["event_year"].min()), int(df["event_year"].max())],
        "git_commit": get_git_commit(),
        "trained_at_utc": trained_at_utc,
        "xgboost_version": xgb.__version__,
        "predictions_vs_actuals": {
            "path_prefix": f"s3://{args.bucket}/{model_key_prefix}/predictions_vs_actuals/",
            "events_exported": exported_prediction_events,
            "note": "Predictions are generated on the final fitted model against the training dataset.",
        },
    }

    s3.put_object(
        Bucket=args.bucket,
        Key=f"{model_key_prefix}/model_metadata.json",
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    s3.put_object(Bucket=args.bucket, Key=f"{model_key_prefix}/_SUCCESS", Body=b"")
    s3.put_object(Bucket=args.bucket, Key=f"{args.models_prefix}/_LATEST", Body=run_id.encode("utf-8"))

    print(f"Wrote model artifacts to s3://{args.bucket}/{model_key_prefix}/")
    print(f"Updated latest pointer: s3://{args.bucket}/{args.models_prefix}/_LATEST")
    print("Metrics:", json.dumps(metrics))


if __name__ == "__main__":
    main()

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


def pick_baseline_feature_column(df: pd.DataFrame) -> str:
    for candidate in ("qualifying_position", "grid_starting_position"):
        if candidate in df.columns:
            return candidate
    raise RuntimeError("Neither qualifying_position nor grid_starting_position exists in training dataset.")


def precision_recall_at_top3(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    actual_top3 = y_true <= 3
    predicted_top3 = y_pred <= 3
    tp = int(np.logical_and(actual_top3, predicted_top3).sum())
    predicted_count = int(predicted_top3.sum())
    actual_count = int(actual_top3.sum())
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / actual_count if actual_count else 0.0
    return precision, recall


def spearman_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Rank correlation via pandas ranking keeps this dependency-light.
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
    spearman: float


def run_loso_cv(df: pd.DataFrame, feature_col: str, config: ModelSettingsConfig) -> list[FoldMetric]:
    seasons = sorted(df["event_year"].dropna().astype(int).unique().tolist())
    if len(seasons) < 2:
        raise RuntimeError("Need at least 2 seasons for leave-one-season-out CV.")

    hp = config.hyperparameters
    metrics: list[FoldMetric] = []
    for held_out in seasons:
        train_df = df[df["event_year"].astype(int) != held_out].copy()
        test_df = df[df["event_year"].astype(int) == held_out].copy()
        if train_df.empty or test_df.empty:
            continue

        preprocessor = ColumnTransformer(
            transformers=[("num", SimpleImputer(strategy="median"), [feature_col])],
            remainder="drop",
        )
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
        pipeline.fit(train_df[[feature_col]], train_df["race_finishing_position"])
        preds = pipeline.predict(test_df[[feature_col]])
        y_true = test_df["race_finishing_position"].to_numpy(dtype=float)
        precision, recall = precision_recall_at_top3(y_true, preds)
        spearman = spearman_rank(y_true, preds)
        metrics.append(
            FoldMetric(
                held_out_season=int(held_out),
                rows=int(len(test_df)),
                precision_at_3=float(precision),
                recall_at_3=float(recall),
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
    feature_col: str,
) -> int:
    scored = df.copy()
    scored["predicted_score"] = pipeline.predict(scored[[feature_col]])
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
    parser = argparse.ArgumentParser(description="Train baseline grid-only model and upload artifacts to S3.")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME", "f1-race-prediction"))
    parser.add_argument("--gold-prefix", default=os.getenv("S3_GOLD_PATH", "gold").rstrip("/"))
    parser.add_argument("--models-prefix", default=os.getenv("S3_MODEL_PATH", "models").rstrip("/"))
    parser.add_argument("--gold-run-id", default=None, help="Optional gold run_id to train from.")
    parser.add_argument("--run-type", default=os.getenv("RUN_TYPE", "baseline"), choices=["baseline", "candidate"])
    parser.add_argument("--config", default=os.getenv("MODEL_CONFIG", "grid_only_v1"), help="Model config name (e.g., grid_only_v1)")
    parser.add_argument("--min-precision-at-3", type=float, default=None, help="Override min precision from config")
    args = parser.parse_args()

    # Load model configuration from YAML
    config = load_config_by_name(args.config)
    print(f"Loaded config: {config.model.name} (v{config.model.version})")
    print(f"Model type: {config.model.type}")
    print(f"Features: {config.features.primary_feature}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gold_run_id = args.gold_run_id or resolve_latest_gold_run_id(args.bucket, args.gold_prefix)
    print(f"Using gold run_id={gold_run_id}")
    df = load_training_dataset(args.bucket, args.gold_prefix, gold_run_id)

    required = {"event_year", "race_finishing_position"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise RuntimeError(f"Missing required columns in training dataset: {missing_required}")

    df = df.dropna(subset=["race_finishing_position"]).copy()
    feature_col = pick_baseline_feature_column(df)
    df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")
    df = df.dropna(subset=[feature_col]).copy()
    df["race_finishing_position"] = pd.to_numeric(df["race_finishing_position"], errors="coerce")
    df = df.dropna(subset=["race_finishing_position"]).copy()

    folds = run_loso_cv(df, feature_col, config)
    metrics = aggregate_metrics(folds)
    min_precision = args.min_precision_at_3 if args.min_precision_at_3 is not None else config.cv.min_precision_at_3
    if metrics["precision_at_3"] < min_precision:
        raise RuntimeError(
            f"precision_at_3={metrics['precision_at_3']:.4f} below minimum {args.min_precision_at_3:.4f}"
        )

    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), [feature_col])],
        remainder="drop",
    )
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
    pipeline.fit(df[[feature_col]], df["race_finishing_position"])

    model_key_prefix = f"{args.models_prefix}/run_id={run_id}"
    s3 = boto3.client("s3")

    model_local_path = "/tmp/xgb_model.ubj"
    pipeline.named_steps["model"].get_booster().save_model(model_local_path)
    s3.upload_file(model_local_path, args.bucket, f"{model_key_prefix}/xgb_model.ubj")

    preprocessor_buffer = BytesIO()
    joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_buffer)
    preprocessor_buffer.seek(0)
    s3.upload_fileobj(preprocessor_buffer, args.bucket, f"{model_key_prefix}/preprocessor.joblib")

    # Convert to ONNX for lightweight inference (optional - may fail if onnxmltools not installed)
    try:
        onnx_model_path = "/tmp/model.onnx"
        booster = pipeline.named_steps["model"].get_booster()
        export_xgboost_to_onnx(
            booster=booster,
            feature_name=feature_col,
            output_path=onnx_model_path,
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
        feature_col=feature_col,
    )

    metadata = {
        "run_id": run_id,
        "run_type": args.run_type,
        "model_config": config.model.name,
        "config_version": config.model.version,
        "dataset_ref": f"gold_driver_event_features/run_id={gold_run_id}",
        "feature_set_name": config.model.name,
        "feature_columns": [feature_col],
        "target_definition": "race_finishing_position_regression",
        "cv_strategy": "leave-one-season-out",
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
    print(
        "Wrote predictions_vs_actuals to "
        f"s3://{args.bucket}/{model_key_prefix}/predictions_vs_actuals/"
    )
    print(f"Updated latest pointer: s3://{args.bucket}/{args.models_prefix}/_LATEST")
    print("Metrics:", json.dumps(metrics))


if __name__ == "__main__":
    main()

import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    payload_path = base_dir / "test_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    args = [
        "python",
        str(base_dir / "train.py"),
        "--bucket",
        str(payload.get("bucket", os.getenv("S3_BUCKET_NAME", "f1-race-prediction"))),
        "--gold-prefix",
        str(payload.get("gold_prefix", os.getenv("S3_GOLD_PATH", "gold"))),
        "--models-prefix",
        str(payload.get("models_prefix", os.getenv("S3_MODEL_PATH", "models"))),
        "--run-type",
        str(payload.get("run_type", "baseline")),
        "--baseline-name",
        str(payload.get("baseline_name", "grid_only_v1")),
        "--feature-set-name",
        str(payload.get("feature_set_name", "grid_only")),
        "--min-precision-at-3",
        str(payload.get("min_precision_at_3", 0.0)),
    ]

    if payload.get("gold_run_id"):
        args.extend(["--gold-run-id", str(payload["gold_run_id"])])

    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()

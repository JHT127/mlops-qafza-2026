"""Register the existing Task 2 model in an MLflow tracking server.

This is opt-in because uploading the 578 MiB model intentionally creates a second
artifact copy. The default local inference path uses the read-only artifact mount.
"""

import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--registry-name", default="olist-late-delivery")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    import mlflow
    import mlflow.sklearn

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    model = joblib.load(args.model_dir / "trained_model.joblib")
    metadata = json.loads(Path("config/model_registry.json").read_text(encoding="utf-8"))
    with mlflow.start_run(run_name=metadata["version"]):
        mlflow.log_param("model_version", metadata["version"])
        mlflow.log_metric("average_precision", metadata["training_metric_value"])
        mlflow.sklearn.log_model(model, "model", registered_model_name=args.registry_name)
    print(f"Registered {args.registry_name} version {metadata['version']}")


if __name__ == "__main__":
    main()
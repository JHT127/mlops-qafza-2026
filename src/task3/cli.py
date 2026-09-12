"""Command-line prediction entry point."""

import argparse
import json
import sys

from .artifacts import load_artifacts
from .config import settings
from .pipeline import InferencePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict late delivery for a JSON order.")
    parser.add_argument("input", nargs="?", help="JSON file path; stdin when omitted")
    args = parser.parse_args()
    raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    order = json.loads(raw)
    artifacts = load_artifacts(
        settings.model_dir, settings.feature_list_path, settings.model_version
    )
    prediction = InferencePipeline(artifacts, settings.prediction_threshold).predict(order)
    print(json.dumps(prediction.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

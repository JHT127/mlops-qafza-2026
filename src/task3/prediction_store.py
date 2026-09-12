"""Durable, bounded JSONL storage for later prediction analysis."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class PredictionStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"task3.predictions.{directory}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = RotatingFileHandler(
                directory / "predictions.jsonl",
                maxBytes=25_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def write(self, record: dict[str, Any]) -> None:
        self.logger.info(json.dumps(record, sort_keys=True, default=str))
"""Inference pipeline that reuses Task 2 fitted objects without refitting."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .artifacts import ModelArtifacts
from .features import RAW_FEATURES, add_date_features
from .validation import records_to_frame, validate_orders

logger = logging.getLogger(__name__)


class PredictionError(ValueError):
    """Raised when an order cannot be transformed or predicted."""


@dataclass(frozen=True)
class Prediction:
    is_late: bool
    probability_late: float
    model_version: str
    latency_ms: float


class InferencePipeline:
    def __init__(self, artifacts: ModelArtifacts, threshold: float = 0.5):
        self.artifacts = artifacts
        self.threshold = threshold

    def _prepare(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        result = validate_orders(records)
        if not result.valid:
            raise PredictionError("Input validation failed: " + "; ".join(result.errors))
        frame = add_date_features(records_to_frame(records))
        for feature in RAW_FEATURES:
            if feature not in frame:
                frame[feature] = None
        frame["main_product_category"] = frame["main_product_category"].where(
            frame["main_product_category"].isin(self.artifacts.top_categories), "other"
        )
        return frame[RAW_FEATURES]

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.artifacts.preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return pd.DataFrame(transformed, columns=self.artifacts.feature_list)

    def predict(self, record: dict[str, Any]) -> Prediction:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[dict[str, Any]]) -> list[Prediction]:
        started = time.perf_counter()
        frame = self._prepare(records)
        transformed = self._transform(frame)
        probabilities = self.artifacts.model.predict_proba(transformed)
        classes = list(self.artifacts.model.classes_)
        late_index = classes.index(1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        predictions = [
            Prediction(
                is_late=bool(probability >= self.threshold),
                probability_late=float(probability),
                model_version=self.artifacts.version,
                latency_ms=elapsed_ms / len(records),
            )
            for probability in probabilities[:, late_index]
        ]
        logger.info(
            "prediction request_count=%d output_count=%d probability_mean=%.6f "
            "latency_ms=%.3f model_version=%s",
            len(records),
            len(predictions),
            sum(item.probability_late for item in predictions) / len(predictions),
            elapsed_ms,
            self.artifacts.version,
        )
        return predictions

"""FastAPI adapter for the Task 3 inference pipeline."""

import json
import logging
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .artifacts import ArtifactError, load_artifacts, load_registry_artifacts
from .config import settings
from .logging_config import configure_logging
from .pipeline import InferencePipeline, PredictionError
from .prediction_store import PredictionStore
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    OrderRequest,
    PredictionResponse,
)

configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger(__name__)
prediction_store = PredictionStore(settings.log_dir)
REQUEST_COUNT = Counter("task3_requests_total", "Inference requests", ["route", "status"])
REQUEST_LATENCY = Histogram("task3_request_latency_seconds", "Inference request latency", ["route"])
PREDICTION_COUNT = Counter("task3_predictions_total", "Predictions by outcome", ["is_late"])
PREDICTION_PROBABILITY = Histogram(
    "task3_prediction_probability",
    "Predicted probability of late delivery",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

app = FastAPI(
    title="Qafza Late Delivery Inference API",
    version="0.1.0",
    description="Leakage-aware late-delivery prediction using the Task 2 fitted model.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> InferencePipeline:
    try:
        if settings.model_source == "mlflow":
            model_uri = f"models:/{settings.mlflow_model_name}/{settings.mlflow_model_stage}"
            artifacts = load_registry_artifacts(
                settings.model_dir,
                settings.feature_list_path,
                settings.model_version,
                model_uri,
                settings.mlflow_tracking_uri,
            )
        else:
            artifacts = load_artifacts(
                settings.model_dir, settings.feature_list_path, settings.model_version
            )
    except (ArtifactError, OSError) as exc:
        raise RuntimeError(str(exc)) from exc
    return InferencePipeline(artifacts, settings.prediction_threshold)


def _record_predictions(predictions: list[PredictionResponse]) -> None:
    for prediction in predictions:
        PREDICTION_COUNT.labels(is_late=str(prediction.is_late).lower()).inc()
        PREDICTION_PROBABILITY.observe(prediction.probability_late)


def _store_prediction(input_record: dict, prediction: PredictionResponse, request_id: str) -> None:
    prediction_store.write(
        {
            "request_id": request_id,
            "input": input_record,
            "output": prediction.model_dump(mode="json"),
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_pipeline()
    except RuntimeError as exc:
        return {"status": "degraded", "detail": str(exc)}
    return {"status": "ok"}


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        pipeline = get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelInfoResponse(
        model_version=pipeline.artifacts.version,
        model_type=type(pipeline.artifacts.model).__name__,
        feature_count=len(pipeline.artifacts.feature_list),
        threshold=pipeline.threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderRequest, request: Request) -> PredictionResponse:
    started = time.perf_counter()
    try:
        pipeline = get_pipeline()
        prediction = pipeline.predict(order.model_dump(mode="json"))
    except PredictionError as exc:
        REQUEST_COUNT.labels(route="predict", status="422").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        REQUEST_COUNT.labels(route="predict", status="503").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = PredictionResponse(**prediction.__dict__)
    REQUEST_COUNT.labels(route="predict", status="200").inc()
    REQUEST_LATENCY.labels(route="predict").observe(time.perf_counter() - started)
    _record_predictions([response])
    _store_prediction(
        order.model_dump(mode="json"),
        response,
        request.headers.get("x-request-id", "generated"),
    )
    logger.info(
        "api_prediction path=%s input=%s output=%s",
        request.url.path,
        json.dumps(order.model_dump(mode="json"), sort_keys=True),
        response.model_dump_json(),
    )
    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    if len(payload.orders) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch exceeds configured limit of {settings.max_batch_size} orders",
        )
    started = time.perf_counter()
    try:
        pipeline = get_pipeline()
        predictions = pipeline.predict_batch(
            [order.model_dump(mode="json") for order in payload.orders]
        )
    except PredictionError as exc:
        REQUEST_COUNT.labels(route="predict_batch", status="422").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        REQUEST_COUNT.labels(route="predict_batch", status="503").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response_items = [PredictionResponse(**prediction.__dict__) for prediction in predictions]
    response = BatchPredictionResponse(
        predictions=response_items,
        count=len(response_items),
        model_version=get_pipeline().artifacts.version,
    )
    REQUEST_COUNT.labels(route="predict_batch", status="200").inc()
    REQUEST_LATENCY.labels(route="predict_batch").observe(time.perf_counter() - started)
    _record_predictions(response_items)
    request_id = request.headers.get("x-request-id", "generated")
    for order, item in zip(payload.orders, response_items, strict=True):
        _store_prediction(order.model_dump(mode="json"), item, request_id)
    logger.info(
        "api_batch_prediction path=%s input_count=%d output_count=%d",
        request.url.path,
        len(payload.orders),
        len(response_items),
    )
    return response


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

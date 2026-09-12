"""Pydantic request and response schemas for the inference API."""

from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonNegative = Annotated[float | None, Field(default=None, ge=0)]


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str | None = None
    customer_id: str | None = None
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: date | datetime
    n_items: NonNegative
    n_distinct_products: NonNegative
    n_distinct_sellers: NonNegative
    total_price: NonNegative
    total_freight_value: NonNegative
    avg_freight_value: NonNegative
    total_weight_g: NonNegative
    max_product_length_cm: NonNegative
    max_product_height_cm: NonNegative
    max_product_width_cm: NonNegative
    total_payment_value: NonNegative
    n_payment_transactions: NonNegative
    max_payment_installments: NonNegative
    main_product_category: str | None = None
    main_payment_type: str | None = None
    customer_state: str | None = None
    main_seller_state: str | None = None
    purchase_dayofweek: Annotated[float | None, Field(default=None, ge=0, le=6)] = None
    purchase_month: Annotated[float | None, Field(default=None, ge=1, le=12)] = None


class PredictionResponse(BaseModel):
    is_late: bool
    probability_late: float = Field(ge=0, le=1)
    model_version: str
    latency_ms: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orders: list[OrderRequest] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    feature_count: int
    threshold: float

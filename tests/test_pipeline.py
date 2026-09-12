from datetime import datetime

import pytest

from task3.pipeline import InferencePipeline, PredictionError
from task3.validation import validate_orders


def minimal_order():
    return {
        "order_purchase_timestamp": datetime(2018, 1, 1, 12, 0).isoformat(),
        "order_estimated_delivery_date": "2018-01-20",
        "main_product_category": "health_beauty",
        "main_payment_type": "credit_card",
        "customer_state": "SP",
        "main_seller_state": "SP",
    }


def test_validation_allows_missing_imputable_features():
    result = validate_orders([minimal_order()])
    assert result.valid


def test_validation_rejects_negative_values():
    order = minimal_order()
    order["total_price"] = -1
    result = validate_orders([order])
    assert not result.valid
    assert "total_price must be >= 0" in result.errors[0]


def test_pipeline_predicts_known_shape(loaded_pipeline):
    prediction = loaded_pipeline.predict(minimal_order())
    assert isinstance(prediction.is_late, bool)
    assert 0 <= prediction.probability_late <= 1
    assert prediction.model_version


def test_pipeline_rejects_bad_dates(loaded_pipeline):
    order = minimal_order()
    order["order_purchase_timestamp"] = "not-a-date"
    with pytest.raises(PredictionError, match="ISO date"):
        loaded_pipeline.predict(order)
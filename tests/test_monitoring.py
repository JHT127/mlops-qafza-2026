import json

from task3.drift import population_stability_index
from task3.prediction_store import PredictionStore
from task3.validation import validate_orders


def valid_order():
    return {
        "order_purchase_timestamp": "2018-01-01T12:00:00",
        "order_estimated_delivery_date": "2018-01-20",
        "total_price": 10,
    }


def test_great_expectations_rejects_negative_numeric_input():
    order = valid_order()
    order["total_price"] = -1
    result = validate_orders([order])
    assert not result.valid
    assert any("Great Expectations" in error for error in result.errors)


def test_population_stability_index_detects_distribution_shift():
    baseline = [0.05] * 50 + [0.95] * 50
    current = [0.45] * 50 + [0.55] * 50
    assert population_stability_index(baseline, current) > 0.1


def test_prediction_store_writes_jsonl(tmp_path):
    store = PredictionStore(tmp_path)
    store.write({"request_id": "abc", "output": {"probability_late": 0.2}})
    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["request_id"] == "abc"

from fastapi.testclient import TestClient

from task3.api import app


def test_health_and_model_routes():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    model = client.get("/model")
    assert model.status_code == 200
    assert model.json()["feature_count"] == 91


def test_predict_route_rejects_unknown_fields():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "order_purchase_timestamp": "2018-01-01T12:00:00",
            "order_estimated_delivery_date": "2018-01-20",
            "unexpected": "not allowed",
        },
    )
    assert response.status_code == 422
from fastapi.testclient import TestClient

from task3.api import app


def test_health_and_model_routes(monkeypatch, loaded_pipeline):
    monkeypatch.setattr("task3.api.get_pipeline", lambda: loaded_pipeline)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    model = client.get("/model")
    assert model.status_code == 200
    assert model.json()["feature_count"] == len(loaded_pipeline.artifacts.feature_list)


def test_predict_route_rejects_unknown_fields(monkeypatch, loaded_pipeline):
    monkeypatch.setattr("task3.api.get_pipeline", lambda: loaded_pipeline)
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
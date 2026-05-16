import sys
import os

os.environ["PROJECT_ROOT"] = "/Users/azeemkhalipha/mlops-demand-forecast"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/serving"))

from fastapi.testclient import TestClient
from app import app
import pytest

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_predict_endpoint(client):
    payload = {
        "qty_lag_1": 10.0, "qty_lag_7": 8.0,
        "qty_lag_30": 9.0, "qty_rolling_avg_7": 9.5,
        "qty_rolling_avg_30": 9.0, "qty_rolling_std_7": 1.2,
        "daily_revenue": 45.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_quantity" in data
    assert data["status"] == "success"
    assert isinstance(data["predicted_quantity"], float)

def test_batch_predict_endpoint(client):
    payload = {
        "inputs": [
            {
                "qty_lag_1": 10.0, "qty_lag_7": 8.0,
                "qty_lag_30": 9.0, "qty_rolling_avg_7": 9.5,
                "qty_rolling_avg_30": 9.0, "qty_rolling_std_7": 1.2,
                "daily_revenue": 45.0
            },
            {
                "qty_lag_1": 5.0, "qty_lag_7": 4.0,
                "qty_lag_30": 6.0, "qty_rolling_avg_7": 5.0,
                "qty_rolling_avg_30": 5.5, "qty_rolling_std_7": 0.8,
                "daily_revenue": 20.0
            }
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2

def test_invalid_input(client):
    payload = {"qty_lag_1": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append("src/serving")
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    payload = {
        "qty_lag_1":          12.0,
        "qty_lag_7":          10.5,
        "qty_lag_30":         9.0,
        "qty_rolling_avg_7":  11.0,
        "qty_rolling_avg_30": 10.0,
        "qty_rolling_std_7":  2.5,
        "daily_revenue":      45.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_demand" in response.json()
    assert response.json()["predicted_demand"] >= 0

def test_predict_missing_field():
    payload = {"qty_lag_1": 12.0}  # Missing fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_batch_predict():
    payload = {
        "records": [
            {
                "qty_lag_1": 12.0, "qty_lag_7": 10.5,
                "qty_lag_30": 9.0, "qty_rolling_avg_7": 11.0,
                "qty_rolling_avg_30": 10.0, "qty_rolling_std_7": 2.5,
                "daily_revenue": 45.0
            },
            {
                "qty_lag_1": 5.0, "qty_lag_7": 4.5,
                "qty_lag_30": 4.0, "qty_rolling_avg_7": 4.8,
                "qty_rolling_avg_30": 4.2, "qty_rolling_std_7": 0.8,
                "daily_revenue": 20.0
            }
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    assert response.json()["record_count"] == 2
    assert len(response.json()["predictions"]) == 2
"""
Unit & integration tests for the Churn Prediction API.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 24, "PhoneService": 1, "PaperlessBilling": 0,
    "MonthlyCharges": 55.0, "TotalCharges": 1320.0
}

# ── System endpoints ──────────────────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_health_has_uptime():
    response = client.get("/health")
    data = response.json()
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "uptime_seconds" in data

# ── Prediction endpoints ──────────────────────────────────────────────────────
def test_predict_valid_input():
    """Model may not be loaded in CI — just check endpoint exists."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code in [200, 503]   # 503 = model not loaded yet

def test_predict_missing_field():
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422           # Unprocessable Entity

def test_predict_invalid_gender():
    bad = {**VALID_CUSTOMER, "gender": 5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422

def test_predict_negative_tenure():
    bad = {**VALID_CUSTOMER, "tenure": -1}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422

def test_batch_predict_valid():
    batch = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code in [200, 503]

def test_batch_predict_empty():
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code in [200, 422]

# ── Response schema ───────────────────────────────────────────────────────────
def test_predict_response_schema():
    response = client.post("/predict", json=VALID_CUSTOMER)
    if response.status_code == 200:
        data = response.json()
        assert "churn_prediction"  in data
        assert "churn_probability" in data
        assert "risk_level"        in data
        assert "recommendation"    in data
        assert data["churn_prediction"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

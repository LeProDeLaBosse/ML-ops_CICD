"""Tests unitaires de l'API (fournis, ils doivent passer tels quels)."""
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_REQUEST = {
    "age": 42,
    "income": 38000,
    "loan_amount": 12000,
    "employment_years": 6.5,
    "num_credit_lines": 3,
    "region": "North",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_and_version():
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision"] in {"accept", "review"}
    assert body["model_version"]


def test_predict_rejects_invalid_input():
    invalid = {**VALID_REQUEST, "age": 12, "region": "Mars"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

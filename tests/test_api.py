"""
tests/test_api.py
Tests für die FastAPI-App (app/main.py) über den TestClient von Starlette.
Läuft ohne echten Server/Docker - main.py lädt Modell + Scaler direkt beim Import.
"""
import pytest
from fastapi.testclient import TestClient
import sys 
import os  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 60,
    "sex": "male",
    "weight": 64,
    "bmi": 24.3,
    "hereditary_diseases": "NoDisease",
    "no_of_dependents": 1,
    "smoker": 0,
    "city": "NewYork",
    "bloodpressure": 72,
    "diabetes": 0,
    "regular_ex": 0,
    "job_title": "Actor",
}


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "features" in body
    assert isinstance(body["features"], list)


def test_predict_valid_input_returns_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_claim_usd" in body
    assert isinstance(body["predicted_claim_usd"], (int, float))
    assert body["predicted_claim_usd"] >= 0
    assert body["input"]["city"] == "NewYork"


def test_predict_missing_required_field_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload["age"]  # Pflichtfeld fehlt

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field,bad_value", [
    ("sex", "unknown"),
    ("city", "Atlantis"),
    ("hereditary_diseases", "Flu"),
    ("job_title", "Wizard"),
])
def test_predict_invalid_categorical_value_returns_422(field, bad_value):
    payload = VALID_PAYLOAD.copy()
    payload[field] = bad_value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert bad_value in response.json()["detail"]


def test_predict_wrong_type_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["age"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("job_title", ["Doctor", "Student", "CEO", "Farmer"])
def test_predict_accepts_all_known_job_titles(job_title):
    """Jeder Wert, der im label_encoders.json steht, muss von der API akzeptiert werden."""
    payload = {**VALID_PAYLOAD, "job_title": job_title}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

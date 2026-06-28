"""
app/main.py
FastAPI REST-API für den Health Insurance Cost Predictor.

Spalten: age, sex, weight, bmi, hereditary_diseases, no_of_dependents,
         smoker, city, bloodpressure, diabetes, regular_ex, job_title
Target:  claim (USD)
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neural_network import InsuranceRegressor

app = FastAPI(
    title="Health Insurance Cost Predictor",
    description="Schätzt Versicherungskosten (claim) in USD via PyTorch Neural Network.",
    version="1.0.0",
)

# ── Metadaten & Modell laden ──────────────────────────────────────────────────
with open("models/scaler.json") as f:
    scaler = json.load(f)

with open("models/label_encoders.json") as f:
    label_encoders = json.load(f)

FEATURE_COLS    = scaler["_feature_cols"]
TARGET_MAX      = scaler["_target_max"]
NUMERICAL_COLS  = scaler["_numerical"]
CATEGORICAL_COLS = scaler["_categorical"]
INPUT_DIM       = len(FEATURE_COLS)

model = InsuranceRegressor(INPUT_DIM)
model.load_state_dict(torch.load("models/insurance_model.pth", map_location="cpu"))
model.eval()


# ── Input-Schema ──────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    age:                 float = Field(..., example=35)
    sex:                 str   = Field(..., example="male",        description="'male' oder 'female'")
    weight:              float = Field(..., example=70.0,          description="Gewicht in kg")
    bmi:                 float = Field(..., example=24.3)
    hereditary_diseases: str   = Field(..., example="NoDisease",   description="z.B. 'NoDisease', 'Epilepsy', ...")
    no_of_dependents:    int   = Field(0,   example=1)
    smoker:              int   = Field(..., example=0,             description="0 = Nichtraucher, 1 = Raucher")
    city:                str   = Field(..., example="NewYork")
    bloodpressure:       float = Field(..., example=72.0)
    diabetes:            int   = Field(..., example=0,             description="0 = Nein, 1 = Ja")
    regular_ex:          int   = Field(..., example=0,             description="0 = kein Sport, 1 = regelmäßig")
    job_title:           str   = Field(..., example="Engineer")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(data: PatientData) -> torch.Tensor:
    row = {}

    # Kategorische Features encoden
    for col in CATEGORICAL_COLS:
        mapping = label_encoders[col]
        val = str(getattr(data, col)).strip()
        if val not in mapping:
            raise HTTPException(
                status_code=422,
                detail=f"Ungültiger Wert '{val}' für '{col}'. Erlaubt: {list(mapping.keys())}"
            )
        row[col] = mapping[val]

    # Numerische Features standardisieren
    for col in NUMERICAL_COLS:
        raw = float(getattr(data, col))
        row[col] = (raw - scaler[col]["mean"]) / scaler[col]["std"]

    # Binäre Features direkt übernehmen
    for col in FEATURE_COLS:
        if col not in row:
            row[col] = float(getattr(data, col))

    features = np.array([[row[col] for col in FEATURE_COLS]], dtype=np.float32)
    return torch.tensor(features)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "insurance_regressor_v1",
        "features": FEATURE_COLS,
    }


@app.post("/predict")
def predict(patient: PatientData):
    """Gibt die geschätzten jährlichen Versicherungskosten (claim) in USD zurück."""
    features = preprocess(patient)

    with torch.no_grad():
        pred_norm = model(features).item()

    claim = round(pred_norm * TARGET_MAX, 2)

    return {
        "predicted_claim_usd": claim,
        "input": patient.model_dump(),
    }

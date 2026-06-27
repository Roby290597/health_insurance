"""
app/main.py
FastAPI REST-API für den Health Insurance Cost Predictor.

Endpoints:
  GET  /           → Health-Check
  POST /predict    → Vorhersage der normierten Versicherungskosten
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# neural_network.py muss im PYTHONPATH liegen (wird über Dockerfile sichergestellt)
from neural_network import InsuranceRegressor

# ── Metadaten ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Health Insurance Cost Predictor",
    description="Vorhersage von Krankenversicherungskosten mittels eines PyTorch Neural Networks.",
    version="1.0.0",
)

# ── Modell & Scaler laden ─────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "insurance_model.pth")
SCALER_PATH  = os.path.join("models", "scaler.json")
ENCODER_PATH = os.path.join("models", "label_encoders.json")

with open(SCALER_PATH) as f:
    scaler = json.load(f)

with open(ENCODER_PATH) as f:
    label_encoders = json.load(f)

FEATURE_COLS = scaler["_feature_cols"]
TARGET_MAX   = scaler["_target_max"]
INPUT_DIM    = len(FEATURE_COLS)

model = InsuranceRegressor(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# ── Input-Schema ──────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    age:          float = Field(..., example=35, description="Alter in Jahren")
    sex:          str   = Field(..., example="male", description="'male' oder 'female'")
    bmi:          float = Field(..., example=27.5, description="Body-Mass-Index")
    children:     Optional[int] = Field(0, example=1, description="Anzahl der Kinder")
    smoker:       str   = Field(..., example="no", description="'yes' oder 'no'")
    region:       str   = Field(..., example="northwest", description="'northeast','northwest','southeast','southwest'")
    # Falls dein Datensatz bloodpressure enthält:
    bloodpressure: Optional[float] = Field(None, example=80.0, description="Blutdruck (optional)")


# ── Hilfsfunktion: Preprocessing ─────────────────────────────────────────────
def preprocess(data: PatientData) -> torch.Tensor:
    row = {}

    # Kategorische Features encoden
    for col, mapping in label_encoders.items():
        val = getattr(data, col, None)
        if val is None:
            raise HTTPException(status_code=422, detail=f"Fehlendes Feld: {col}")
        val_lower = str(val).lower()
        if val_lower not in mapping:
            raise HTTPException(
                status_code=422,
                detail=f"Ungültiger Wert '{val}' für '{col}'. Erlaubt: {list(mapping.keys())}"
            )
        row[col] = mapping[val_lower]

    # Numerische Features standardisieren
    for col, params in scaler.items():
        if col.startswith("_"):
            continue
        raw_val = getattr(data, col, None)
        if raw_val is None:
            raw_val = 0.0
        row[col] = (float(raw_val) - params["mean"]) / params["std"]

    # Nicht-normalisierte numerische Felder direkt übernehmen (z.B. children)
    for col in FEATURE_COLS:
        if col not in row:
            row[col] = float(getattr(data, col, 0) or 0)

    # Feature-Reihenfolge sicherstellen
    features = np.array([[row[col] for col in FEATURE_COLS]], dtype=np.float32)
    return torch.tensor(features)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "healthy", "model": "insurance_regressor_v1", "features": FEATURE_COLS}


@app.post("/predict")
def predict(patient: PatientData):
    """
    Gibt die vorhergesagten Versicherungskosten zurück (in der Originalskala).
    """
    features = preprocess(patient)

    with torch.no_grad():
        prediction_norm = model(features).item()

    # Zurückskalieren
    prediction = round(prediction_norm * TARGET_MAX, 2)

    return {
        "predicted_claim": prediction,
        "currency": "USD",
        "note": "Schätzung basierend auf Trainingsdaten"
    }

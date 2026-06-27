"""
train_model.py
Spalten: age, sex, weight, bmi, hereditary_diseases, no_of_dependents,
         smoker, city, bloodpressure, diabetes, regular_ex, job_title, claim
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


from neural_network import InsuranceRegressor

np.random.seed(123)
torch.manual_seed(123)

# ── Daten laden ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/insurance_data_raw.csv", sep=",")
df = df.dropna()
print(f"Geladen: {df.shape[0]} Zeilen | Spalten: {df.columns.tolist()}")

TARGET_COL = "claim"

# ── Kategorische Features encoden ─────────────────────────────────────────────
# String-Spalten automatisch erkennen
CATEGORICAL_COLS = df.select_dtypes(include="object").columns.tolist()
if TARGET_COL in CATEGORICAL_COLS:
    CATEGORICAL_COLS.remove(TARGET_COL)
print(f"Kategorisch: {CATEGORICAL_COLS}")

encoding_maps = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoding_maps[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    print(f"  {col}: {encoding_maps[col]}")

# ── Numerische Features standardisieren ───────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
# Nur kontinuierliche numerische Spalten skalieren (nicht binäre 0/1-Spalten)
BINARY_COLS = ["smoker", "diabetes", "regular_ex"]
NUMERICAL_COLS = [
    c for c in FEATURE_COLS
    if c not in CATEGORICAL_COLS and c not in BINARY_COLS
]
print(f"Numerisch (skaliert): {NUMERICAL_COLS}")

scaler_params = {}
for col in NUMERICAL_COLS:
    mean = float(df[col].mean())
    std  = float(df[col].std())
    if std == 0:
        std = 1.0
    df[col] = (df[col] - mean) / std
    scaler_params[col] = {"mean": mean, "std": std}

# ── Target normalisieren ───────────────────────────────────────────────────────
target_max = float(df[TARGET_COL].max())
df[TARGET_COL] = df[TARGET_COL] / target_max

# ── Train/Test Split ──────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

# ── Modell trainieren ─────────────────────────────────────────────────────────
input_dim = X_train.shape[1]
model     = InsuranceRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nTraining | Input-Dim: {input_dim} | Features: {FEATURE_COLS}")
EPOCHS = 300
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train: {loss.item():.5f} | Val: {val_loss:.5f}")

# ── Speichern ─────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/insurance_model.pth")
print("\n✓ models/insurance_model.pth")

scaler_params["_target_max"]   = target_max
scaler_params["_feature_cols"] = FEATURE_COLS
scaler_params["_numerical"]    = NUMERICAL_COLS
scaler_params["_categorical"]  = CATEGORICAL_COLS
with open("models/scaler.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print("✓ models/scaler.json")

with open("models/label_encoders.json", "w") as f:
    json.dump(encoding_maps, f, indent=2, ensure_ascii=False)
print("✓ models/label_encoders.json")

"""
train_model.py
Trainiert InsuranceRegressor und speichert:
  - models/insurance_model.pth   (Modellgewichte)
  - models/scaler.json           (Mittelwerte & Std für Normalisierung)
  - models/label_encoders.json   (Encoding-Maps für kategorische Features)
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from neural_network import InsuranceRegressor

# ── Reproduzierbarkeit ────────────────────────────────────────────────────────
np.random.seed(123)
torch.manual_seed(123)

# ── Daten laden ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/insurance_data_mod.csv", sep=",")
df = df.dropna()

# ── Kategorische Features encoden ─────────────────────────────────────────────
CATEGORICAL_COLS = ["sex", "smoker", "region"]   # ggf. anpassen
label_encoders = {}
encoding_maps = {}

for col in CATEGORICAL_COLS:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        encoding_maps[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

# ── Numerische Features standardisieren ───────────────────────────────────────
NUMERICAL_COLS = ["age", "bmi", "bloodpressure"] if "bloodpressure" in df.columns \
    else ["age", "bmi"]

scaler_params = {}
for col in NUMERICAL_COLS:
    if col in df.columns:
        mean = float(df[col].mean())
        std  = float(df[col].std())
        df[col] = (df[col] - mean) / std
        scaler_params[col] = {"mean": mean, "std": std}

# ── Target normalisieren (wie in deinem Original-Code) ────────────────────────
TARGET_COL = "claim" if "claim" in df.columns  else "charges"
target_max = float(df[TARGET_COL].max())
df[TARGET_COL] = df[TARGET_COL].astype(float) / target_max

# ── Features & Target ─────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
print(f"Features: {FEATURE_COLS}")
X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

# ── Modell ────────────────────────────────────────────────────────────────────
input_dim = X_train.shape[1]
model     = InsuranceRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss  = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

# ── Speichern ─────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/insurance_model.pth")
print("✓ Modell gespeichert: models/insurance_model.pth")

scaler_params["_target_max"] = target_max
scaler_params["_feature_cols"] = FEATURE_COLS
with open("models/scaler.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print("✓ Scaler gespeichert: models/scaler.json")

with open("models/label_encoders.json", "w") as f:
    json.dump(encoding_maps, f, indent=2, ensure_ascii=False)
print("✓ Label-Encodings gespeichert: models/label_encoders.json")

print(f"\nFeatures ({input_dim}): {FEATURE_COLS}")

# Deployment-Anleitung: FastAPI + Docker
### Health Insurance Cost Predictor

---

## Voraussetzungen

- `train_model.py` wurde erfolgreich ausgeführt → `models/` Ordner existiert
- API läuft lokal (`http://localhost:8000` erreichbar)
- Docker Desktop installiert und gestartet

---

## Schritt 1: Docker Desktop einrichten

1. Docker Desktop downloaden: https://www.docker.com/products/docker-desktop/
2. Installieren und starten
3. Settings → General → **"Use the WSL 2 based engine"** aktivieren
4. Settings → Resources → WSL Integration → **Ubuntu aktivieren**
5. Docker Desktop neu starten

Testen ob Docker läuft:
```bash
docker --version
# Erwartete Ausgabe: Docker version 26.x.x, build ...
```

---

## Schritt 2: Projektstruktur prüfen

Vor dem Build sicherstellen dass folgende Dateien vorhanden sind:

```
health_insurance/
├── app/
│   ├── __init__.py
│   └── main.py
├── models/
│   ├── insurance_model.pth
│   ├── scaler.json
│   └── label_encoders.json
├── neural_network.py
├── train_model.py
├── requirements.txt
└── Dockerfile
```

---

## Schritt 3: Docker Image bauen

Im Root-Verzeichnis des Projekts (wo das Dockerfile liegt):

```bash
docker build -t health-insurance-api .
```

Was hier passiert:
- Docker liest das `Dockerfile`
- Installiert Python 3.11 in einem isolierten Container
- Kopiert den Code hinein
- Installiert alle Packages aus `requirements.txt`
- Baut ein fertiges Image namens `health-insurance-api`

Build erfolgreich wenn am Ende steht:
```
Successfully built <image-id>
Successfully tagged health-insurance-api:latest
```

---

## Schritt 4: Container starten

```bash
docker run -p 8000:8000 health-insurance-api
```

Flags erklärt:
- `-p 8000:8000` → Port 8000 des Containers wird auf Port 8000 des Hosts gemappt
- `health-insurance-api` → Name des Images das gestartet wird

Im Hintergrund laufen lassen (optional):
```bash
docker run -d -p 8000:8000 --name insurance-api health-insurance-api
```

---

## Schritt 5: API testen

**Health-Check im Browser:**
```
http://localhost:8000
```

**Swagger UI (interaktive Doku):**
```
http://localhost:8000/docs
```

**Predict-Endpoint via curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "job_title": "Actor"
  }'
```

Erwartete Antwort:
```json
{
  "predicted_claim_usd": 13112.60,
  "input": { ... }
}
```

---

## Nützliche Docker-Befehle

```bash
# Laufende Container anzeigen
docker ps

# Container stoppen
docker stop insurance-api

# Container löschen
docker rm insurance-api

# Alle Images anzeigen
docker images

# Image löschen
docker rmi health-insurance-api

# Logs eines laufenden Containers anzeigen
docker logs insurance-api

# In einen laufenden Container einsteigen
docker exec -it insurance-api bash
```

---

## REST API Endpunkte

| Method | Endpoint   | Beschreibung                        |
|--------|------------|-------------------------------------|
| GET    | `/`        | Health-Check, zeigt Feature-Liste   |
| GET    | `/docs`    | Swagger UI (interaktive Doku)       |
| GET    | `/redoc`   | ReDoc (alternative Doku)            |
| POST   | `/predict` | Vorhersage der Versicherungskosten  |

### POST /predict — Input-Schema

| Feld                  | Typ   | Beispiel      | Beschreibung              |
|-----------------------|-------|---------------|---------------------------|
| `age`                 | float | 35            | Alter in Jahren           |
| `sex`                 | str   | "male"        | "male" oder "female"      |
| `weight`              | float | 70.0          | Gewicht in kg             |
| `bmi`                 | float | 24.3          | Body-Mass-Index           |
| `hereditary_diseases` | str   | "NoDisease"   | z.B. "Epilepsy"           |
| `no_of_dependents`    | int   | 1             | Anzahl Abhängige          |
| `smoker`              | int   | 0             | 0 = Nein, 1 = Ja          |
| `city`                | str   | "NewYork"     | Stadt                     |
| `bloodpressure`       | float | 72.0          | Blutdruck                 |
| `diabetes`            | int   | 0             | 0 = Nein, 1 = Ja          |
| `regular_ex`          | int   | 0             | 0 = kein Sport, 1 = Ja   |
| `job_title`           | str   | "Engineer"    | Berufsbezeichnung         |

---

## Workflow Zusammenfassung

```
Daten → train_model.py → models/ → FastAPI (main.py) → Docker Image → Container → REST API
```

1. `python train_model.py` → Modell trainieren & speichern
2. `uvicorn app.main:app --reload` → lokal testen
3. `docker build -t health-insurance-api .` → Image bauen
4. `docker run -p 8000:8000 health-insurance-api` → deployen



## Nächste Schritte — Health Insurance API

1. **Docker** — Image bauen & Container starten (`docker build`, `docker run`)
2. **Docker Compose** — `docker-compose.yml` anlegen, mit `docker compose up` starten
3. **Error Handling & Logging** — saubere Fehlermeldungen, jeden Request loggen
4. **API Key Authentifizierung** — Endpoint mit `X-API-Key` absichern, Key aus Umgebungsvariable lesen
5. **Tests & README** — `pytest` Grundtests schreiben, Repo dokumentieren und aufräumen

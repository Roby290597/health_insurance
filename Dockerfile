# Dockerfile
# Health Insurance Cost Predictor – FastAPI + PyTorch

FROM python:3.11-slim

WORKDIR /app

# System-Dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python-Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY app/ ./app/
COPY models/ ./models/
COPY neural_network.py .

# Port freigeben
EXPOSE 8000

# Starten
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Health Insurance Cost Predictor

Predicts the yearly insurance claim (USD) of a person from demographic and health data. The project covers the full path from data exploration and model comparison to a PyTorch model served as a REST API with FastAPI and Docker.


## Data

The data comes from Kaggle: [Insurance Claim Analysis: Demographic and Health](https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health).
It contains 15,000 records with 12 features (age, sex, weight, BMI, hereditary diseases, number of dependents, smoker, city, blood pressure, diabetes, regular exercise, job title) and the target `claim`.

## Model Comparison

The experiments are in `experiments/`. Each model was evaluated on a held-out test set of 3,000 records (60/20/20 split) with the target normalized to [0, 1].

| Model                     | R²    | Pearson r | Spearman r | RMSE  |
|---------------------------|-------|-----------|------------|-------|
| XGBoost                   | 0.959 | 0.979     | 0.977      | 0.038 |
| Neural network ReLU       | 0.742 | 0.862     | 0.831      | 0.098 |
| Neural network GELU       | 0.742 | 0.862     | 0.831      | 0.098 |
| Linear regression         | n/a   | 0.863     | 0.845      | n/a   |

XGBoost clearly performs best. The small neural network performs about as well as linear regression, which suggests it mostly learns linear relationships in this setup. The API currently serves the neural network, since the goal of the project was the deployment workflow with PyTorch. The model explanations with SHAP are in `experiments/insurance_analysis.ipynb`.

## Repository Structure

```
health_insurance/
├── app/main.py            # FastAPI app (/ and /predict)
├── models/                # trained model, scaler and label encoders
├── experiments/           # notebooks: data analysis, benchmarking, SHAP
├── tests/                 # pytest tests for API and model
├── neural_network.py      # model definition
├── train_model.py         # training script, writes to models/
├── Dockerfile
└── requirements.txt
```

Run the tests with `pytest` from the project root.

# Deployment: FastAPI + Docker

## Requirements

- `train_model.py` has been run successfully → `models/` folder exists
- API runs locally (`http://localhost:8000`)
- Docker Desktop is installed and running

---

## Step 1: Set up Docker Desktop

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Install and start it
3. Settings → General → enable **"Use the WSL 2 based engine"**
4. Settings → Resources → WSL Integration → **enable Ubuntu**
5. Restart Docker Desktop

Check that Docker is running:
```bash
docker --version
# Expected output: Docker version 26.x.x, build ...
```

---

## Step 2: Check the project structure

Before building, make sure the following files exist:

```
health_insurance/
├── app/
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

## Step 3: Build the Docker image

In the project root (where the Dockerfile is located):

```bash
docker build -t health-insurance-api .
```

What happens here:
- Docker reads the `Dockerfile`
- Installs Python 3.11 in an isolated container
- Copies the code into it
- Installs all packages from `requirements.txt`
- Builds a finished image called `health-insurance-api`

The build was successful if it ends with:
```
Successfully built <image-id>
Successfully tagged health-insurance-api:latest
```

---

## Step 4: Start the container

```bash
docker run -p 8000:8000 health-insurance-api
```

Flags explained:
- `-p 8000:8000` → maps port 8000 of the container to port 8000 of the host
- `health-insurance-api` → name of the image to start

Run in the background (optional):
```bash
docker run -d -p 8000:8000 --name insurance-api health-insurance-api
```

---

## Step 5: Test the API

**Health check in the browser:**
```
http://localhost:8000
```

**Swagger UI (interactive docs):**
```
http://localhost:8000/docs
```

**Predict endpoint via curl:**
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

Expected response:
```json
{
  "predicted_claim_usd": 13112.60,
  "input": { ... }
}
```



## Workflow Summary

```
Data → train_model.py → models/ → FastAPI (main.py) → Docker image → Container → REST API
```

1. `python train_model.py` → train & save the model
2. `uvicorn app.main:app --reload` → test locally
3. `docker build -t health-insurance-api .` → build the image
4. `docker run -p 8000:8000 health-insurance-api` → deploy



## Possible Extensions

The project is finished in its current form. Possible next steps would be:

1. **Serve XGBoost**: deploy the best performing model instead of the neural network
2. **Docker Compose**: add a `docker-compose.yml` and start with `docker compose up`
3. **Error Handling & Logging**: log every request and return consistent error messages
4. **API Key Authentication**: secure the endpoint with `X-API-Key`, read from an environment variable

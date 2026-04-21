# AirSense — Air Quality Intelligence Platform 🌍

> **AI221 MLOps Project** | Domain: Earth & Environmental Intelligence

A production-grade ML system that predicts, forecasts, and analyses Beijing air quality data using a fully automated MLOps pipeline — 6 ML models, 4 Docker services, CI/CD, and a live glassmorphism dashboard.

---

## 🏗️ Architecture

```
Raw CSV Data (420K rows, 12 Beijing stations)
        │
        ▼
┌─────────────────────────┐
│  Prefect  (port 4200)   │  ← Orchestrates: ingest → preprocess → train → notify
└───────────┬─────────────┘
            │  tracks every experiment
            ▼
┌─────────────────────────┐
│  MLflow   (port 5000)   │  ← Logs params, metrics, model artifacts per run
└───────────┬─────────────┘
            │  saves .joblib / .pkl to artifacts/
            ▼
┌─────────────────────────┐
│  FastAPI  (port 8000)   │  ← REST API — loads all models at startup
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Frontend (port 8080)   │  ← Glassmorphism SPA — all charts call the API live
└─────────────────────────┘
```

---


## 🗂️ Project Structure

```
ML-PROJ/
├── app/                    # FastAPI application
│   ├── main.py             # Entry point + lifespan model loading
│   ├── routers.py          # All API endpoint handlers
│   ├── schemas.py          # Pydantic request/response models
│   ├── model_store.py      # Artifact loading cache (loaded at startup)
│   └── middleware.py       # Request logging middleware
├── ml/                     # ML training scripts (one per task)
│   ├── train_classification.py   # XGBoost + Random Forest → AQI category
│   ├── train_regression.py       # GBM + RF + Ridge → PM2.5 µg/m³
│   ├── train_timeseries.py       # Prophet + ARIMA → 24h forecast
│   ├── train_clustering.py       # K-Means + DBSCAN → station profiles
│   ├── train_dimensionality.py   # PCA + t-SNE → 2D projections
│   └── train_recommendation.py   # Content-based activity window recommendation
├── data/                   # Data pipeline modules
│   ├── ingest.py           # Dataset loaders (PRSA, GlobalAQI, UCI)
│   ├── preprocess.py       # Cleaning, imputation, encoding
│   └── feature_engineering.py  # Lag/rolling features, AQI labels, shared utils
├── pipelines/
│   └── training_flow.py    # Prefect orchestration flow (persist_result=False)
├── tests/                  # Pytest + DeepChecks test suites
├── frontend/               # Vanilla JS/CSS glassmorphism SPA (served by NGINX)
├── datasets/               # Raw CSV files (not committed)
├── artifacts/              # Saved model .joblib/.pkl files
├── .github/workflows/      # CI/CD: lint, test, train, docker_build
├── Dockerfile              # Multi-stage build (deps → app)
├── docker-compose.yml      # 4-service orchestration
└── requirements.txt
```

---

## 🧠 ML Tasks & Results

| Task | Models Compared | Winner | Key Metric |
|---|---|---|---|
| **Classification** | XGBoost, Random Forest | XGBoost | F1=0.862, Acc=0.866 |
| **Regression** | GBM, Random Forest, Ridge | GBM | RMSE=13.15, R²≈0.95 |
| **Time Series** | Prophet, ARIMA | ARIMA | RMSE=68.65 |
| **Clustering** | K-Means (k=3–5), DBSCAN | K-Means | Best silhouette score |
| **Dim. Reduction** | PCA (95% var), t-SNE | — | Used for dashboard viz |
| **Recommendation** | Content-based filtering | — | `POST /recommend/activity-window` |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop running
- Python 3.10+

### 1. Clone & configure

```bash
git clone <repo-url>
cd ML-PROJ
cp .env.example .env   # add your Discord webhook URL if desired
```

### 2. Start all services

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend Dashboard | http://localhost:8080 |
| FastAPI (health check) | http://localhost:8000/health |
| MLflow Tracking | http://localhost:5000 |
| Prefect Orchestration | http://localhost:4200 |

> **Note:** Swagger/ReDoc (`/docs`) is disabled by default. Set `ENABLE_DOCS=true` in `.env` and restart `airsense-api` to enable it locally.

### 3. Train models (local, fast)

```bash
# Requires: pip install -r requirements.txt
python pipelines/training_flow.py
```

Expected total runtime: ~15 minutes on a modern CPU.

> **Important:** The pipeline uses `PREFECT_API_URL=http://localhost:4200/api` from `.env` to report runs to the Docker Prefect UI. All tasks use `persist_result=False` so DataFrames are passed in-memory — this keeps training fast while still showing the full run in the Prefect UI at port 4200.

### 4. Restart the API to load new models

```bash
docker compose restart api
```

---

## 🧪 Running Tests

```bash
# Unit + integration tests (fast, no trained models needed)
pytest tests/test_data.py tests/test_api.py tests/test_models.py -v --tb=short

# ML quality tests with DeepChecks (requires trained models in artifacts/)
pytest tests/test_deepchecks.py -v --tb=short --timeout=180

# Lint
python -m ruff check .
```

---

## ⚙️ CI/CD Pipelines

| Workflow | Trigger | What it does |
|---|---|---|
| `lint.yml` | Every push | `ruff check` — fails on any lint error |
| `test.yml` | Every push / PR to main | pytest unit + integration tests |
| `train.yml` | Push to `main` | Full pipeline retrain + DeepChecks + commit artifacts |
| `docker_build.yml` | Push to `main` | Build Docker image + smoke test |

### Required GitHub Secrets

| Secret | Purpose |
|---|---|
| `DISCORD_WEBHOOK_URL` | Pipeline completion / failure notifications |

---

## 📊 Datasets

| Dataset | Rows | Usage |
|---|---|---|
| PRSA Beijing (12 stations) | ~420K hourly | Primary training data (2013–2017) |
| Global AQI + Lat/Long | ~23K cities | Global map visualisation |
| UCI Air Quality | ~9K hourly | Secondary experiments |

---

## 🔧 Key Configuration

All configuration is via environment variables (`.env`):

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow backend (file store) |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect server for local runs |
| `ARTIFACTS_PATH` | `artifacts` | Where trained models are saved |
| `DATASETS_PATH` | `datasets` | Raw CSV location |
| `ENABLE_DOCS` | `false` | Enable Swagger UI at `/docs` |
| `DISCORD_WEBHOOK_URL` | — | Discord notification on pipeline complete/fail |

### MLflow Version Notice

MLflow is pinned to `==2.10.2` in `requirements.txt`. **Do not upgrade** without first adding `run_uuid: <run_id>` to all `mlruns/**/meta.yaml` files (excluding `models/`). See the comment in `requirements.txt` for details.

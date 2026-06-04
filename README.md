<div align="center">

# 🌍 AirSense

### Air Quality Intelligence Platform

**A production-grade MLOps system for predicting, forecasting, and analysing Beijing air quality**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Prefect](https://img.shields.io/badge/Prefect-024DFD?style=for-the-badge&logo=prefect&logoColor=white)](https://prefect.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)

> **AI221 MLOps Project** — 6 ML models · 4 Docker services · Full CI/CD · Live glassmorphism dashboard

</div>

---

## 🏗️ System Architecture

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
│  MLflow   (port 5000)   │  ← Logs params, metrics & model artifacts per run
└───────────┬─────────────┘
            │  saves .joblib / .pkl to artifacts/
            ▼
┌─────────────────────────┐
│  FastAPI  (port 8000)   │  ← REST API — loads all 6 models at startup
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Frontend (port 8080)   │  ← Glassmorphism SPA — all charts call the API live
└─────────────────────────┘
```

---

## 🧠 ML Tasks & Results

| Task | Models | Winner | Key Metric |
|------|--------|--------|------------|
| 🏷️ **Classification** | XGBoost, Random Forest | XGBoost | F1=0.862, Acc=0.866 |
| 📈 **Regression** | GBM, Random Forest, Ridge | GBM | RMSE=13.15, R²≈0.95 |
| 🕐 **Time Series** | Prophet, ARIMA | ARIMA | RMSE=68.65 |
| 🔵 **Clustering** | K-Means (k=3–5), DBSCAN | K-Means | Best silhouette score |
| 📉 **Dim. Reduction** | PCA (95% var), t-SNE | — | Dashboard visualisation |
| 🎯 **Recommendation** | Content-based filtering | — | `POST /recommend/activity-window` |

---

## 📁 Project Structure

```
AirSense/
├── app/                         # FastAPI application
│   ├── main.py                  # Entry point + lifespan model loading
│   ├── routers.py               # All API endpoint handlers
│   ├── schemas.py               # Pydantic request/response models
│   ├── model_store.py           # Artifact loading cache
│   └── middleware.py            # Request logging middleware
│
├── ml/                          # ML training scripts (one per task)
│   ├── train_classification.py  # XGBoost + RF → AQI category
│   ├── train_regression.py      # GBM + RF + Ridge → PM2.5 µg/m³
│   ├── train_timeseries.py      # Prophet + ARIMA → 24h forecast
│   ├── train_clustering.py      # K-Means + DBSCAN → station profiles
│   ├── train_dimensionality.py  # PCA + t-SNE → 2D projections
│   └── train_recommendation.py  # Content-based activity window
│
├── data/                        # Data pipeline modules
│   ├── ingest.py                # Dataset loaders (PRSA, GlobalAQI, UCI)
│   ├── preprocess.py            # Cleaning, imputation, encoding
│   └── feature_engineering.py  # Lag/rolling features, AQI labels
│
├── pipelines/
│   └── training_flow.py         # Prefect orchestration flow
│
├── tests/                       # Pytest + DeepChecks test suites
├── frontend/                    # Vanilla JS/CSS glassmorphism SPA (NGINX)
├── datasets/                    # Raw CSV files (not committed)
├── artifacts/                   # Saved model .joblib/.pkl files
├── .github/workflows/           # CI/CD: lint, test, train, docker_build
├── Dockerfile                   # Multi-stage build
├── docker-compose.yml           # 4-service orchestration
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop running
- Python 3.10+

### 1. Clone & configure

```bash
git clone https://github.com/marbo786/AirSense.git
cd AirSense
cp .env.example .env   # add your Discord webhook URL if desired
```

### 2. Start all services

```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| 🖥️ Frontend Dashboard | http://localhost:8080 |
| ⚡ FastAPI (health check) | http://localhost:8000/health |
| 📊 MLflow Tracking | http://localhost:5000 |
| 🔄 Prefect Orchestration | http://localhost:4200 |

> **Note:** Swagger/ReDoc (`/docs`) is disabled by default. Set `ENABLE_DOCS=true` in `.env` and restart `airsense-api` to enable it locally.

### 3. Train models

```bash
# Requires: pip install -r requirements.txt
python pipelines/training_flow.py
```

Expected runtime: ~15 minutes on a modern CPU.

> The pipeline uses `PREFECT_API_URL=http://localhost:4200/api` from `.env`. All tasks use `persist_result=False` so DataFrames are passed in-memory — keeping training fast while still reporting to the Prefect UI.

### 4. Reload the API with new models

```bash
docker compose restart api
```

---

## 🧪 Testing

```bash
# Unit + integration tests (no trained models needed)
pytest tests/test_data.py tests/test_api.py tests/test_models.py -v --tb=short

# ML quality tests with DeepChecks (requires trained models in artifacts/)
pytest tests/test_deepchecks.py -v --tb=short --timeout=180

# Lint
python -m ruff check .
```

---

## ⚙️ CI/CD Pipelines

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `lint.yml` | Every push | `ruff check` — fails on any lint error |
| `test.yml` | Every push / PR to main | pytest unit + integration tests |
| `train.yml` | Push to `main` | Full pipeline retrain + DeepChecks + commit artifacts |
| `docker_build.yml` | Push to `main` | Build Docker image + smoke test |

### Required GitHub Secret

| Secret | Purpose |
|--------|---------|
| `DISCORD_WEBHOOK_URL` | Pipeline completion / failure notifications |

---

## 📊 Datasets

| Dataset | Rows | Usage |
|---------|------|-------|
| PRSA Beijing (12 stations) | ~420K hourly | Primary training data (2013–2017) |
| Global AQI + Lat/Long | ~23K cities | Global map visualisation |
| UCI Air Quality | ~9K hourly | Secondary experiments |

---

## 🔧 Configuration

All config is via environment variables (`.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow backend (file store) |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect server for local runs |
| `ARTIFACTS_PATH` | `artifacts` | Where trained models are saved |
| `DATASETS_PATH` | `datasets` | Raw CSV location |
| `ENABLE_DOCS` | `false` | Enable Swagger UI at `/docs` |
| `DISCORD_WEBHOOK_URL` | — | Discord notification on pipeline events |

> **MLflow version notice:** MLflow is pinned to `==2.10.2`. Do not upgrade without adding `run_uuid: <run_id>` to all `mlruns/**/meta.yaml` files first. See `requirements.txt` for details.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🤖 ML | scikit-learn, XGBoost, Prophet, statsmodels |
| 🔁 Orchestration | Prefect |
| 📦 Experiment Tracking | MLflow |
| ⚡ API | FastAPI + Pydantic |
| 🖥️ Frontend | Vanilla JS/CSS (Glassmorphism SPA), NGINX |
| 🐳 Infra | Docker, Docker Compose |
| ✅ Testing | Pytest, DeepChecks |
| 🔍 Linting | Ruff |
| 🚀 CI/CD | GitHub Actions |

---



</div>

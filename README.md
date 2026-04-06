# AirSense — Air Quality Intelligence Platform 🌍

> **AI221 MLOps Project** | Domain: Earth & Environmental Intelligence

A production-grade ML system that predicts, forecasts, and analyses Beijing air quality data using a fully automated MLOps pipeline.

---

## 🗂️ Project Structure

```
ML-PROJ/
├── app/                    # FastAPI application
│   ├── main.py             # App entry point + lifespan
│   ├── routers.py          # All endpoint route handlers
│   ├── schemas.py          # Pydantic request/response models
│   ├── model_store.py      # Artifact loading cache
│   └── middleware.py       # Request logging middleware
├── ml/                     # ML training scripts
│   ├── train_classification.py   # XGBoost + Random Forest (AQI category)
│   ├── train_regression.py       # GBM + RF + Ridge (PM2.5)
│   ├── train_timeseries.py       # Prophet + ARIMA (24h forecast)
│   ├── train_clustering.py       # K-Means + DBSCAN (station profiles)
│   ├── train_dimensionality.py   # PCA + t-SNE
│   └── train_recommendation.py   # Activity window recommendation
├── data/                   # Data pipeline modules
│   ├── ingest.py           # Dataset loaders
│   ├── preprocess.py       # Cleaning, imputation, encoding
│   └── feature_engineering.py  # Lag/rolling features, AQI labels
├── pipelines/
│   └── training_flow.py    # Prefect orchestration flow
├── tests/                  # Pytest + DeepChecks
│   ├── test_data.py
│   ├── test_api.py
│   ├── test_models.py
│   └── test_deepchecks.py
├── dashboard/
│   └── app.py              # Streamlit 5-page dashboard
├── datasets/               # Raw CSV files
├── artifacts/              # Saved models + evaluation outputs
├── .github/workflows/      # CI/CD pipelines
├── Dockerfile              # Multi-stage container
├── docker-compose.yml      # 4-service orchestration
└── requirements.txt
```

---

## 🧠 ML Tasks

| Task | Models | Endpoint |
|---|---|---|
| **Classification** | XGBoost, Random Forest | `POST /predict/aqi-category` |
| **Regression** | Gradient Boosting, RF, Ridge | `POST /predict/pm25` |
| **Time Series** | Prophet, ARIMA | `POST /forecast/timeseries` |
| **Clustering** | K-Means, DBSCAN | `POST /cluster/station` |
| **Dim. Reduction** | PCA, t-SNE | Used in dashboard visualisations |
| **Recommendation** | Content-based filtering | `POST /recommend/activity-window` |

---

## 🚀 Quick Start

### 1. Setup environment

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your Discord webhook URL
```

### 2. Train models (Prefect pipeline)

```bash
python pipelines/training_flow.py
```

### 3. Start FastAPI

```bash
uvicorn app.main:app --reload
# Swagger UI → http://localhost:8000/docs
```

### 4. Start Streamlit dashboard

```bash
streamlit run dashboard/app.py
# Dashboard → http://localhost:8501
```

### 5. Docker Compose (all services)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prefect | http://localhost:4200 |
| Streamlit | http://localhost:8501 |

---

## 🧪 Running Tests

```bash
# Unit + integration tests
pytest tests/test_data.py tests/test_api.py tests/test_models.py -v

# DeepChecks ML tests (requires trained models)
pytest tests/test_deepchecks.py -v

# Lint
ruff check .
```

---

## ⚙️ CI/CD Pipelines

| Workflow | Trigger | Purpose |
|---|---|---|
| `lint.yml` | Every push | ruff code quality |
| `test.yml` | Every push | pytest unit + integration |
| `train.yml` | Push to `main` | Retrain models + DeepChecks |
| `docker_build.yml` | Push to `main` | Build + smoke test Docker image |

### GitHub Secret Required

Add `DISCORD_WEBHOOK_URL` in **Settings → Secrets → Actions** for Discord notifications.

---

## 📊 Datasets

| Dataset | Rows | Usage |
|---|---|---|
| PRSA Beijing (12 stations) | ~420K | Primary training data |
| Global AQI + Lat/Long | ~23K | Classification + map viz |
| UCI Air Quality | ~9K | Secondary experiments |

---

## 🏗️ Architecture

```
Datasets → Prefect Pipeline → MLflow Tracking → Artifacts
                                                    ↓
                             FastAPI ← model_store.py
                                ↓
                         Streamlit Dashboard
                         GitHub Actions CI/CD
                         Docker Compose
```

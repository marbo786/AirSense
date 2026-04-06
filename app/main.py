"""
AirSense FastAPI Application — main entry point.
Loads all ML models at startup, registers all routers, and mounts logging middleware.
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import model_store
from app.middleware import LoggingMiddleware
from app.routers import (
    cluster_router,
    forecast_router,
    health_router,
    metrics_router,
    predict_router,
    recommend_router,
    upload_router,
)

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("airsense")

# ── MLflow config ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌍 AirSense starting — loading models...")
    model_store.load_all()
    logger.info("✅ Models ready: %s", model_store.loaded_names())
    yield
    logger.info("👋 AirSense shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AirSense — Air Quality Intelligence API",
    description=(
        "Production-grade ML API for Beijing air quality prediction, "
        "forecasting, clustering, and activity recommendation.\n\n"
        "**Domain**: Earth & Environmental Intelligence\n"
        "**Models**: XGBoost, GBM, Prophet, ARIMA, K-Means, DBSCAN, PCA"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(forecast_router)
app.include_router(cluster_router)
app.include_router(recommend_router)
app.include_router(upload_router)
app.include_router(metrics_router)

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "AirSense",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/predict/aqi-category",
            "/predict/pm25",
            "/forecast/timeseries",
            "/cluster/station",
            "/recommend/activity-window",
            "/upload/csv",
        ],
    }

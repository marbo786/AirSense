"""
FastAPI routers for prediction, forecasting, clustering, recommendation, upload, and health.
"""

import io
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app import model_store
from app.schemas import (
    AQIClassifyRequest,
    AQIClassifyResponse,
    BatchPredictionResponse,
    ClusterRequest,
    ClusterResponse,
    ExperimentsResponse,
    ExperimentSummary,
    ForecastRequest,
    ForecastResponse,
    GlobalMapResponse,
    HealthResponse,
    PM25Request,
    PM25Response,
    ProjectionPoint,
    ProjectionsResponse,
    RecommendRequest,
    RecommendResponse,
    TimeSeriesResponse,
)
from data.ingest import PRSA_STATIONS

logger = logging.getLogger(__name__)

ARTIFACTS = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))

AQI_MAP = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy for Sensitive Groups",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous",
}

CLUSTER_DESC = {
    0: "Low-pollution zone (clean air profile)",
    1: "Moderate-pollution urban zone",
    2: "High-pollution industrial zone",
    3: "Heavy-traffic hotspot",
    4: "Seasonal pollution zone",
}

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", 10000))
ALLOWED_STATIONS = set(PRSA_STATIONS)


@lru_cache(maxsize=1)
def _global_map_nodes() -> list[dict]:
    df = pd.read_csv("datasets/AQI and Lat Long of Countries.csv").fillna(0)
    return [
        {
            "city": str(row.get("City", "Unknown")),
            "country": str(row.get("Country", "Unknown")),
            "lat": float(row.get("lat", 0.0)),
            "lng": float(row.get("lng", 0.0)),
            "aqi_value": float(row.get("AQI Value", 0.0)),
            "aqi_category": str(row.get("AQI Category", "Unknown")),
        }
        for _, row in df.iterrows()
    ]


@lru_cache(maxsize=1)
def _projection_points() -> list[ProjectionPoint]:
    path = ARTIFACTS / "projections.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna(0)
    points = [
        ProjectionPoint(
            pca_x=float(row.get("pca_x", 0)),
            pca_y=float(row.get("pca_y", 0)),
            tsne_x=float(row.get("tsne_x", 0)),
            tsne_y=float(row.get("tsne_y", 0)),
            aqi_category=str(row.get("aqi_category", "Unknown")),
            station=str(row.get("station", "Unknown")),
        )
        for _, row in df.iterrows()
    ]
    return points


@lru_cache(maxsize=16)
def _weekly_station_series(station: str) -> tuple[list[str], list[float]]:
    path = f"datasets/PRSA_Data_{station}_20130301-20170228.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    ts = df.set_index("datetime")["PM2.5"].clip(lower=0)
    weekly = ts.resample("W").mean().fillna(0)
    dates = [d.strftime("%Y-%m-%d") for d in weekly.index]
    values = [round(float(v), 2) for v in weekly.values]
    return dates, values


def _pm25_to_category(pm25: float) -> str:
    if pm25 <= 12:
        return "Good"
    if pm25 <= 35.4:
        return "Moderate"
    if pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    if pm25 <= 150.4:
        return "Unhealthy"
    if pm25 <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


# ── Health ─────────────────────────────────────────────────────────────────────

health_router = APIRouter(tags=["Health"])


@health_router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", models_loaded=model_store.loaded_names())


# ── Classification ─────────────────────────────────────────────────────────────

predict_router = APIRouter(prefix="/predict", tags=["Prediction"])


@predict_router.post("/aqi-category", response_model=AQIClassifyResponse)
def predict_aqi_category(req: AQIClassifyRequest):
    clf = model_store.get("classifier")
    scaler = model_store.get("classifier_scaler")
    features = model_store.get("classifier_features")
    if clf is None:
        raise HTTPException(503, "Classifier model not loaded. Run training pipeline first.")

    feature_map = {
        "PM10": req.PM10,
        "SO2": req.SO2,
        "NO2": req.NO2,
        "CO": req.CO,
        "O3": req.O3,
        "TEMP": req.TEMP,
        "PRES": req.PRES,
        "DEWP": req.DEWP,
        "RAIN": req.RAIN,
        "WSPM": req.WSPM,
        "wd_encoded": req.wd_encoded,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "month_sin": 0.0,
        "month_cos": 1.0,
        "PM2.5_lag1": req.PM2_5_lag1,
        "PM2.5_lag6": req.PM2_5_lag6,
        "PM2.5_lag24": req.PM2_5_lag24,
        "PM2.5_roll_mean3": req.PM2_5_roll_mean3,
        "PM2.5_roll_mean24": req.PM2_5_roll_mean24,
    }
    X = np.array([[feature_map.get(f, 0.0) for f in features]])
    if scaler:
        X = scaler.transform(X)

    label = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X).max()) if hasattr(clf, "predict_proba") else 1.0
    return AQIClassifyResponse(
        aqi_label=label,
        aqi_category=AQI_MAP.get(label, "Unknown"),
        confidence=round(prob, 3),
    )


@predict_router.get("/map-global", response_model=GlobalMapResponse)
def get_global_map():
    try:
        return GlobalMapResponse(nodes=_global_map_nodes())
    except Exception as e:
        logger.error(f"Failed to load map data: {e}")
        raise HTTPException(500, f"Error loading map data: {e}")


@predict_router.post("/pm25", response_model=PM25Response)
def predict_pm25(req: PM25Request):
    reg = model_store.get("regressor")
    scaler = model_store.get("regressor_scaler")
    features = model_store.get("regressor_features")
    if reg is None:
        raise HTTPException(503, "Regressor not loaded.")

    feature_map = {
        "PM10": req.PM10,
        "SO2": req.SO2,
        "NO2": req.NO2,
        "CO": req.CO,
        "O3": req.O3,
        "TEMP": req.TEMP,
        "PRES": req.PRES,
        "DEWP": req.DEWP,
        "RAIN": req.RAIN,
        "WSPM": req.WSPM,
        "wd_encoded": req.wd_encoded,
        "hour_sin": req.hour_sin,
        "hour_cos": req.hour_cos,
        "month_sin": req.month_sin,
        "month_cos": req.month_cos,
        "PM2.5_lag1": req.PM2_5_lag1,
        "PM2.5_lag3": req.PM2_5_lag3,
        "PM2.5_lag6": req.PM2_5_lag6,
        "PM2.5_lag12": req.PM2_5_lag12,
        "PM2.5_lag24": req.PM2_5_lag24,
        "PM2.5_roll_mean3": req.PM2_5_roll_mean3,
        "PM2.5_roll_mean6": req.PM2_5_roll_mean6,
        "PM2.5_roll_mean24": req.PM2_5_roll_mean24,
        "PM2.5_roll_std3": req.PM2_5_roll_std3,
    }
    X = np.array([[feature_map.get(f, 0.0) for f in features]])
    if scaler:
        X = scaler.transform(X)

    pm25 = float(max(0, reg.predict(X)[0]))
    return PM25Response(
        pm25_predicted=round(pm25, 2),
        aqi_category=_pm25_to_category(pm25),
    )


# ── Time Series ────────────────────────────────────────────────────────────────

forecast_router = APIRouter(prefix="/forecast", tags=["Forecasting"])


@forecast_router.post("/timeseries", response_model=ForecastResponse)
def forecast_timeseries(req: ForecastRequest):
    forecaster = model_store.get("forecaster")
    forecaster_type = model_store.get("forecaster_type", "Unknown")
    if forecaster is None:
        raise HTTPException(503, "Forecaster not loaded.")

    horizon = req.horizon
    try:
        if forecaster_type == "Prophet":
            future = forecaster.make_future_dataframe(periods=horizon, freq="h")
            preds = forecaster.predict(future).tail(horizon)[
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ]
            forecast_list = [
                {
                    "hour_offset": i + 1,
                    "datetime": str(row["ds"]),
                    "pm25_predicted": round(max(0, row["yhat"]), 2),
                    "lower": round(max(0, row["yhat_lower"]), 2),
                    "upper": round(max(0, row["yhat_upper"]), 2),
                }
                for i, (_, row) in enumerate(preds.iterrows())
            ]
        else:  # ARIMA
            preds = forecaster.forecast(steps=horizon)
            forecast_list = [
                {"hour_offset": i + 1, "pm25_predicted": round(max(0, float(v)), 2)}
                for i, v in enumerate(preds)
            ]
    except Exception as e:
        raise HTTPException(500, f"Forecast failed: {e}")

    return ForecastResponse(station=req.station, horizon=horizon, forecast=forecast_list)


# ── Clustering ─────────────────────────────────────────────────────────────────

cluster_router = APIRouter(prefix="/cluster", tags=["Clustering"])


@cluster_router.post("/station", response_model=ClusterResponse)
def cluster_station(req: ClusterRequest):
    clusterer = model_store.get("clusterer")
    scaler = model_store.get("clusterer_scaler")
    features = model_store.get("clusterer_features")
    if clusterer is None:
        raise HTTPException(503, "Clusterer not loaded.")

    feature_map = {
        "PM2.5_mean": req.PM2_5_mean,
        "PM10_mean": req.PM10_mean,
        "SO2_mean": req.SO2_mean,
        "NO2_mean": req.NO2_mean,
        "CO_mean": req.CO_mean,
        "O3_mean": req.O3_mean,
    }
    X = np.array([[feature_map.get(f, 0.0) for f in features]])
    if scaler:
        X = scaler.transform(X)

    label = int(clusterer.predict(X)[0]) if hasattr(clusterer, "predict") else -1
    return ClusterResponse(
        cluster_id=label,
        cluster_description=CLUSTER_DESC.get(label, "Unknown cluster profile"),
    )


# ── Recommendation ─────────────────────────────────────────────────────────────

recommend_router = APIRouter(prefix="/recommend", tags=["Recommendation"])


@recommend_router.post("/activity-window", response_model=RecommendResponse)
def recommend_activity(req: RecommendRequest):
    from ml.train_recommendation import recommend

    profile = model_store.get("recommendation_profile")
    if profile is None:
        raise HTTPException(503, "Recommendation profile not loaded.")

    recs = recommend(req.station, top_n=req.top_n, profile=profile)
    if not recs:
        raise HTTPException(404, f"Station '{req.station}' not found in profile.")
    return RecommendResponse(station=req.station, recommendations=recs)


# ── CSV Batch Upload ───────────────────────────────────────────────────────────

upload_router = APIRouter(prefix="/upload", tags=["Batch Upload"])


@upload_router.post("/csv", response_model=BatchPredictionResponse)
async def batch_predict_csv(file: UploadFile = File(...)):
    """Accept a CSV file and return PM2.5 regression predictions for each row."""
    reg = model_store.get("regressor")
    scaler = model_store.get("regressor_scaler")
    features = model_store.get("regressor_features")
    if reg is None:
        raise HTTPException(503, "Regressor not loaded.")

    if file.content_type not in {"text/csv", "application/vnd.ms-excel"}:
        raise HTTPException(400, "Invalid content type. Please upload a CSV file.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"CSV too large. Max size is {MAX_UPLOAD_BYTES} bytes.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"CSV has too many rows. Max rows: {MAX_UPLOAD_ROWS}.")

    available = [f for f in features if f in df.columns]
    if not available:
        raise HTTPException(400, f"CSV must contain at least some of: {features}")

    X = df[[f for f in features if f in df.columns]].fillna(0).values
    # Pad missing columns with zeros
    if X.shape[1] < len(features):
        padded = np.zeros((X.shape[0], len(features)))
        col_idx = [features.index(c) for c in available]
        for i, idx in enumerate(col_idx):
            padded[:, idx] = X[:, i]
        X = padded

    if scaler:
        X = scaler.transform(X)

    preds = reg.predict(X)
    results = [
        {
            "row": i,
            "pm25_predicted": round(max(0, float(p)), 2),
            "aqi_category": _pm25_to_category(float(p)),
        }
        for i, p in enumerate(preds)
    ]
    return BatchPredictionResponse(rows_processed=len(results), predictions=results)


# ── Metrics ────────────────────────────────────────────────────────────────────

metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])


@metrics_router.get("/experiments", response_model=ExperimentsResponse)
def get_experiments():
    path = ARTIFACTS / "training_summary.json"
    if not path.exists():
        return ExperimentsResponse(experiments=[])
    try:
        with open(path, "r") as f:
            data = json.load(f)
        summaries = []
        for task, payload in data.items():
            summaries.append(
                ExperimentSummary(
                    task=task.replace("_", " ").title(),
                    best_model=payload.get("best_model", "N/A"),
                    metrics=payload.get("metrics", {}),
                )
            )
        return ExperimentsResponse(experiments=summaries)
    except Exception as e:
        logger.error(f"Failed to read experiments: {e}")
        raise HTTPException(500, str(e))


@metrics_router.get("/projections", response_model=ProjectionsResponse)
def get_projections():
    try:
        points = _projection_points()
        # Limiting to 2500 max for UI performance
        if len(points) > 2500:
            import random

            points = random.sample(points, 2500)
        return ProjectionsResponse(points=points)
    except Exception as e:
        logger.error(f"Failed to read projections: {e}")
        raise HTTPException(500, str(e))


@metrics_router.get("/time-series", response_model=TimeSeriesResponse)
def get_time_series(station: str = "Aotizhongxin"):
    """Fetch resampled weekly averages for a historical PRSA station."""
    # Since downloading 45000 hours to JS is massive, we resample to Weekly
    # In a full prod system we'd use a timeseries DB, but we read the csv here
    if station not in ALLOWED_STATIONS:
        raise HTTPException(400, f"Invalid station. Must be one of: {sorted(ALLOWED_STATIONS)}")

    try:
        dates, values = _weekly_station_series(station)
        return TimeSeriesResponse(dates=dates, values=values, station=station)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset for {station} not found.")
    except Exception as e:
        raise HTTPException(500, f"Error processing time-series: {e}")

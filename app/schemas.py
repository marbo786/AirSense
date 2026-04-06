"""
Pydantic schemas for all FastAPI request / response models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Shared ────────────────────────────────────────────────────────────────────


class MapNode(BaseModel):
    city: str
    country: str
    lat: float
    lng: float
    aqi_value: float
    aqi_category: str


class GlobalMapResponse(BaseModel):
    nodes: list[MapNode]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


# ── Classification ─────────────────────────────────────────────────────────────


class AQIClassifyRequest(BaseModel):
    PM10: float = Field(..., ge=0, description="PM10 concentration µg/m³")
    SO2: float = Field(..., ge=0)
    NO2: float = Field(..., ge=0)
    CO: float = Field(..., ge=0)
    O3: float = Field(..., ge=0)
    TEMP: float = Field(..., description="Temperature °C")
    PRES: float = Field(..., description="Atmospheric pressure hPa")
    DEWP: float = Field(..., description="Dew point °C")
    RAIN: float = Field(0.0, ge=0, description="Rainfall mm")
    WSPM: float = Field(0.0, ge=0, description="Wind speed m/s")
    wd_encoded: int = Field(0, description="Wind direction encoded integer")
    # Optional lag/rolling features (default 0 if not provided)
    PM2_5_lag1: float = Field(0.0, alias="PM2.5_lag1")
    PM2_5_lag6: float = Field(0.0, alias="PM2.5_lag6")
    PM2_5_lag24: float = Field(0.0, alias="PM2.5_lag24")
    PM2_5_roll_mean3: float = Field(0.0, alias="PM2.5_roll_mean3")
    PM2_5_roll_mean24: float = Field(0.0, alias="PM2.5_roll_mean24")

    model_config = {"populate_by_name": True}


class AQIClassifyResponse(BaseModel):
    aqi_label: int
    aqi_category: str
    confidence: float


# ── Regression ─────────────────────────────────────────────────────────────────


class PM25Request(BaseModel):
    PM10: float = Field(..., ge=0)
    SO2: float = Field(..., ge=0)
    NO2: float = Field(..., ge=0)
    CO: float = Field(..., ge=0)
    O3: float = Field(..., ge=0)
    TEMP: float
    PRES: float
    DEWP: float
    RAIN: float = 0.0
    WSPM: float = 0.0
    wd_encoded: int = 0
    hour_sin: float = 0.0
    hour_cos: float = 1.0
    month_sin: float = 0.0
    month_cos: float = 1.0
    PM2_5_lag1: float = Field(0.0, alias="PM2.5_lag1")
    PM2_5_lag3: float = Field(0.0, alias="PM2.5_lag3")
    PM2_5_lag6: float = Field(0.0, alias="PM2.5_lag6")
    PM2_5_lag12: float = Field(0.0, alias="PM2.5_lag12")
    PM2_5_lag24: float = Field(0.0, alias="PM2.5_lag24")
    PM2_5_roll_mean3: float = Field(0.0, alias="PM2.5_roll_mean3")
    PM2_5_roll_mean6: float = Field(0.0, alias="PM2.5_roll_mean6")
    PM2_5_roll_mean24: float = Field(0.0, alias="PM2.5_roll_mean24")
    PM2_5_roll_std3: float = Field(0.0, alias="PM2.5_roll_std3")

    model_config = {"populate_by_name": True}


class PM25Response(BaseModel):
    pm25_predicted: float
    unit: str = "µg/m³"
    aqi_category: str


# ── Time Series ────────────────────────────────────────────────────────────────


class ForecastRequest(BaseModel):
    station: str = Field("Aotizhongxin", description="Beijing station name")
    horizon: int = Field(24, ge=1, le=72, description="Forecast horizon in hours")


class ForecastResponse(BaseModel):
    station: str
    horizon: int
    forecast: list[dict[str, Any]]


# ── Clustering ─────────────────────────────────────────────────────────────────


class ClusterRequest(BaseModel):
    PM2_5_mean: float = Field(..., alias="PM2.5_mean")
    PM10_mean: float
    SO2_mean: float
    NO2_mean: float
    CO_mean: float
    O3_mean: float
    model_config = {"populate_by_name": True}


class ClusterResponse(BaseModel):
    cluster_id: int
    cluster_description: str


# ── Recommendation ─────────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    station: str
    top_n: int = Field(5, ge=1, le=24)


class RecommendResponse(BaseModel):
    station: str
    recommendations: list[dict[str, Any]]


# ── Batch upload ───────────────────────────────────────────────────────────────


class BatchPredictionResponse(BaseModel):
    rows_processed: int
    predictions: list[dict[str, Any]]


# ── Metrics ────────────────────────────────────────────────────────────────────


class TimeSeriesResponse(BaseModel):
    dates: list[str]
    values: list[float]
    station: str


class ExperimentSummary(BaseModel):
    task: str
    best_model: str
    metrics: dict[str, Any]


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentSummary]


class ProjectionPoint(BaseModel):
    pca_x: float
    pca_y: float
    tsne_x: float
    tsne_y: float
    aqi_category: str
    station: str


class ProjectionsResponse(BaseModel):
    points: list[ProjectionPoint]

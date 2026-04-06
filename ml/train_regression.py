"""
Regression: Predict PM2.5 concentration (µg/m³)
using Gradient Boosting, Random Forest Regressor, and Linear Regression.
MLflow tracks all experiments; best model saved to artifacts/.
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
ARTIFACTS_PATH.mkdir(exist_ok=True)

FEATURE_COLS = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "wd_encoded",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "PM2.5_lag1",
    "PM2.5_lag3",
    "PM2.5_lag6",
    "PM2.5_lag12",
    "PM2.5_lag24",
    "PM2.5_roll_mean3",
    "PM2.5_roll_mean6",
    "PM2.5_roll_mean24",
    "PM2.5_roll_std3",
]
TARGET_COL = "PM2.5"


def prepare_data(df: pd.DataFrame) -> tuple:
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, available


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run(df: pd.DataFrame) -> dict:
    """Run PM2.5 regression experiments. Returns best model info."""
    mlflow.set_experiment("PM25_Regression")
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(df)

    experiments = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Ridge": Ridge(alpha=1.0),
    }

    results = {}
    for name, model in experiments.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = _metrics(y_test, preds)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            results[name] = {"model": model, "metrics": metrics}
            logger.info("%s Regression — %s", name, metrics)

    # Best = lowest RMSE
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, ARTIFACTS_PATH / "regressor.joblib")
    joblib.dump(scaler, ARTIFACTS_PATH / "regressor_scaler.joblib")
    joblib.dump(features, ARTIFACTS_PATH / "regressor_features.joblib")
    logger.info("Best regressor: %s — saved to artifacts/", best_name)

    return {
        "best_model": best_name,
        "metrics": results[best_name]["metrics"],
        "all_results": {k: v["metrics"] for k, v in results.items()},
    }

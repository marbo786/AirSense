"""
Classification: Predict AQI category (Good/Moderate/Unhealthy/etc.)
from pollutant + weather features using Random Forest and XGBoost.
MLflow tracks all experiments; best model is saved to artifacts/.
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
    "PM2.5_lag6",
    "PM2.5_lag24",
    "PM2.5_roll_mean3",
    "PM2.5_roll_mean24",
]
TARGET_COL = "AQI_Label"


def prepare_data(df: pd.DataFrame) -> tuple:
    """Extract features and labels, then split."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, available


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple[RandomForestClassifier, dict]:
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple[XGBClassifier, dict]:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    return model, metrics


def run(df: pd.DataFrame) -> dict:
    """Run classification experiment. Returns best model info."""
    mlflow.set_experiment("AQI_Classification")
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(df)

    results = {}

    # Random Forest
    with mlflow.start_run(run_name="RandomForest"):
        model_rf, metrics_rf = train_random_forest(X_train, y_train, X_test, y_test)
        mlflow.log_params({"n_estimators": 200, "max_depth": 12})
        mlflow.log_metrics(metrics_rf)
        mlflow.sklearn.log_model(model_rf, "model")
        results["RandomForest"] = {"model": model_rf, "metrics": metrics_rf}
        logger.info("RF Classification — %s", metrics_rf)

    # XGBoost
    with mlflow.start_run(run_name="XGBoost"):
        model_xgb, metrics_xgb = train_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_params({"n_estimators": 300, "max_depth": 8, "lr": 0.05})
        mlflow.log_metrics(metrics_xgb)
        mlflow.xgboost.log_model(model_xgb, "model")
        results["XGBoost"] = {"model": model_xgb, "metrics": metrics_xgb}
        logger.info("XGB Classification — %s", metrics_xgb)

    # Pick best by F1
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_macro"])
    best_model = results[best_name]["model"]

    # Save best model + scaler
    joblib.dump(best_model, ARTIFACTS_PATH / "classifier.joblib")
    joblib.dump(scaler, ARTIFACTS_PATH / "classifier_scaler.joblib")
    joblib.dump(features, ARTIFACTS_PATH / "classifier_features.joblib")
    logger.info("Best classifier: %s — saved to artifacts/", best_name)

    return {
        "best_model": best_name,
        "metrics": results[best_name]["metrics"],
        "all_results": {k: v["metrics"] for k, v in results.items()},
    }

"""
Time Series: 24-hour PM2.5 forecast using Prophet and ARIMA.
MLflow tracks metrics; best forecaster saved to artifacts/.
"""

import logging
import os
import pickle
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
ARTIFACTS_PATH.mkdir(exist_ok=True)

FORECAST_HORIZON = 24  # hours


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred))}


def prepare_ts_data(df: pd.DataFrame, station: str = "Aotizhongxin") -> pd.DataFrame:
    """Return hourly PM2.5 series for a single station."""
    mask = df["station"] == station if "station" in df.columns else pd.Series([True] * len(df))
    ts = df[mask].set_index("datetime")["PM2.5"].dropna().sort_index()
    ts = ts.resample("h").mean().interpolate()
    return ts


def train_prophet(ts: pd.Series) -> tuple:
    """Train Facebook Prophet on hourly PM2.5 series."""
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet not installed. Run: pip install prophet")

    train = ts.iloc[:-FORECAST_HORIZON]
    test = ts.iloc[-FORECAST_HORIZON:]

    prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="h")
    forecast = model.predict(future)
    preds = forecast.tail(FORECAST_HORIZON)["yhat"].values
    preds = np.clip(preds, 0, None)

    return model, _metrics(test.values, preds)


def train_arima(ts: pd.Series) -> tuple:
    """Train ARIMA(2,1,2) on hourly PM2.5 series."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("statsmodels not installed.")

    train = ts.iloc[:-FORECAST_HORIZON]
    test = ts.iloc[-FORECAST_HORIZON:]

    model = ARIMA(train.values, order=(2, 1, 2))
    result = model.fit()
    preds = result.forecast(steps=FORECAST_HORIZON)
    preds = np.clip(preds, 0, None)

    return result, _metrics(test.values, preds)


def run(df: pd.DataFrame) -> dict:
    """Run time series experiments. Returns best model info."""
    mlflow.set_experiment("PM25_TimeSeries")
    ts = prepare_ts_data(df)
    if len(ts) < (FORECAST_HORIZON * 2):
        raise RuntimeError(
            f"Insufficient time-series points ({len(ts)}). Need at least {FORECAST_HORIZON * 2}."
        )
    results = {}

    # Prophet
    try:
        with mlflow.start_run(run_name="Prophet"):
            model_p, metrics_p = train_prophet(ts)
            mlflow.log_params({"changepoint_prior_scale": 0.05, "horizon": FORECAST_HORIZON})
            mlflow.log_metrics(metrics_p)
            results["Prophet"] = {"model": model_p, "metrics": metrics_p}
            logger.info("Prophet TS — %s", metrics_p)
    except Exception as e:
        logger.warning("Prophet training failed: %s", e)

    # ARIMA
    try:
        with mlflow.start_run(run_name="ARIMA"):
            model_a, metrics_a = train_arima(ts)
            mlflow.log_params({"order": "(2,1,2)", "horizon": FORECAST_HORIZON})
            mlflow.log_metrics(metrics_a)
            results["ARIMA"] = {"model": model_a, "metrics": metrics_a}
            logger.info("ARIMA TS — %s", metrics_a)
    except Exception as e:
        logger.warning("ARIMA training failed: %s", e)

    if not results:
        raise RuntimeError("All time series models failed.")

    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best_model = results[best_name]["model"]

    with open(ARTIFACTS_PATH / "forecaster.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(ARTIFACTS_PATH / "forecaster_type.txt", "w") as f:
        f.write(best_name)

    logger.info("Best forecaster: %s — saved.", best_name)
    return {
        "best_model": best_name,
        "metrics": results[best_name]["metrics"],
        "all_results": {k: v["metrics"] for k, v in results.items()},
    }

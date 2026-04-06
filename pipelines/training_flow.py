"""
AirSense Prefect Training Pipeline
Orchestrates the full ML training workflow end-to-end:
  ingest → preprocess → feature engineer → train (all models) → evaluate → save → notify
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path for Prefect tasks
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import timedelta

import requests
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
ARTIFACTS_PATH.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Notification ───────────────────────────────────────────────────────────────


def _discord_notify(message: str, color: int = 0x00FF00) -> None:
    if not DISCORD_WEBHOOK_URL or "YOUR_ID" in DISCORD_WEBHOOK_URL:
        logging.warning("Discord webhook not configured — skipping notification.")
        return
    payload = {
        "embeds": [
            {
                "title": "🌍 AirSense ML Pipeline",
                "description": message,
                "color": color,
            }
        ]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logging.warning("Discord notification failed: %s", e)


# ── Tasks ──────────────────────────────────────────────────────────────────────


@task(
    name="Ingest Data",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def ingest_data():
    logger = get_run_logger()
    from data.ingest import load_all

    datasets = load_all()
    logger.info("Ingested datasets: %s", list(datasets.keys()))
    return datasets


@task(name="Preprocess Data", retries=2, retry_delay_seconds=30)
def preprocess_data(datasets: dict):
    logger = get_run_logger()
    from data.preprocess import preprocess_global_aqi, preprocess_prsa, preprocess_uci

    prsa = preprocess_prsa(datasets["prsa"])
    global_aqi = preprocess_global_aqi(datasets["global_aqi"])
    uci = preprocess_uci(datasets["uci"])
    logger.info(
        "Preprocessed PRSA: %s, GlobalAQI: %s, UCI: %s", prsa.shape, global_aqi.shape, uci.shape
    )
    return {"prsa": prsa, "global_aqi": global_aqi, "uci": uci}


@task(name="Feature Engineering", retries=2, retry_delay_seconds=30)
def feature_engineering(preprocessed: dict):
    logger = get_run_logger()
    from data.feature_engineering import build_prsa_features, build_station_profile

    prsa_feat = build_prsa_features(preprocessed["prsa"])
    station_profile = build_station_profile(prsa_feat)
    logger.info(
        "PRSA with features: %s | Station profiles: %s", prsa_feat.shape, station_profile.shape
    )
    return {
        "prsa_feat": prsa_feat,
        "station_profile": station_profile,
        "global_aqi": preprocessed["global_aqi"],
    }


@task(name="Train Classification", retries=1, retry_delay_seconds=60)
def train_classification(engineered: dict):
    logger = get_run_logger()
    from ml.train_classification import run

    result = run(engineered["prsa_feat"])
    logger.info(
        "Classification done — best: %s, metrics: %s", result["best_model"], result["metrics"]
    )
    return result


@task(name="Train Regression", retries=1, retry_delay_seconds=60)
def train_regression(engineered: dict):
    logger = get_run_logger()
    from ml.train_regression import run

    result = run(engineered["prsa_feat"])
    logger.info(
        "Regression done — best: %s, RMSE: %.2f", result["best_model"], result["metrics"]["rmse"]
    )
    return result


@task(name="Train Time Series", retries=1, retry_delay_seconds=60)
def train_timeseries(engineered: dict):
    logger = get_run_logger()
    from ml.train_timeseries import run

    result = run(engineered["prsa_feat"])
    logger.info(
        "TimeSeries done — best: %s, RMSE: %.2f", result["best_model"], result["metrics"]["rmse"]
    )
    return result


@task(name="Train Clustering", retries=1, retry_delay_seconds=60)
def train_clustering(engineered: dict):
    logger = get_run_logger()
    from ml.train_clustering import run

    result = run(engineered["station_profile"])
    logger.info("Clustering done — best: %s", result["best_model"])
    return result


@task(name="Train Dimensionality Reduction", retries=1, retry_delay_seconds=60)
def train_dimensionality(engineered: dict):
    logger = get_run_logger()
    from ml.train_dimensionality import run

    result = run(engineered["prsa_feat"])
    logger.info("Dimensionality Reduction done.")
    return result


@task(name="Train Recommendation", retries=1, retry_delay_seconds=60)
def train_recommendation(engineered: dict):
    logger = get_run_logger()
    from ml.train_recommendation import run

    result = run(engineered["prsa_feat"])
    logger.info("Recommendation profile built — shape: %s", result["profile_shape"])
    return result


@task(name="Save Summary", retries=2, retry_delay_seconds=10)
def save_summary(clf_result, reg_result, ts_result, clust_result):
    logger = get_run_logger()
    summary = {
        "classification": {
            "best_model": clf_result["best_model"],
            "metrics": clf_result["metrics"],
            "all": clf_result["all_results"],
        },
        "regression": {
            "best_model": reg_result["best_model"],
            "metrics": reg_result["metrics"],
            "all": reg_result["all_results"],
        },
        "timeseries": {
            "best_model": ts_result["best_model"],
            "metrics": ts_result["metrics"],
            "all": ts_result["all_results"],
        },
        "clustering": {
            "best_model": clust_result["best_model"],
            "all": clust_result["all_results"],
        },
    }
    out_path = ARTIFACTS_PATH / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to %s", out_path)
    return summary


# ── Flow ───────────────────────────────────────────────────────────────────────


@flow(
    name="AirSense Training Pipeline",
    description="End-to-end ML training: ingest → preprocess → features → train → save → notify",
)
def training_pipeline():
    logger = get_run_logger()
    logger.info("🚀 Starting AirSense training pipeline...")

    try:
        datasets = ingest_data()
        preprocessed = preprocess_data(datasets)
        engineered = feature_engineering(preprocessed)

        # Train all models (sequential to avoid memory issues)
        clf_result = train_classification(engineered)
        reg_result = train_regression(engineered)
        ts_result = train_timeseries(engineered)
        clust_result = train_clustering(engineered)
        train_dimensionality(engineered)
        train_recommendation(engineered)

        save_summary(clf_result, reg_result, ts_result, clust_result)

        msg = (
            f"✅ **Training complete!**\n\n"
            f"**Classification** → {clf_result['best_model']} "
            f"(F1={clf_result['metrics']['f1_macro']:.3f})\n"
            f"**Regression** → {reg_result['best_model']} "
            f"(RMSE={reg_result['metrics']['rmse']:.2f})\n"
            f"**Time Series** → {ts_result['best_model']} "
            f"(RMSE={ts_result['metrics']['rmse']:.2f})\n"
            f"**Clustering** → {clust_result['best_model']}"
        )
        _discord_notify(msg, color=0x00FF00)
        logger.info("✅ Pipeline complete.")

    except Exception as e:
        _discord_notify(f"❌ **Pipeline FAILED!**\n\n`{e}`", color=0xFF0000)
        logger.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    training_pipeline()

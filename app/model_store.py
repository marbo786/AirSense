"""
Model loader — loads all saved ML artifacts at startup and caches them.
Uses a simple dict store so routers can import and use models directly.
"""

import logging
import os
import pickle
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))

# Global model store — populated at startup
_store: dict = {}


def load_all() -> dict:
    """Load all model artifacts into the store. Safe to call at startup."""
    _store.clear()

    _try_load("classifier", ARTIFACTS_PATH / "classifier.joblib")
    _try_load("classifier_scaler", ARTIFACTS_PATH / "classifier_scaler.joblib")
    _try_load("classifier_features", ARTIFACTS_PATH / "classifier_features.joblib")

    _try_load("regressor", ARTIFACTS_PATH / "regressor.joblib")
    _try_load("regressor_scaler", ARTIFACTS_PATH / "regressor_scaler.joblib")
    _try_load("regressor_features", ARTIFACTS_PATH / "regressor_features.joblib")

    _try_load("clusterer", ARTIFACTS_PATH / "clusterer.joblib")
    _try_load("clusterer_scaler", ARTIFACTS_PATH / "clusterer_scaler.joblib")
    _try_load("clusterer_features", ARTIFACTS_PATH / "clusterer_features.joblib")

    _try_load("recommendation_profile", ARTIFACTS_PATH / "recommendation_profile.joblib")

    forecaster_path = ARTIFACTS_PATH / "forecaster.pkl"
    if forecaster_path.exists():
        with open(forecaster_path, "rb") as f:
            _store["forecaster"] = pickle.load(f)
        logger.info("Loaded forecaster")

    forecaster_type_path = ARTIFACTS_PATH / "forecaster_type.txt"
    if forecaster_type_path.exists():
        _store["forecaster_type"] = forecaster_type_path.read_text().strip()

    logger.info("Models loaded: %s", list(_store.keys()))
    return _store


def _try_load(name: str, path: Path) -> None:
    if path.exists():
        _store[name] = joblib.load(path)
        logger.info("Loaded %s", name)
    else:
        logger.warning("Artifact not found: %s", path)


def get(key: str, default=None):
    """Retrieve an artifact from the store."""
    return _store.get(key, default)


def loaded_names() -> list[str]:
    return list(_store.keys())

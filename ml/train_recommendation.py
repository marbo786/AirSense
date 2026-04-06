"""
Recommendation: Suggest safe outdoor activity windows (hours with Good/Moderate AQI)
for a given city using content-based filtering on historical AQI patterns.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
ARTIFACTS_PATH.mkdir(exist_ok=True)

SAFE_CATEGORIES = {"Good", "Moderate"}


def build_station_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (station, hour) pair, compute the fraction of time
    the AQI category is Good or Moderate — the 'safety score'.
    """
    if "AQI_Category" not in df.columns:
        raise ValueError("AQI_Category column required. Run feature_engineering first.")
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

    df["is_safe"] = df["AQI_Category"].isin(SAFE_CATEGORIES).astype(int)

    profile = (
        df.groupby(["station", "hour"])["is_safe"]
        .mean()
        .reset_index()
        .rename(columns={"is_safe": "safety_score"})
    )
    return profile


def run(df: pd.DataFrame) -> dict:
    """Build and save the recommendation profile."""
    profile = build_station_hourly_profile(df)
    joblib.dump(profile, ARTIFACTS_PATH / "recommendation_profile.joblib")
    logger.info("Recommendation profile saved — %d rows", len(profile))
    return {"status": "ok", "profile_shape": profile.shape}


def recommend(
    station: str,
    top_n: int = 5,
    profile: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Return the top_n safest hours of day for the given station.

    Parameters
    ----------
    station : station name (e.g., 'Aotizhongxin')
    top_n   : number of hour windows to return
    profile : pre-loaded profile DataFrame (optional — loads from disk if None)
    """
    if profile is None:
        profile = joblib.load(ARTIFACTS_PATH / "recommendation_profile.joblib")

    mask = profile["station"].str.lower() == station.lower()
    station_data = profile[mask].sort_values("safety_score", ascending=False)

    if station_data.empty:
        logger.warning("Station '%s' not found in profile.", station)
        return []

    top = station_data.head(top_n)
    return [
        {
            "hour": int(row["hour"]),
            "time_label": f"{int(row['hour']):02d}:00 – {int(row['hour']) + 1:02d}:00",
            "safety_score": round(float(row["safety_score"]), 3),
            "recommendation": "✅ Safe" if row["safety_score"] >= 0.7 else "⚠️ Moderate",
        }
        for _, row in top.iterrows()
    ]

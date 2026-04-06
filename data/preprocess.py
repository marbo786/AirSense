"""
Preprocessing module — cleans, imputes, and normalises raw datasets.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# AQI category mapping for Global AQI dataset
AQI_CATEGORY_MAP = {
    "Good": 0,
    "Moderate": 1,
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 3,
    "Very Unhealthy": 4,
    "Hazardous": 5,
}

PRSA_POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
PRSA_WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
PRSA_NUMERIC = PRSA_POLLUTANTS + PRSA_WEATHER


def preprocess_prsa(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the PRSA Beijing dataset."""
    df = df.copy()

    # Build datetime index
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(
            columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
        )
    )
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Impute numeric columns with per-station forward-fill then median
    for col in PRSA_NUMERIC:
        if col in df.columns:
            df[col] = df.groupby("station")[col].transform(lambda s: s.ffill().fillna(s.median()))

    # Encode wind direction
    if "wd" in df.columns:
        le = LabelEncoder()
        df["wd_encoded"] = le.fit_transform(df["wd"].fillna("N"))

    # Drop raw date parts (we have datetime now)
    df = df.drop(columns=["No", "year", "month", "day", "hour"], errors="ignore")

    # Clip obvious sensor errors (negatives on pollutants)
    for col in PRSA_POLLUTANTS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    logger.info("PRSA preprocessed shape: %s", df.shape)
    return df


def preprocess_global_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the global AQI country-level dataset."""
    df = df.copy()
    df = df.dropna(subset=["AQI Value", "AQI Category"])

    # Encode AQI category as integer
    df["AQI_Label"] = df["AQI Category"].map(AQI_CATEGORY_MAP).fillna(1).astype(int)

    # Clip AQI values to valid sensor range
    aqi_cols = [c for c in df.columns if "AQI Value" in c]
    for col in aqi_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0, upper=500)

    logger.info("Global AQI preprocessed shape: %s", df.shape)
    return df


def preprocess_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the UCI air quality sensor dataset."""
    df = df.copy()

    # Parse datetime
    try:
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    except Exception:
        logger.warning("UCI datetime parse failed, skipping datetime column.")

    # UCI uses -200 as sentinel for missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(-200, np.nan)

    # Drop columns that are mostly missing
    df = df.dropna(thresh=int(len(df) * 0.5), axis=1)

    # Impute remaining NaNs with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    df = df.drop(columns=["Date", "Time"], errors="ignore")
    logger.info("UCI preprocessed shape: %s", df.shape)
    return df


def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler_type: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler]:
    """Fit and apply a scaler to selected feature columns. Returns (scaled_df, fitted_scaler)."""
    df = df.copy()
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
    return df, scaler

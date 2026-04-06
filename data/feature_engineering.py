"""
Feature engineering module — creates lag features, rolling statistics,
datetime cyclical encodings, and AQI category labels from raw pollutant values.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_datetime_features(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    """Add cyclical hour/month/day-of-week features from a datetime column."""
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])

    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear

    # Cyclical encoding (sin/cos) avoids discontinuity at midnight / year-end
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    group_col: str | None = "station",
) -> pd.DataFrame:
    """Add lagged values of target_col. Optionally grouped by station."""
    df = df.copy()
    if group_col and group_col in df.columns:
        for lag in lags:
            df[f"{target_col}_lag{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lags:
            df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
    group_col: str | None = "station",
) -> pd.DataFrame:
    """Add rolling mean and std of target_col."""
    df = df.copy()
    for w in windows:
        if group_col and group_col in df.columns:
            rolled = df.groupby(group_col)[target_col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            rolled_std = df.groupby(group_col)[target_col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).std()
            )
        else:
            rolled = df[target_col].shift(1).rolling(w, min_periods=1).mean()
            rolled_std = df[target_col].shift(1).rolling(w, min_periods=1).std()

        df[f"{target_col}_roll_mean{w}"] = rolled
        df[f"{target_col}_roll_std{w}"] = rolled_std.fillna(0)
    return df


def assign_aqi_category_from_pm25(df: pd.DataFrame, pm25_col: str = "PM2.5") -> pd.DataFrame:
    """Derive AQI category label from PM2.5 values using US EPA breakpoints."""
    df = df.copy()

    def _category(pm25: float) -> str:
        if pm25 <= 12:
            return "Good"
        elif pm25 <= 35.4:
            return "Moderate"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive Groups"
        elif pm25 <= 150.4:
            return "Unhealthy"
        elif pm25 <= 250.4:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    df["AQI_Category"] = df[pm25_col].apply(_category)

    category_map = {
        "Good": 0,
        "Moderate": 1,
        "Unhealthy for Sensitive Groups": 2,
        "Unhealthy": 3,
        "Very Unhealthy": 4,
        "Hazardous": 5,
    }
    df["AQI_Label"] = df["AQI_Category"].map(category_map)
    return df


def build_prsa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline for the PRSA dataset."""
    df = add_datetime_features(df, dt_col="datetime")
    df = add_lag_features(df, "PM2.5", lags=[1, 3, 6, 12, 24])
    df = add_lag_features(df, "PM10", lags=[1, 6, 24])
    df = add_rolling_features(df, "PM2.5", windows=[3, 6, 12, 24])
    df = assign_aqi_category_from_pm25(df)

    df = df.dropna(subset=["PM2.5_lag1", "PM2.5_lag24"])
    logger.info("PRSA features shape after engineering: %s", df.shape)
    return df


def build_station_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-station statistics for clustering."""
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    agg = {col: ["mean", "std", "max"] for col in pollutants if col in df.columns}
    profile = df.groupby("station").agg(agg)
    profile.columns = ["_".join(c) for c in profile.columns]
    profile = profile.reset_index()
    logger.info("Station profile shape: %s", profile.shape)
    return profile

"""
Data ingestion module — loads all AirSense datasets from disk.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATASETS_PATH = Path(os.getenv("DATASETS_PATH", "datasets"))

PRSA_STATIONS = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]


def load_prsa_data() -> pd.DataFrame:
    """Load and concatenate all 12 Beijing PRSA station CSV files."""
    frames = []
    for station in PRSA_STATIONS:
        path = DATASETS_PATH / f"PRSA_Data_{station}_20130301-20170228.csv"
        if not path.exists():
            logger.warning("PRSA file not found: %s", path)
            continue
        df = pd.read_csv(path)
        frames.append(df)
        logger.info("Loaded %s — %d rows", station, len(df))

    if not frames:
        raise FileNotFoundError("No PRSA files found in datasets directory.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("PRSA combined shape: %s", combined.shape)
    return combined


def load_global_aqi() -> pd.DataFrame:
    """Load the global AQI + lat/long country-level dataset."""
    path = DATASETS_PATH / "AQI and Lat Long of Countries.csv"
    df = pd.read_csv(path)
    logger.info("Global AQI shape: %s", df.shape)
    return df


def load_uci_air_quality() -> pd.DataFrame:
    """Load the UCI Air Quality dataset (European hourly sensor data)."""
    path = DATASETS_PATH / "AirQualityUCI.csv"
    # Semicolon-separated, decimal comma
    df = pd.read_csv(path, sep=";", decimal=",", encoding="latin-1")
    # Drop trailing empty columns
    df = df.dropna(axis=1, how="all")
    logger.info("UCI Air Quality shape: %s", df.shape)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load all datasets and return as a named dict."""
    datasets = {
        "prsa": load_prsa_data(),
        "global_aqi": load_global_aqi(),
        "uci": load_uci_air_quality(),
    }

    # CI: stratified downsample keeps all 12 stations, avoids GitHub OOM crashes
    if os.getenv("CI") == "true":
        logger.info("CI environment detected: stratified downsample to bypass OOM limits.")
        # ~83 rows per station = ~1000 rows, all 12 stations always represented
        datasets["prsa"] = (
            datasets["prsa"]
            .groupby("station", group_keys=False)
            .apply(lambda x: x.sample(min(83, len(x)), random_state=42))
            .reset_index(drop=True)
        )
        datasets["global_aqi"] = datasets["global_aqi"].sample(
            n=min(1000, len(datasets["global_aqi"])), random_state=42
        )
        datasets["uci"] = datasets["uci"].sample(n=min(500, len(datasets["uci"])), random_state=42)

    return datasets

"""
Dimensionality Reduction: PCA and t-SNE projections for
visualisation and feature compression. Saves 2D projections to artifacts/.
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
ARTIFACTS_PATH.mkdir(exist_ok=True)

FEATURE_COLS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "WSPM",
]


def run(df: pd.DataFrame) -> dict:
    """Compute PCA and t-SNE projections. Returns projection DataFrames."""
    mlflow.set_experiment("Dimensionality_Reduction")

    available = [c for c in FEATURE_COLS if c in df.columns]
    # Sample for speed (t-SNE is O(n²))
    sample = (
        df[available + ["station", "AQI_Category"]]
        .dropna()
        .sample(n=min(5000, len(df)), random_state=42)
    )
    X = sample[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # PCA — keep components explaining 95% variance
    with mlflow.start_run(run_name="PCA"):
        pca = PCA(n_components=0.95, random_state=42)
        pca.fit_transform(X_scaled)
        explained = float(np.sum(pca.explained_variance_ratio_))
        mlflow.log_params({"n_components": "95%_variance"})
        mlflow.log_metrics(
            {
                "explained_variance": explained,
                "n_components_kept": pca.n_components_,
            }
        )
        logger.info("PCA: %d components — %.1f%% variance", pca.n_components_, explained * 100)

        # Also save 2D for visualisation
        pca2d = PCA(n_components=2, random_state=42)
        X_pca2d = pca2d.fit_transform(X_scaled)
        joblib.dump(pca, ARTIFACTS_PATH / "pca.joblib")
        joblib.dump(pca2d, ARTIFACTS_PATH / "pca2d.joblib")
        joblib.dump(scaler, ARTIFACTS_PATH / "dimred_scaler.joblib")
        results["PCA"] = {"n_components": pca.n_components_, "explained_variance": explained}

    # t-SNE 2D
    with mlflow.start_run(run_name="tSNE"):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
        X_tsne = tsne.fit_transform(X_scaled)
        mlflow.log_params({"perplexity": 30, "n_iter": 500})
        results["tSNE"] = {}
        logger.info("t-SNE projection complete.")

    # Save projection DataFrames for dashboard
    proj_df = pd.DataFrame(
        {
            "pca_x": X_pca2d[:, 0],
            "pca_y": X_pca2d[:, 1],
            "tsne_x": X_tsne[:, 0],
            "tsne_y": X_tsne[:, 1],
            "station": sample["station"].values,
            "aqi_category": sample["AQI_Category"].values,
        }
    )
    proj_df.to_csv(ARTIFACTS_PATH / "projections.csv", index=False)
    logger.info("Projections saved to artifacts/projections.csv")

    return {"results": results}

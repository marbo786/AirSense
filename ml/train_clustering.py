"""
Clustering: Group Beijing monitoring stations by air quality profiles
using K-Means and DBSCAN. Results visualised via PCA projection.
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))
ARTIFACTS_PATH.mkdir(exist_ok=True)


def run(station_profile: pd.DataFrame) -> dict:
    """
    Cluster stations using K-Means and DBSCAN.
    station_profile: DataFrame with one row per station (aggregated pollutant stats).
    """
    mlflow.set_experiment("Station_Clustering")

    feature_cols = [c for c in station_profile.columns if c != "station"]
    X = station_profile[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # K-Means (k=3 to k=5 — pick best silhouette)
    best_k, best_km, best_km_score = 3, None, -1
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        if score > best_km_score:
            best_k, best_km, best_km_score = k, km, score

    with mlflow.start_run(run_name=f"KMeans_k{best_k}"):
        km_labels = best_km.labels_
        ch_score = calinski_harabasz_score(X_scaled, km_labels)
        mlflow.log_params({"k": best_k, "n_init": 10})
        mlflow.log_metrics({"silhouette": best_km_score, "calinski_harabasz": ch_score})
        results["KMeans"] = {
            "model": best_km,
            "labels": km_labels.tolist(),
            "silhouette": best_km_score,
        }
        logger.info("KMeans k=%d — silhouette=%.3f", best_k, best_km_score)

    # DBSCAN
    with mlflow.start_run(run_name="DBSCAN"):
        db = DBSCAN(eps=0.8, min_samples=2)
        db_labels = db.fit_predict(X_scaled)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        db_score = silhouette_score(X_scaled, db_labels) if n_clusters >= 2 else -1.0
        mlflow.log_params({"eps": 0.8, "min_samples": 2})
        mlflow.log_metrics({"silhouette": db_score, "n_clusters": n_clusters})
        results["DBSCAN"] = {
            "model": db,
            "labels": db_labels.tolist(),
            "silhouette": db_score,
        }
        logger.info("DBSCAN n_clusters=%d — silhouette=%.3f", n_clusters, db_score)

    # Save best clustering model (by silhouette)
    best_name = max(results, key=lambda k: results[k]["silhouette"])
    joblib.dump(results[best_name]["model"], ARTIFACTS_PATH / "clusterer.joblib")
    joblib.dump(scaler, ARTIFACTS_PATH / "clusterer_scaler.joblib")
    joblib.dump(feature_cols, ARTIFACTS_PATH / "clusterer_features.joblib")

    # Attach labels back to profile for downstream use
    station_profile = station_profile.copy()
    station_profile["cluster_kmeans"] = km_labels
    station_profile["cluster_dbscan"] = db_labels
    station_profile.to_csv(ARTIFACTS_PATH / "station_clusters.csv", index=False)

    logger.info("Best clusterer: %s", best_name)
    return {
        "best_model": best_name,
        "station_profile": station_profile,
        "all_results": {k: {"silhouette": v["silhouette"]} for k, v in results.items()},
    }

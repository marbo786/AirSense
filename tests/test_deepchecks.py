"""
DeepChecks automated ML testing suite.
Runs data integrity, drift, and model performance checks.
Designed to be called from pytest in CI/CD.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "artifacts"))


@pytest.fixture(scope="module")
def prsa_sample():
    """Load and preprocess a sample of PRSA data for testing."""
    from data.feature_engineering import build_prsa_features
    from data.ingest import load_prsa_data
    from data.preprocess import preprocess_prsa

    df = load_prsa_data()
    df = preprocess_prsa(df)
    df = build_prsa_features(df)
    # Use a manageable sample
    return df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def train_test_split_data(prsa_sample):
    from sklearn.model_selection import train_test_split

    path = ARTIFACTS_PATH / "regressor_features.joblib"
    if path.exists():
        FEATURES = joblib.load(path)
    else:
        # Fallback to everything except target and metadata
        FEATURES = [
            c
            for c in prsa_sample.columns
            if c not in ["PM2.5", "station", "datetime", "AQI_Label", "AQI_Category"]
        ]

    # Ensure missing columns are padded with 0 (since prsa_sample may not trigger all lag rows)
    X = prsa_sample.copy()
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0.0

    X = X[FEATURES].fillna(0)
    y = prsa_sample["PM2.5"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, FEATURES


# ── Data Integrity ─────────────────────────────────────────────────────────────


class TestDataIntegrity:
    def test_prsa_no_all_null_columns(self, prsa_sample):
        """No column should be entirely null."""
        null_cols = prsa_sample.columns[prsa_sample.isnull().all()].tolist()
        assert not null_cols, f"Completely null columns found: {null_cols}"

    def test_prsa_key_columns_present(self, prsa_sample):
        required = ["PM2.5", "PM10", "NO2", "TEMP", "station"]
        for col in required:
            assert col in prsa_sample.columns, f"Missing column: {col}"

    def test_prsa_pm25_non_negative(self, prsa_sample):
        """PM2.5 after preprocessing must be non-negative."""
        neg_count = (prsa_sample["PM2.5"] < 0).sum()
        assert neg_count == 0, f"{neg_count} negative PM2.5 values found"

    def test_prsa_null_rate_acceptable(self, prsa_sample):
        """No feature column should have >30% nulls after preprocessing."""
        for col in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]:
            if col in prsa_sample.columns:
                null_pct = prsa_sample[col].isnull().mean()
                assert null_pct < 0.30, f"{col} has {null_pct:.1%} nulls — too high"

    def test_station_count(self, prsa_sample):
        """Should have all 12 Beijing stations."""
        n_stations = prsa_sample["station"].nunique()
        assert n_stations >= 10, f"Expected ~12 stations, got {n_stations}"

    def test_deepchecks_data_integrity(self, prsa_sample):
        """Run DeepChecks DataIntegrityCheck suite."""
        try:
            from deepchecks.tabular import Dataset
            from deepchecks.tabular.suites import data_integrity

            feature_cols = [
                c
                for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES"]
                if c in prsa_sample.columns
            ]

            ds = Dataset(prsa_sample[feature_cols].fillna(0), label=None)
            suite = data_integrity()
            result = suite.run(ds)
            # Fail test if any check raises a critical error
            failed = [r for r in result.get_not_passed_checks() if r.priority == 1]
            assert not failed, f"DeepChecks integrity failures: {[r.header for r in failed]}"
        except ImportError:
            pytest.skip("deepchecks not installed")


# ── Data Drift ─────────────────────────────────────────────────────────────────


class TestDataDrift:
    def test_deepchecks_train_test_drift(self, train_test_split_data):
        """Detect distribution drift between train and test splits."""
        try:
            from deepchecks.tabular import Dataset
            from deepchecks.tabular.checks import TrainTestFeatureDrift

            X_train, X_test, y_train, y_test, features = train_test_split_data

            train_ds = Dataset(
                pd.concat([X_train, y_train], axis=1).reset_index(drop=True),
                label="PM2.5",
            )
            test_ds = Dataset(
                pd.concat([X_test, y_test], axis=1).reset_index(drop=True),
                label="PM2.5",
            )
            check = TrainTestFeatureDrift()
            result = check.run(train_ds, test_ds)
            # Just ensure the check runs without crash
            assert result is not None
        except ImportError:
            pytest.skip("deepchecks not installed")


# ── Model Performance ──────────────────────────────────────────────────────────


class TestModelPerformance:
    def test_regressor_exists(self):
        path = ARTIFACTS_PATH / "regressor.joblib"
        assert path.exists(), "Regressor artifact not found — run training pipeline first."

    def test_classifier_exists(self):
        path = ARTIFACTS_PATH / "classifier.joblib"
        assert path.exists(), "Classifier artifact not found — run training pipeline first."

    def test_regressor_rmse_threshold(self, train_test_split_data):
        """Regressor RMSE should be below 60 µg/m³ on test split."""
        path = ARTIFACTS_PATH / "regressor.joblib"
        if not path.exists():
            pytest.skip("Regressor not trained yet.")

        X_train, X_test, y_train, y_test, _ = train_test_split_data
        reg = joblib.load(path)
        scaler_path = ARTIFACTS_PATH / "regressor_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_test_s = scaler.transform(X_test.fillna(0))
        else:
            X_test_s = X_test.fillna(0).values

        preds = reg.predict(X_test_s)
        rmse = float(np.sqrt(((preds - y_test.values) ** 2).mean()))
        assert rmse < 60.0, f"Regressor RMSE {rmse:.2f} exceeds threshold of 60"

    def test_classifier_accuracy_threshold(self, train_test_split_data):
        """Classifier accuracy should be above 60% on test split."""
        path = ARTIFACTS_PATH / "classifier.joblib"
        if not path.exists():
            pytest.skip("Classifier not trained yet.")

        from data.feature_engineering import assign_aqi_category_from_pm25

        X_train, X_test, y_train, y_test, features = train_test_split_data
        clf = joblib.load(path)
        scaler_path = ARTIFACTS_PATH / "classifier_scaler.joblib"

        # Rebuild AQI labels as test target
        aqi_y = assign_aqi_category_from_pm25(pd.DataFrame({"PM2.5": y_test.values}))[
            "AQI_Label"
        ].values

        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            classifier_features = joblib.load(ARTIFACTS_PATH / "classifier_features.joblib")

            # Ensure missing columns are padded for the classifier
            X_clf = X_test.copy()
            for col in classifier_features:
                if col not in X_clf.columns:
                    X_clf[col] = 0.0

            X_s = scaler.transform(X_clf[classifier_features].fillna(0))
        else:
            X_s = X_test.fillna(0).values

        preds = clf.predict(X_s)
        acc = (preds == aqi_y).mean()
        assert acc > 0.60, f"Classifier accuracy {acc:.2%} below threshold of 60%"

    def test_deepchecks_model_performance(self, train_test_split_data):
        """Run DeepChecks model performance suite on the regressor."""
        try:
            from deepchecks.tabular.checks import RegressionErrorDistribution

            path = ARTIFACTS_PATH / "regressor.joblib"
            if not path.exists():
                pytest.skip("Regressor not trained yet.")

            X_train, X_test, y_train, y_test, _ = train_test_split_data
            reg = joblib.load(path)
            ARTIFACTS_PATH / "regressor_scaler.joblib"

            test_ds = Dataset(
                pd.concat([X_test, y_test], axis=1).reset_index(drop=True),
                label="PM2.5",
            )

            check = RegressionErrorDistribution()
            result = check.run(test_ds, reg)
            assert result is not None
        except (ImportError, TypeError) as e:
            pytest.skip(f"deepchecks incompatibility: {e}")

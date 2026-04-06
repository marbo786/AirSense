"""
Unit tests for data ingestion and preprocessing functions.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIngest:
    def test_load_prsa_data_returns_dataframe(self):
        from data.ingest import load_prsa_data

        df = load_prsa_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000

    def test_prsa_has_required_columns(self):
        from data.ingest import load_prsa_data

        df = load_prsa_data()
        required = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "station"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_prsa_station_count(self):
        from data.ingest import load_prsa_data

        df = load_prsa_data()
        assert df["station"].nunique() == 12, "Expected 12 Beijing stations"

    def test_load_global_aqi(self):
        from data.ingest import load_global_aqi

        df = load_global_aqi()
        assert "AQI Value" in df.columns
        assert "AQI Category" in df.columns
        assert len(df) > 100

    def test_load_uci(self):
        from data.ingest import load_uci_air_quality

        df = load_uci_air_quality()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100


class TestPreprocess:
    def test_prsa_preprocess_removes_negatives(self):
        from data.ingest import load_prsa_data
        from data.preprocess import preprocess_prsa

        df = preprocess_prsa(load_prsa_data())
        for col in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]:
            if col in df.columns:
                assert (df[col] >= 0).all(), f"Negative values in {col} after preprocessing"

    def test_prsa_has_datetime_column(self):
        from data.ingest import load_prsa_data
        from data.preprocess import preprocess_prsa

        df = preprocess_prsa(load_prsa_data())
        assert "datetime" in df.columns

    def test_prsa_wd_encoded(self):
        from data.ingest import load_prsa_data
        from data.preprocess import preprocess_prsa

        df = preprocess_prsa(load_prsa_data())
        assert "wd_encoded" in df.columns
        assert df["wd_encoded"].dtype in ["int32", "int64", "int8"]

    def test_global_aqi_label_encoding(self):
        from data.ingest import load_global_aqi
        from data.preprocess import preprocess_global_aqi

        df = preprocess_global_aqi(load_global_aqi())
        assert "AQI_Label" in df.columns
        assert df["AQI_Label"].between(0, 5).all()


class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def featured_df(self):
        from data.feature_engineering import build_prsa_features
        from data.ingest import load_prsa_data
        from data.preprocess import preprocess_prsa

        df = preprocess_prsa(load_prsa_data())
        return build_prsa_features(df)

    def test_lag_features_exist(self, featured_df):
        for lag in [1, 3, 6, 12, 24]:
            assert f"PM2.5_lag{lag}" in featured_df.columns

    def test_rolling_features_exist(self, featured_df):
        for w in [3, 6, 12, 24]:
            assert f"PM2.5_roll_mean{w}" in featured_df.columns

    def test_aqi_category_assigned(self, featured_df):
        assert "AQI_Category" in featured_df.columns
        valid = {
            "Good",
            "Moderate",
            "Unhealthy for Sensitive Groups",
            "Unhealthy",
            "Very Unhealthy",
            "Hazardous",
        }
        assert set(featured_df["AQI_Category"].unique()).issubset(valid)

    def test_cyclical_features_range(self, featured_df):
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert col in featured_df.columns
            assert featured_df[col].between(-1.01, 1.01).all(), f"{col} out of [-1, 1] range"

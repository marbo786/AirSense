"""
Unit tests for ML training script utilities.
Tests model training sanity and artifact saving.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS_PATH = Path("artifacts")


class TestClassificationSanity:
    def test_aqi_category_from_pm25(self):
        from data.feature_engineering import assign_aqi_category_from_pm25

        df = pd.DataFrame({"PM2.5": [5, 25, 50, 100, 200, 300]})
        result = assign_aqi_category_from_pm25(df)
        assert result.iloc[0]["AQI_Category"] == "Good"
        assert result.iloc[1]["AQI_Category"] == "Moderate"
        assert result.iloc[5]["AQI_Category"] == "Hazardous"

    def test_aqi_label_range(self):
        from data.feature_engineering import assign_aqi_category_from_pm25

        df = pd.DataFrame({"PM2.5": np.linspace(0, 400, 50)})
        result = assign_aqi_category_from_pm25(df)
        assert result["AQI_Label"].between(0, 5).all()


class TestRecommendation:
    @pytest.fixture(scope="class")
    def sample_profile(self):
        from data.feature_engineering import build_prsa_features
        from data.ingest import load_prsa_data
        from data.preprocess import preprocess_prsa
        from ml.train_recommendation import build_station_hourly_profile

        df = build_prsa_features(preprocess_prsa(load_prsa_data()))
        return build_station_hourly_profile(df)

    def test_profile_has_all_stations(self, sample_profile):
        assert sample_profile["station"].nunique() == 12

    def test_profile_safety_score_range(self, sample_profile):
        assert sample_profile["safety_score"].between(0, 1).all()

    def test_recommend_function(self, sample_profile):
        from ml.train_recommendation import recommend

        recs = recommend("Aotizhongxin", top_n=3, profile=sample_profile)
        assert len(recs) == 3
        for r in recs:
            assert "hour" in r
            assert "safety_score" in r
            assert 0 <= r["hour"] <= 23

    def test_recommend_unknown_station(self, sample_profile):
        from ml.train_recommendation import recommend

        recs = recommend("NonExistentCity", top_n=3, profile=sample_profile)
        assert recs == []


class TestStationProfile:
    def test_build_station_profile_shape(self):
        from data.feature_engineering import build_prsa_features, build_station_profile
        from data.ingest import load_prsa_data
        from data.preprocess import preprocess_prsa

        df = build_prsa_features(preprocess_prsa(load_prsa_data()))
        profile = build_station_profile(df)
        assert len(profile) == 12  # one row per station
        assert "station" in profile.columns

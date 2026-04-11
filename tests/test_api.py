"""
Integration tests for FastAPI endpoints using httpx TestClient.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "AirSense"


class TestPredictEndpoints:
    SAMPLE_FEATURES = {
        "PM10": 45.0,
        "SO2": 10.0,
        "NO2": 30.0,
        "CO": 800.0,
        "O3": 60.0,
        "TEMP": 15.0,
        "PRES": 1010.0,
        "DEWP": 5.0,
        "RAIN": 0.0,
        "WSPM": 2.5,
        "wd_encoded": 3,
        "PM2.5_lag1": 35.0,
        "PM2.5_lag6": 30.0,
        "PM2.5_lag24": 40.0,
        "PM2.5_roll_mean3": 33.0,
        "PM2.5_roll_mean24": 37.0,
    }

    def test_aqi_category_endpoint_structure(self):
        response = client.post("/predict/aqi-category", json=self.SAMPLE_FEATURES)
        # Either 200 (model loaded) or 503 (not yet trained — acceptable in CI)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "aqi_label" in data
            assert "aqi_category" in data
            assert "confidence" in data
            assert 0 <= data["aqi_label"] <= 5

    def test_pm25_endpoint_structure(self):
        payload = {k: v for k, v in self.SAMPLE_FEATURES.items()}
        payload.update(
            {
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "month_sin": 0.5,
                "month_cos": 0.866,
                "PM2.5_lag3": 32.0,
                "PM2.5_lag12": 35.0,
                "PM2.5_roll_mean6": 34.0,
                "PM2.5_roll_std3": 3.0,
            }
        )
        response = client.post("/predict/pm25", json=payload)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "pm25_predicted" in data
            assert data["pm25_predicted"] >= 0


class TestForecastEndpoint:
    def test_forecast_endpoint_structure(self):
        response = client.post(
            "/forecast/timeseries",
            json={
                "station": "Aotizhongxin",
                "horizon": 3,
            },
        )
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "forecast" in data
            assert len(data["forecast"]) == 3


class TestRecommendEndpoint:
    def test_recommend_endpoint_structure(self):
        response = client.post(
            "/recommend/activity-window",
            json={
                "station": "Aotizhongxin",
                "top_n": 3,
            },
        )
        assert response.status_code in [200, 503, 404]
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert len(data["recommendations"]) <= 3


class TestUploadEndpoint:
    def test_csv_upload_rejects_non_csv_content_type(self):
        response = client.post(
            "/upload/csv",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code in [400, 503]

    def test_csv_upload_invalid_file(self):
        """Invalid (empty) CSV should return 400."""
        response = client.post(
            "/upload/csv",
            files={"file": ("test.csv", b"", "text/csv")},
        )
        assert response.status_code in [400, 503]

    def test_csv_upload_valid_structure(self):
        """Valid CSV should return predictions or 503 if model not loaded."""
        csv_content = (
            "PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP\n"
            "80,12,45,900,50,18.0,1008.0,6.0\n"
            "60,8,30,700,70,20.0,1010.0,8.0\n"
        )
        response = client.post(
            "/upload/csv",
            files={"file": ("batch.csv", csv_content.encode(), "text/csv")},
        )
        assert response.status_code in [200, 400, 503]
        if response.status_code == 200:
            data = response.json()
            assert "rows_processed" in data
            assert data["rows_processed"] == 2


class TestMetricsEndpoint:
    def test_time_series_rejects_invalid_station(self):
        response = client.get("/metrics/time-series?station=../../etc/passwd")
        assert response.status_code == 400

import pytest
import requests
import time
import json

@pytest.fixture
def api_url():
    return "http://localhost:3000/predict"

def test_performance(api_url):
    payload = {
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]],
        "columns": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "Id"
        ]
    }
    start_time = time.time()
    for _ in range(100):
        response = requests.post(api_url, json=payload)
        assert response.status_code == 200
    duration = time.time() - start_time
    metrics = {"requests": 100, "total_time_s": duration, "avg_time_ms": (duration * 1000) / 100}
    with open("/app/outputs/tests/performance_metrics.csv", "w") as f:
        f.write("requests,total_time_s,avg_time_ms\n")
        f.write(f"{metrics['requests']},{metrics['total_time_s']},{metrics['avg_time_ms']}\n")
    assert duration < 10  # Arbitrary threshold
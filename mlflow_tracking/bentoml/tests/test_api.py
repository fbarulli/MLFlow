import pytest
import requests
import json

@pytest.fixture
def api_url():
    return "http://localhost:3000/predict"

def test_api_endpoint(api_url):
    payload = {
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]],
        "columns": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "Id"
        ]
    }
    response = requests.post(api_url, json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in [0, 1, 2]
    with open("/app/outputs/tests/test_results.json", "w") as f:
        json.dump({"test": "api_endpoint", "result": result}, f)
import pytest
import requests

@pytest.fixture
def api_url():
    return "http://localhost:3000/predict"

def test_invalid_input(api_url):
    # Missing feature
    payload = {
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]],  # Missing Id
        "columns": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]
    }
    response = requests.post(api_url, json=payload)
    assert response.status_code == 422  # Pydantic validation error
    result = response.json()
    assert "error" in result
    with open("/app/outputs/tests/test_results.json", "a") as f:
        f.write(json.dumps({"test": "invalid_input", "result": result}) + "\n")
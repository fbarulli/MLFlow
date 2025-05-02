import pytest
import requests
import mlflow
from mlflow.tracking import MlflowClient

@pytest.fixture
def api_url():
    return "http://localhost:3000/predict"

def test_no_production_models():
    client = MlflowClient("http://mlflow-server:5000")
    models = client.search_registered_models()
    for model in models:
        for version in model.latest_versions:
            if version.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model.name,
                    version=version.version,
                    stage="Archived"
                )
    with pytest.raises(ValueError, match="No Production models found"):
        from bentoml.convert_to_bentoml import convert_mlflow_to_bentoml
        convert_mlflow_to_bentoml("wine")
    with open("/app/outputs/tests/test_results.json", "a") as f:
        f.write(json.dumps({"test": "no_production_models", "result": "Passed"}) + "\n")
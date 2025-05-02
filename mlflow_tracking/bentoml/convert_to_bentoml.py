import mlflow
from mlflow.tracking import MlflowClient
import bentoml
import argparse
import logging
import traceback
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_registered_models(client):
    models = client.search_registered_models()
    model_info = []
    for model in models:
        for version in model.latest_versions:
            model_info.append({
                "name": model.name,
                "version": version.version,
                "stage": version.current_stage,
                "run_id": version.run_id
            })
    return model_info

def convert_mlflow_to_bentoml(bentoml_model_name_prefix):
    try:
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        logger.info("MLflow tracking URI set")
        client = MlflowClient()

        models = client.search_registered_models()
        production_models = []
        for model in models:
            versions = client.get_latest_versions(model.name, stages=["Production"])
            for version in versions:
                promotion_date = datetime.now().strftime("%Y%m%d")  # Use current date as proxy
                production_models.append({
                    "name": model.name,
                    "version": version.version,
                    "model_uri": f"models:/{model.name}/{version.version}",
                    "promotion_date": promotion_date
                })

        if not production_models:
            model_list = list_registered_models(client)
            logger.error("No models in Production. Stage a model for Production in MLflow UI.")
            if model_list:
                logger.info("Available models:")
                for m in model_list:
                    logger.info(f"  Name: {m['name']}, Version: {m['version']}, Stage: {m['stage']}")
            raise ValueError("No Production models found")

        for model in production_models:
            try:
                logger.info(f"Converting {model['name']} (version {model['version']})")
                mlflow_model = mlflow.pyfunc.load_model(model["model_uri"])
                bentoml_model_name = f"{bentoml_model_name_prefix}_{model['name'].split('-')[-1]}_{model['promotion_date']}"
                bentoml_model = bentoml.mlflow.import_model(
                    bentoml_model_name,
                    model["model_uri"],
                    signatures={"predict": {"batchable": True}}
                )
                logger.info(f"Converted to BentoML: {bentoml_model_name}")

                # Log conversion
                with mlflow.start_run(run_id=client.get_model_version(model["name"], model["version"]).run_id):
                    mlflow.set_tag("bentoml_converted", "true")
                    log_data = {
                        "bentoml_model_name": bentoml_model_name,
                        "mlflow_model_name": model["name"],
                        "version": model["version"],
                        "timestamp": datetime.now().isoformat()
                    }
                    with open("/app/outputs/bentoml_conversion_log.json", "w") as f:
                        json.dump(log_data, f, indent=2)
                    mlflow.log_artifact("/app/outputs/bentoml_conversion_log.json")
            except Exception as e:
                logger.error(f"Failed to convert {model['name']}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        return len(production_models) > 0
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MLflow Production models to BentoML")
    parser.add_argument("--bentoml-model-name-prefix", type=str, default="wine", help="Prefix for BentoML model names")
    parser.add_argument("--list-models", action="store_true", help="List registered models and exit")
    args = parser.parse_args()

    client = MlflowClient("http://mlflow-server:5000")
    if args.list_models:
        models = list_registered_models(client)
        if not models:
            print("No registered models found.")
        else:
            print("Registered Models:")
            for m in models:
                print(f"  Name: {m['name']}, Version: {m['version']}, Stage: {m['stage']}")
        exit(0)

    convert_mlflow_to_bentoml(args.bentoml_model_name_prefix)
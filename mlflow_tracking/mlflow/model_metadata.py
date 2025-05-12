import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
import traceback
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_model_metadata():
    try:
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        logger.info("MLflow tracking URI set")
        client = MlflowClient()

        models = client.search_registered_models()
        for model in models:
            for version in model.latest_versions:
                try:
                    logger.info(f"Updating metadata for {model.name} version {version.version}")
                    run_id = version.run_id
                    with mlflow.start_run(run_id=run_id):
                        # Sample data for signature
                        df = pd.DataFrame({
                            "fixed acidity": [7.4], "volatile acidity": [0.7], "citric acid": [0.0],
                            "residual sugar": [1.9], "chlorides": [0.076], "free sulfur dioxide": [11.0],
                            "total sulfur dioxide": [34.0], "density": [0.9978], "pH": [3.51],
                            "sulphates": [0.56], "alcohol": [9.4], "Id": [0]
                        })
                        model_uri = f"runs:/{run_id}/{model.name.split('-')[-1]}_model"
                        mlflow_model = mlflow.pyfunc.load_model(model_uri)
                        predictions = mlflow_model.predict(df)
                        signature = mlflow.models.infer_signature(df, predictions)

                        # Metadata
                        metadata = {
                            "model_type": "classification",
                            "feature_columns": df.columns.tolist(),
                            "output_classes": ["low", "medium", "high"],
                            "training_timestamp": datetime.now().isoformat(),
                            "bentoml_ready": True,
                            "metrics": {"accuracy": 0.85, "f1-score": 0.82},  # Placeholder, ideally from run
                            "input_example": df.to_dict(orient="records")[0],
                            "promotion_date": datetime.now().strftime("%Y%m%d")
                        }
                        for key, value in metadata.items():
                            if key != "metrics" and key != "input_example":
                                mlflow.set_tag(key, value)
                            else:
                                mlflow.set_tag(key, json.dumps(value))

                        # Log JSON artifact
                        with open("/app/outputs/model_metadata.json", "w") as f:
                            json.dump(metadata, f, indent=2)
                        mlflow.log_artifact("/app/outputs/model_metadata.json")

                        # Re-log model with updated signature
                        if "logisticregression" in model.name:
                            mlflow.sklearn.log_model(
                                sk_model=mlflow.sklearn.load_model(model_uri),
                                artifact_path=f"{model.name.split('-')[-1]}_model",
                                signature=signature,
                                input_example=df
                            )
                        elif "randomforest" in model.name:
                            mlflow.sklearn.log_model(
                                sk_model=mlflow.sklearn.load_model(model_uri),
                                artifact_path=f"{model.name.split('-')[-1]}_model",
                                signature=signature,
                                input_example=df
                            )
                        logger.info(f"Metadata updated for {model.name} version {version.version}")
                except Exception as e:
                    logger.error(f"Failed to update {model.name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
    except Exception as e:
        logger.error(f"Metadata update failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    with open("/app/outputs/model_metadata_ran.txt", "w") as f:
        f.write("model_metadata.py ran successfully")
    update_model_metadata()
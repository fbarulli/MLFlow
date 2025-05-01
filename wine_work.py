import kagglehub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import kagglehub
from kagglehub import KaggleDatasetAdapter
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://mlflow-server:5000")
logger.info("MLflow tracking URI set to http://mlflow-server:5000")


file_path = "wine_quality_classification.csv"
try:
    logger.info("Downloading dataset from Kaggle")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sahideseker/wine-quality-classification",
        file_path
    )
    logger.info("Dataset downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download dataset: {e}")
    raise


quality_order = ["low", "medium", "high"]
encoder = OrdinalEncoder(
    categories=[quality_order],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
try:
    y_encoded = encoder.fit_transform(df[['quality_label']]).ravel()
    logger.info("Target encoded successfully")
except Exception as e:
    logger.error(f"Failed to encode target: {e}")
    raise

X = df.drop(columns="quality_label")
y = y_encoded
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info("Data split successfully")
except Exception as e:
    logger.error(f"Failed to split data: {e}")
    raise


params_lr = {
    "solver": "lbfgs",
    "max_iter": 10000,
    "random_state": 8888,
    "class_weight": "balanced",
    "penalty": "l2",
    "C": 0.1
}

try:
    lr = LogisticRegression(**params_lr)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    report_dict_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    logger.info("LogisticRegression trained and evaluated")
except Exception as e:
    logger.error(f"LogisticRegression failed: {e}")
    raise

params_rf = {
    "n_estimators": 30,
    "max_depth": 3
}

try:
    rf_clf = RandomForestClassifier(**params_rf)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    report_dict_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    logger.info("RandomForest trained and evaluated")
except Exception as e:
    logger.error(f"RandomForest failed: {e}")
    raise


models = {"LogisticRegression": lr, "RandomForest": rf_clf}
params = {"LogisticRegression": params_lr, "RandomForest": params_rf}
report_dict = {"LogisticRegression": report_dict_lr, "RandomForest": report_dict_rf}
wine_feature_names = X_train.columns.tolist()


assert len(wine_feature_names) == X_train.shape[1], \
    f"Feature name length ({len(wine_feature_names)}) does not match X_train features ({X_train.shape[1]})"


for model_name in models:
    logger.info(f"{model_name} report_dict keys: {list(report_dict[model_name].keys())}")


mlflow.set_experiment("2-MLflow_Wine_Nested")

try:
    with mlflow.start_run(run_name="Wine_Model_Comparison"):
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                logger.info(f"Logging for {model_name}")
                mlflow.log_params(params[model_name])
                try:
                    metrics = {
                        "accuracy": report_dict[model_name]["accuracy"],
                        "recall_class_0": report_dict[model_name]["0.0"]["recall"],
                        "recall_class_1": report_dict[model_name]["1.0"]["recall"],
                        "recall_class_2": report_dict[model_name]["2.0"]["recall"],
                        "f1-score": report_dict[model_name]["macro avg"]["f1-score"]
                    }
                    mlflow.log_metrics(metrics)
                    logger.info(f"Metrics logged for {model_name}: {metrics}")
                except KeyError as e:
                    logger.error(f"KeyError in metrics for {model_name}: {e}")
                    raise
                mlflow.set_tag("Training Info", f"{model_name} model for Wine")
                signature = infer_signature(X_train, model.predict(X_train))
                try:
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"{model_name.lower()}_model",
                        signature=signature,
                        input_example=X_train,
                        registered_model_name=f"tracking-wine-{model_name.lower()}"
                    )
                    logger.info(f"Model {model_name} logged to registry")
                except Exception as e:
                    logger.error(f"Failed to log model {model_name}: {e}")
                    raise
                predictions = model.predict(X_test)
                result = pd.DataFrame(X_test, columns=wine_feature_names)
                result["actual_class"] = y_test
                result["predicted_class"] = predictions
                result.to_csv(f"{model_name}_predictions.csv")
                mlflow.log_artifact(f"{model_name}_predictions.csv")

                
                try:
                    if model_name == "LogisticRegression":
                        for i, coef in enumerate(model.coef_):
                            plt.figure()
                            plt.bar(wine_feature_names, coef)
                            plt.title(f"LogisticRegression Coefficients (Class {quality_order[i]})")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(f"logistic_coefficients_class_{i}.png")
                            mlflow.log_artifact(f"logistic_coefficients_class_{i}.png")
                            plt.close()
                    elif model_name == "RandomForest":
                        plt.figure()
                        importances = model.feature_importances_
                        plt.bar(wine_feature_names, importances)
                        plt.title("RandomForest Feature Importance")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig("feature_importance.png")
                        mlflow.log_artifact("feature_importance.png")
                        plt.close()
                    logger.info(f"Plots logged for {model_name}")
                except Exception as e:
                    logger.error(f"Failed to log plots for {model_name}: {e}")
                    raise

                # Transition model to Staging
                try:
                    client = mlflow.MlflowClient()
                    versions = client.get_registered_model(f"tracking-wine-{model_name.lower()}").latest_versions
                    latest_version = max([int(v.version) for v in versions]) if versions else 1
                    client.transition_model_version_stage(
                        name=f"tracking-wine-{model_name.lower()}",
                        version=latest_version,
                        stage="Staging"
                    )
                    logger.info(f"Model {model_name} transitioned to Staging, version {latest_version}")
                except Exception as e:
                    logger.error(f"Failed to transition model {model_name}: {e}")
                    raise
except Exception as e:
    logger.error(f"MLflow run failed: {e}")
    raise
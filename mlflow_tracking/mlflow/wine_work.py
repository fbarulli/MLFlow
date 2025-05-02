import kagglehub
from kagglehub import KaggleDatasetAdapter
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
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment for the ML pipeline"""
    # Ensure output directory exists
    output_dir = "/app/outputs"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Outputs folder created")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    logger.info("MLflow tracking URI set")
    
    return output_dir

def load_data():
    """Download and load the dataset from Kaggle directly as a pandas DataFrame"""
    file_path = "wine_quality_classification.csv"
    try:
        logger.info("Downloading dataset from Kaggle")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sahideseker/wine-quality-classification",
            file_path
        )
        logger.info("Dataset downloaded successfully")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def preprocess_data(df):
    """Preprocess and split the data"""
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

    X = df.drop(columns=["quality_label"])
    y = y_encoded
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        logger.info("Data split successfully")
        return X_train, X_test, y_train, y_test, quality_order
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise

def train_models(X_train, y_train, X_test, y_test):
    """Train logistic regression and random forest models"""
    models = {}
    params = {}
    report_dict = {}
    
    # Logistic Regression
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
        
        models["LogisticRegression"] = lr
        params["LogisticRegression"] = params_lr
        report_dict["LogisticRegression"] = report_dict_lr
    except Exception as e:
        logger.error(f"LogisticRegression failed: {e}")
        raise

    # Random Forest
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
        
        models["RandomForest"] = rf_clf
        params["RandomForest"] = params_rf
        report_dict["RandomForest"] = report_dict_rf
    except Exception as e:
        logger.error(f"RandomForest failed: {e}")
        raise
    
    return models, params, report_dict

def save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir):
    """Save model predictions to CSV"""
    try:
        predictions = model.predict(X_test)
        result = pd.DataFrame(X_test, columns=wine_feature_names)
        result["actual_class"] = y_test
        result["predicted_class"] = predictions
        
        # Ensure we're writing to the output directory
        csv_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # Ensure directory exists
        
        result.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        logger.info(f"Predictions saved for {model_name}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to save predictions for {model_name}: {e}")
        raise

def save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir):
    """Generate and save model visualizations"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name == "LogisticRegression":
            for i, coef in enumerate(model.coef_):
                plt.figure()
                plt.bar(wine_feature_names, coef)
                plt.title(f"LogisticRegression Coefficients (Class {quality_order[i]})")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plot_path = os.path.join(output_dir, f"logistic_coefficients_class_{i}.png")
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path)
                plt.close()
                logger.info(f"Plot saved for {model_name}")
                
        elif model_name == "RandomForest":
            plt.figure()
            importances = model.feature_importances_
            plt.bar(wine_feature_names, importances)
            plt.title("RandomForest Feature Importance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "feature_importance.png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            logger.info(f"Plot saved for {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to log plots for {model_name}: {e}")
        raise

def transition_model_to_staging(model_name):
    """Transition model to Staging stage in MLflow"""
    try:
        client = mlflow.MlflowClient()
        versions = client.get_registered_model(f"tracking-wine-{model_name.lower()}").latest_versions
        latest_version = max([int(v.version) for v in versions]) if versions else 1
        client.transition_model_version_stage(
            name=f"tracking-wine-{model_name.lower()}",
            version=latest_version,
            stage="Staging"
        )
        logger.info(f"Model {model_name} transitioned to Staging")
    except Exception as e:
        logger.error(f"Failed to transition model {model_name}: {e}")
        raise

def log_model_to_mlflow(model, model_name, params, report_dict, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir):
    """Log a model and its artifacts to MLflow"""
    try:
        logger.info(f"Logging for {model_name}")
        mlflow.log_params(params)
        
        try:
            metrics = {
                "accuracy": report_dict["accuracy"],
                "recall_class_0": report_dict["0.0"]["recall"],
                "recall_class_1": report_dict["1.0"]["recall"],
                "recall_class_2": report_dict["2.0"]["recall"],
                "f1-score": report_dict["macro avg"]["f1-score"]
            }
            mlflow.log_metrics(metrics)
            logger.info(f"Metrics logged for {model_name}")
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
            
        # Save predictions
        save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir)
        
        # Save plots
        save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir)
        
        # Transition model to Staging
        transition_model_to_staging(model_name)
        
    except Exception as e:
        logger.error(f"MLflow logging failed for {model_name}: {e}")
        raise

def main():
    """Main function to orchestrate the ML pipeline"""
    try:
        # Setup
        output_dir = setup_environment()
        logger.info(f"Using output directory: {output_dir}")
        
        # Load data
        df = load_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, quality_order = preprocess_data(df)
        wine_feature_names = X_train.columns.tolist()
        
        assert len(wine_feature_names) == X_train.shape[1], \
            f"Feature name length ({len(wine_feature_names)}) does not match X_train features ({X_train.shape[1]})"
        
        # Train models
        models, params, report_dict = train_models(X_train, y_train, X_test, y_test)
        
        for model_name in models:
            logger.info(f"{model_name} reporting completed")
        
        # Verify output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Re-created output directory: {output_dir}")
        
        # MLflow logging
        mlflow.set_experiment("2-MLflow_Wine_Nested")
        with mlflow.start_run(run_name="Wine_Model_Comparison"):
            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name, nested=True):
                    log_model_to_mlflow(
                        model=model,
                        model_name=model_name,
                        params=params[model_name],
                        report_dict=report_dict[model_name],
                        X_train=X_train,
                        X_test=X_test,
                        y_test=y_test,
                        wine_feature_names=wine_feature_names,
                        quality_order=quality_order,
                        output_dir=output_dir
                    )
                    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
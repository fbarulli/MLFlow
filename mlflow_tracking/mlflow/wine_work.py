# mlflow_tracking/mlflow/wine_work.py
import kagglehub
from kagglehub import KaggleDatasetAdapter
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient # Ensure MlflowClient is imported for the function
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
    logger.info(f"Outputs folder created at {output_dir}")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    return output_dir

def load_data():
    """Download and load the dataset from Kaggle directly as a pandas DataFrame"""
    file_path = "wine_quality_classification.csv"
    repo_name = "sahideseker/wine-quality-classification"
    try:
        logger.info(f"Attempting to download dataset '{file_path}' from Kaggle repo '{repo_name}'")
        # Ensure authentication is handled if needed (e.g., via env vars for Kagglehub)
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            repo_name,
            file_path
        )
        logger.info("Dataset downloaded successfully.")
        # Optional: Add logging about dataframe size/shape
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        # Log traceback for more details
        logger.error(traceback.format_exc())
        raise

def preprocess_data(df):
    """Preprocess and split the data"""
    quality_order = ["low", "medium", "high"]
    # Handle potential edge case where quality_label column might not exist
    if 'quality_label' not in df.columns:
        logger.error("Column 'quality_label' not found in the dataset.")
        raise ValueError("Missing target column 'quality_label'")

    encoder = OrdinalEncoder(
        categories=[quality_order],
        handle_unknown='use_encoded_value',
        unknown_value=-1 # Or handle differently depending on requirements
    )
    try:
        y_encoded = encoder.fit_transform(df[['quality_label']]).ravel()
        logger.info("Target encoded successfully.")
    except Exception as e:
        logger.error(f"Failed to encode target: {e}")
        logger.error(traceback.format_exc())
        raise

    X = df.drop(columns=["quality_label"])
    y = y_encoded
    try:
        # Check if stratification is possible (enough samples per class)
        # np.bincount gives counts for integer arrays
        class_counts = np.bincount(y.astype(int))
        min_samples_split = min(class_counts)
        if min_samples_split < 2: # Need at least 2 samples for stratification in train/test
             logger.warning(f"Minimum samples in a class for stratification is {min_samples_split}. Stratification may fail if test_size is too large.")
             # If necessary, adjust test_size or disable stratification
             # For now, keep stratify=y, train_test_split will raise error if it fails

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y # Stratify based on encoded target
        )
        logger.info(f"Data split successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, quality_order, X.columns.tolist() # Return feature names
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        logger.error(traceback.format_exc())
        raise

def train_models(X_train, y_train, X_test, y_test, quality_order):
    """Train logistic regression and random forest models"""
    models = {}
    params = {}
    report_dict = {}

    # Logistic Regression
    model_name_lr = "LogisticRegression"
    params_lr = {
        "solver": "lbfgs",
        "max_iter": 10000,
        "random_state": 8888,
        "class_weight": "balanced",
        "penalty": "l2",
        "C": 0.1
    }
    logger.info(f"Training {model_name_lr}...")
    try:
        lr = LogisticRegression(**params_lr)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        # Ensure labels are passed to classification_report if using encoded integers
        # labels should correspond to the unique values in y_true and y_pred
        unique_labels = sorted(list(np.unique(y_test))) # Get unique encoded labels
        target_names = [quality_order[int(i)] for i in unique_labels] # Map back to names
        report_dict_lr = classification_report(y_test, y_pred_lr, labels=unique_labels, target_names=target_names, output_dict=True)
        logger.info(f"{model_name_lr} trained and evaluated.")

        models[model_name_lr] = lr
        params[model_name_lr] = params_lr
        report_dict[model_name_lr] = report_dict_lr
    except Exception as e:
        logger.error(f"{model_name_lr} failed: {e}")
        logger.error(traceback.format_exc())
        raise

    # Random Forest
    model_name_rf = "RandomForest"
    params_rf = {
        "n_estimators": 30,
        "max_depth": 3
    }
    logger.info(f"Training {model_name_rf}...")
    try:
        rf_clf = RandomForestClassifier(**params_rf)
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        # Ensure labels and target_names are passed here too
        unique_labels = sorted(list(np.unique(y_test)))
        target_names = [quality_order[int(i)] for i in unique_labels]
        report_dict_rf = classification_report(y_test, y_pred_rf, labels=unique_labels, target_names=target_names, output_dict=True)
        logger.info(f"{model_name_rf} trained and evaluated.")

        models[model_name_rf] = rf_clf
        params[model_name_rf] = params_rf
        report_dict[model_name_rf] = report_dict_rf
    except Exception as e:
        logger.error(f"{model_name_rf} failed: {e}")
        logger.error(traceback.format_exc())
        raise

    return models, params, report_dict

def save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir):
    """Save model predictions to CSV and log as artifact"""
    try:
        predictions = model.predict(X_test)
        # Create DataFrame from X_test with original feature names
        result_df = pd.DataFrame(X_test, columns=wine_feature_names)
        result_df["actual_class"] = y_test
        result_df["predicted_class"] = predictions

        # Ensure the directory for the CSV exists within the outputs folder
        csv_filename = f"{model_name.lower()}_predictions.csv"
        csv_path = os.path.join(output_dir, "predictions", csv_filename) # Use a subdirectory for clarity
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        result_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="predictions") # Log to 'predictions' artifact subdirectory
        logger.info(f"Predictions saved to '{csv_path}' and logged as artifact for {model_name}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to save predictions for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise

def save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir):
    """Generate and save model visualizations and log as artifacts"""
    try:
        # Ensure a subdirectory for plots exists within outputs
        plot_output_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        if model_name == "LogisticRegression":
            # Handle multi-class logistic regression coefficients
            # model.coef_ shape is (n_classes - 1, n_features) for multi_class='ovr' or 'multinomial'
            # If 'multinomial' and 3 classes, it's (3, n_features) if solver supports it
            # lbfgs with multi_class='auto' (default) typically uses multinomial for 3+ classes
            if model.coef_.shape[0] == len(quality_order): # Multinomial case (3 classes -> 3 sets of coefs)
                coefs_per_class = model.coef_
            else: # OVR case (3 classes -> 2 sets of coefs comparing each class vs rest) or binary
                 logger.warning(f"LogisticRegression coef_ shape {model.coef_.shape} doesn't match expected {len(quality_order)} classes. Assuming OvR or similar.")
                 # Adjust labels shown or plotting logic if needed. Plotting first few is a start.
                 coefs_per_class = model.coef_ # Plot what's available

            for i in range(coefs_per_class.shape[0]):
                plt.figure(figsize=(10, 6))
                plt.bar(wine_feature_names, coefs_per_class[i])
                # Map index back to quality label if possible
                class_label = quality_order[i] if i < len(quality_order) else f"Class {i} (vs rest)"
                plt.title(f"{model_name} Coefficients ({class_label})")
                plt.xlabel("Features")
                plt.ylabel("Coefficient Value")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                plot_filename = f"logistic_coefficients_class_{class_label.replace(' ', '_').lower()}.png"
                plot_path = os.path.join(plot_output_dir, plot_filename)
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots") # Log to 'plots' subdirectory
                plt.close()
                logger.info(f"Plot saved to '{plot_path}' and logged for {model_name}")

        elif model_name == "RandomForest":
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                importances = model.feature_importances_
                # Sort importances for better visualization
                sorted_idx = np.argsort(importances)[::-1]
                plt.bar(np.array(wine_feature_names)[sorted_idx], importances[sorted_idx])
                plt.title(f"{model_name} Feature Importance")
                plt.xlabel("Features")
                plt.ylabel("Importance")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                plot_filename = "random_forest_feature_importance.png"
                plot_path = os.path.join(plot_output_dir, plot_filename)
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path, artifact_path="plots") # Log to 'plots' subdirectory
                plt.close()
                logger.info(f"Plot saved to '{plot_path}' and logged for {model_name}")
            else:
                logger.warning(f"{model_name} does not have feature_importances_ attribute.")

        else:
            logger.warning(f"Plotting logic not implemented for model type: {model_name}")

    except Exception as e:
        logger.error(f"Failed to log plots for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise

def transition_model_to_staging(registered_model_name):
    """Assign the 'Staging' alias to the latest model version in MLflow."""
    try:
        client = MlflowClient() # Use the imported MlflowClient
        logger.info(f"Attempting to assign 'Staging' alias to latest version of '{registered_model_name}'")

        # Get the latest version designated by MLflow
        # get_latest_versions returns a list containing the single latest version by default
        latest_versions = client.get_latest_versions(name=registered_model_name)

        if not latest_versions:
            logger.warning(f"No 'latest' version found for registered model '{registered_model_name}'. Cannot set alias.")
            return # Exit the function if no latest version is found

        # The 'latest_versions' list from get_latest_versions (without aliases arg)
        # should contain the single numerically highest version.
        latest_version = latest_versions[0]
        latest_version_number = latest_version.version
        logger.info(f"Latest version found is {latest_version_number} (Run ID: {latest_version.run_id}).")

        # --- Replace the deprecated transition_model_version_stage call ---
        # First, optionally remove the 'Staging' alias from any previous version
        # This ensures only one version has the 'Staging' alias at a time.
        try:
            versions_with_current_alias = client.get_latest_versions(name=registered_model_name, aliases=["Staging"])
            for version_with_alias in versions_with_current_alias:
                if version_with_alias.version != latest_version_number:
                     logger.info(f"Removing 'Staging' alias from version {version_with_alias.version}")
                     client.delete_registered_model_alias(
                         name=registered_model_name,
                         alias="Staging"
                     )
        except Exception as remove_alias_err:
             logger.warning(f"Could not remove existing 'Staging' alias: {remove_alias_err}")
             # Continue even if removing the old alias fails

        # Use set_registered_model_alias to set the 'Staging' alias on the latest version
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Staging", # Use 'Staging' as the alias name
            version=latest_version_number
        )
        # --- End of replacement ---

        logger.info(f"Registered model '{registered_model_name}' version {latest_version_number} successfully aliased as 'Staging'.")

    except Exception as e:
        logger.error(f"Failed to set 'Staging' alias for model '{registered_model_name}': {e}")
        logger.error(traceback.format_exc())
        # Decide if you want to re-raise the exception or just log the error
        # raise # Uncomment if you want errors here to stop the pipeline


def log_model_to_mlflow(model, model_name, params, report_dict, X_train, X_test, y_test, wine_feature_names, quality_order, output_dir):
    """Log a model and its artifacts to MLflow"""
    try:
        logger.info(f"Starting MLflow logging for {model_name}...")
        mlflow.log_params(params)
        logger.info(f"Parameters logged for {model_name}.")

        # Ensure metrics are logged, handling potential KeyError more gracefully
        try:
            # Metrics from classification_report dict
            metrics = {
                "accuracy": report_dict.get("accuracy"), # Use .get for safety
                "weighted_avg_precision": report_dict.get("weighted avg", {}).get("precision"),
                "weighted_avg_recall": report_dict.get("weighted avg", {}).get("recall"),
                "weighted_avg_f1-score": report_dict.get("weighted avg", {}).get("f1-score"),
                "macro_avg_precision": report_dict.get("macro avg", {}).get("precision"),
                "macro_avg_recall": report_dict.get("macro avg", {}).get("recall"),
                "macro_avg_f1-score": report_dict.get("macro avg", {}).get("f1-score"),
            }
            # Add metrics for individual classes if they exist in the report
            for i, label_name in enumerate(quality_order):
                 label_key = str(float(i)) # Keys in report_dict are strings like "0.0", "1.0" etc.
                 if label_key in report_dict:
                      metrics[f"precision_{label_name}"] = report_dict[label_key].get("precision")
                      metrics[f"recall_{label_name}"] = report_dict[label_key].get("recall")
                      metrics[f"f1-score_{label_name}"] = report_dict[label_key].get("f1-score")
                      metrics[f"support_{label_name}"] = report_dict[label_key].get("support")

            # Filter out None values if any metric was missing
            metrics = {k: v for k, v in metrics.items() if v is not None}

            mlflow.log_metrics(metrics)
            logger.info(f"Metrics logged for {model_name}: {metrics}")
        except Exception as metric_err:
            logger.error(f"Failed to log metrics for {model_name}: {metric_err}")
            logger.error(traceback.format_exc())
            # Continue even if metric logging fails

        mlflow.set_tag("Training Info", f"{model_name} model for Wine")
        logger.info(f"Tag 'Training Info' set for {model_name}.")


        # Check if model has predict method for signature inference
        if not hasattr(model, 'predict'):
             logger.error(f"Model {model_name} does not have a 'predict' method. Cannot infer signature.")
             signature = None
        else:
            try:
                # Sample a smaller input example if X_train is very large
                input_example = X_train.sample(min(100, X_train.shape[0]), random_state=42) if X_train.shape[0] > 100 else X_train
                signature = infer_signature(input_example, model.predict(input_example))
                logger.info(f"Signature inferred successfully for {model_name}.")
            except Exception as sig_err:
                logger.error(f"Failed to infer signature for {model_name}: {sig_err}")
                logger.error(traceback.format_exc())
                signature = None # Proceed without signature if inference fails


        registered_model_name = f"tracking-wine-{model_name.lower()}"
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name.lower()}_model", # Artifact path within the run
                signature=signature,
                input_example=X_train.head(5), # Use a small subset for input example
                registered_model_name=registered_model_name # Register the model
            )
            logger.info(f"Model {model_name} logged to MLflow run and registered as '{registered_model_name}'.")

            # After successful registration/logging, attempt to transition (set alias)
            transition_model_to_staging(registered_model_name)

        except Exception as e:
            logger.error(f"Failed to log model {model_name} to registry: {e}")
            logger.error(traceback.format_exc())
            # Re-raise if model logging is critical
            raise

        # Save predictions - happens within the run context
        save_predictions(model, X_test, y_test, wine_feature_names, model_name, output_dir)

        # Save plots - happens within the run context
        save_model_plots(model, model_name, wine_feature_names, quality_order, output_dir)

        logger.info(f"MLflow logging completed for {model_name}.")

    except Exception as e:
        logger.error(f"Overall MLflow logging process failed for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise # Re-raise to signal failure in this step

def main():
    """Main function to orchestrate the ML pipeline"""
    logger.info("Starting ML pipeline...")
    try:
        # Setup
        output_dir = setup_environment()
        logger.info(f"Using output directory: {output_dir}")

        # Load data
        df = load_data()

        # Preprocess data
        # Pass wine_feature_names from preprocess_data to training/logging
        X_train, X_test, y_train, y_test, quality_order, wine_feature_names = preprocess_data(df)

        # Assert feature names match number of columns
        if len(wine_feature_names) != X_train.shape[1]:
             logger.error(f"Feature name length ({len(wine_feature_names)}) does not match X_train features ({X_train.shape[1]})")
             raise AssertionError("Feature name mismatch")
        logger.info(f"Feature names ({len(wine_feature_names)}) match X_train shape ({X_train.shape[1]}).")


        # Train models
        models, params, report_dict = train_models(X_train, y_train, X_test, y_test, quality_order)

        # Verify output directory exists before logging artifacts
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Re-created output directory: {output_dir}")

        # MLflow logging within runs
        # Set the experiment name
        mlflow.set_experiment("Wine_Quality_Classification") # Use a clearer name

        # Start a parent run for the overall process
        with mlflow.start_run(run_name="Wine_Model_Training_Run"):
            logger.info(f"Started parent MLflow run: {mlflow.active_run().info.run_id}")

            # Log experiment parameters/info that apply to both models
            mlflow.log_param("dataset", "kaggle_wine_quality_classification")
            mlflow.log_param("train_test_split_ratio", 0.8)
            mlflow.log_param("random_state", 42)

            for model_name, model in models.items():
                logger.info(f"Starting nested MLflow run for {model_name}...")
                # Start a nested run for each model
                with mlflow.start_run(run_name=model_name, nested=True):
                    logger.info(f"Started nested MLflow run: {mlflow.active_run().info.run_id}")

                    # Log model-specific details
                    mlflow.log_param("model_type", model_name)

                    # Log the model and its artifacts
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
                        output_dir=output_dir # Pass the determined output directory
                    )
                    logger.info(f"Completed nested MLflow run for {model_name}.")

            logger.info("Completed parent MLflow run.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        # Log the exception to the current active run if possible
        if mlflow.active_run():
             mlflow.set_tag("status", "Failed")
             mlflow.log_param("error_message", str(e))
        raise # Re-raise the exception to signal failure externally


if __name__ == "__main__":
    # Add basic argument parsing or environment variable checks if needed
    # e.g., for MLFLOW_TRACKING_URI, experiment name etc.
    main()
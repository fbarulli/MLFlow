# mlflow_tracking/scripts/import_mlflow_model.py
import bentoml
import mlflow
import sys
import logging
import os
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_model_by_alias(registered_model_name: str, alias_name: str, bentoml_model_name_prefix: str = "wine_service_model"):
    """
    Finds an MLflow model version by alias and imports it into the BentoML model store.
    Returns the resulting BentoML tag.
    """
    try:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not mlflow_tracking_uri:
            logger.error("MLFLOW_TRACKING_URI environment variable is not set.")
            raise ValueError("MLFLOW_TRACKING_URI not set")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")
        client = mlflow.MlflowClient()

        logger.info(f"Attempting to find MLflow model '{registered_model_name}' with alias '{alias_name}'")

        # Find the model version with the given alias
        versions_with_alias = client.get_latest_versions(name=registered_model_name, aliases=[alias_name])

        if not versions_with_alias:
             logger.error(f"No model version found for '{registered_model_name}' with alias '{alias_name}'.")
             # Log available models for debugging
             logger.info("Available models and aliases:")
             try:
                 all_models = client.search_registered_models()
                 if not all_models:
                      logger.info("  (No registered models found)")
                 for model in all_models:
                      versions = client.search_model_versions(f"name='{model.name}'")
                      for v in versions:
                           aliases_str = ', '.join(v.aliases) if v.aliases else 'None'
                           logger.info(f"  - Model: {v.name}, Version: {v.version}, Stage: {v.current_stage}, Aliases: [{aliases_str}]")
             except Exception as list_err:
                 logger.warning(f"Could not list available models: {list_err}")


             raise ValueError(f"No model version found with alias '{alias_name}' for '{registered_model_name}'")

        # Get the single version with the alias (assuming only one is expected)
        model_version = versions_with_alias[0]
        logger.info(f"Found MLflow model version: {model_version.version} (Run ID: {model_version.run_id})")

        # Construct the MLflow model URI
        # Use the relative artifact path from the source field
        # Example source: dbfs:/databricks/mlflow-tracking/12345/abcdef.../artifacts/model_artifact_path
        # We need runs:/<run_id>/<model_artifact_path>
        # The source string structure can vary slightly depending on backend store (file, s3, abfss, dbfs etc.)
        # A safer way is to explicitly get the run and artifact path if source is not simple run URI
        # Let's assume source is like .../artifacts/<artifact_path> or runs:/<run_id>/<artifact_path>
        artifact_path_in_run = model_version.source.split("artifacts/")[-1] if "artifacts/" in model_version.source else model_version.source.split("/")[-1] # Simplified attempt
        # A more robust way would use mlflow.tracking.artifact_utils.get_artifact_uri
        # But get_artifact_uri often needs a run_id, and model_version has that.
        # Let's stick to the source path part after artifacts/ or just the last segment as a fallback.
        # The 'run:/' syntax should work for most cases where source points to run artifacts.
        mlflow_model_uri = f"runs:/{model_version.run_id}/{artifact_path_in_run}"

        logger.info(f"MLflow model URI for import: {mlflow_model_uri}")

        # Generate desired BentoML model name based on MLflow model name and alias
        # Use lowercase and replace problematic characters if any
        safe_mlflow_name = registered_model_name.lower().replace('-', '_').replace(' ', '_')
        safe_alias_name = alias_name.lower().replace('-', '_').replace(' ', '_')
        # A simple naming convention: <prefix>_<mlflow_name>_<alias>
        bentoml_model_name = f"{bentoml_model_name_prefix}_{safe_mlflow_name}_{safe_alias_name}"

        # Check if a BentoML model with this name already exists
        # We don't check for a specific version here, just the name.
        # If a model with this name exists, import_model will likely create a new version.
        # If you want to avoid importing if *any* version exists, you'd need bentoml.models.list(tag=f"{bentoml_model_name}:*")

        logger.info(f"Importing MLflow model to BentoML store with name '{bentoml_model_name}'")
        # Import the model into the BentoML store.
        # bentoml.mlflow.import_model automatically handles complex MLflow models
        # The first argument is the desired *BentoML model name*. BentoML will assign a version.
        bentomodel = bentoml.mlflow.import_model(
            bentoml_model_name, # The desired BentoML model *name*
            mlflow_model_uri,   # The source MLflow model URI
            signatures={"predict": {"batchable": True}}, # Define signatures if needed
            # Optionally add metadata: metadata={"mlflow_run_id": model_version.run_id, "mlflow_version": model_version.version, ...}
        )
        logger.info(f"Successfully imported MLflow model to BentoML model store. Resulting tag: {bentomodel.tag}")

        # Return the resulting BentoML tag object
        return bentomodel.tag.human_readable() # Return human-readable string tag

    except Exception as e:
        logger.error(f"Error importing MLflow model '{registered_model_name}' with alias '{alias_name}': {e}")
        logger.error(traceback.format_exc())
        # Re-raise or sys.exit as appropriate for the calling script
        raise # Re-raise the exception


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python import_mlflow_model.py <registered_mlflow_model_name> <mlflow_alias_name> [bentoml_model_name_prefix]", file=sys.stderr)
        sys.exit(1)

    # These variable names are correct, they capture the command line arguments
    registered_mlflow_model_name = sys.argv[1]
    mlflow_alias_name = sys.argv[2]
    bentoml_name_prefix = sys.argv[3] if len(sys.argv) > 3 else "wine_service_model" # Default prefix

    try:
        # Import the model and capture the resulting BentoML tag
        # --- FIX IS HERE ---
        bentoml_tag = import_model_by_alias(
            registered_model_name=registered_mlflow_model_name, # Use the correct parameter name as the keyword
            alias_name=mlflow_alias_name,
            bentoml_model_name_prefix=bentoml_name_prefix # This keyword was already correct
        )
        # --- END FIX ---

        # Print the tag to stdout ONLY on success, so the calling script can capture it
        print(bentoml_tag)
        sys.exit(0) # Exit successfully
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1) # Exit with error code

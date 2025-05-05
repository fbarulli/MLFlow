# mlflow_tracking/bentoml/convert_to_bentoml.py
import mlflow
from mlflow.tracking import MlflowClient
import bentoml
import argparse
import logging
import traceback
import json
from datetime import datetime
import os # Import os for artifact path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_registered_models(client):
    """Lists registered models, versions, stages, and aliases."""
    models = client.search_registered_models()
    model_info = []
    if not models:
        return model_info # Return empty list if no models

    logger.info("Fetching details for all registered models...")
    for model in models:
        try:
            latest_versions = client.get_latest_versions(model.name, stages=["Production", "Staging", "Archived", "None"])
            # Also get versions by alias
            # Note: As of some MLflow versions, get_latest_versions might not list *all* versions easily.
            # A more thorough approach to list *all* versions and their aliases/stages requires iterating
            # through model.latest_versions or using client.search_model_versions()
            # For simplicity, we'll list the latest versions by stage/alias we know exist.
            # You might need a more complex loop if you have many aliases/versions.

            # Let's refine this to list all versions associated with the model and their stages/aliases
            all_versions_for_model = client.search_model_versions(f"name='{model.name}'")

            for version in all_versions_for_model:
                info = {
                    "name": version.name,
                    "version": version.version,
                    "stage": version.current_stage, # Still report stage for visibility
                    "aliases": version.aliases, # Report aliases
                    "run_id": version.run_id,
                    "last_updated_timestamp": version.last_updated_timestamp
                }
                model_info.append(info)
                # logger.debug(f"  - Version {info['version']}: Stage={info['stage']}, Aliases={info['aliases']}") # Debugging

        except Exception as e:
             logger.warning(f"Could not retrieve details for model '{model.name}': {e}")
             logger.debug(traceback.format_exc())

    # Sort by name and then version for readability
    model_info.sort(key=lambda x: (x['name'], int(x['version'])))
    return model_info


def convert_mlflow_to_bentoml(bentoml_model_name_prefix):
    """
    Finds MLflow models assigned the 'Production' alias and converts them to BentoML.
    """
    try:
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        client = MlflowClient()

        models_with_production_alias = []
        production_alias_name = "Production" # Define the alias name

        logger.info(f"Searching for MLflow models with alias '{production_alias_name}'...")

        # Iterate through all registered models to find those with the 'Production' alias
        registered_models = client.search_registered_models()
        if not registered_models:
            logger.warning("No registered models found in MLflow.")
            return False

        for model in registered_models:
            try:
                # Use get_latest_versions with the 'aliases' argument to find versions with the specific alias
                versions_with_alias = client.get_latest_versions(name=model.name, aliases=[production_alias_name])

                # get_latest_versions with an alias should typically return at most one version.
                # Loop through results just in case, though normally it's [version_with_alias] or [].
                for version in versions_with_alias:
                    # Use the timestamp of the version itself for better traceability in the BentoML tag
                    # Fallback to current time if timestamp is unavailable
                    timestamp_str = datetime.fromtimestamp(version.last_updated_timestamp / 1000).strftime("%Y%m%d%H%M%S") if version.last_updated_timestamp else datetime.now().strftime("%Y%m%d%H%M%S_now")

                    models_with_production_alias.append({
                        "name": model.name,
                        "version": version.version,
                        # Construct the model_uri using the version's run_id and source path
                        # This is the most reliable way to load a specific version artifact
                        "model_uri": f"runs:/{version.run_id}/{version.source.split('/')[-1]}",
                        "timestamp_str": timestamp_str # Used for BentoML tag
                    })
                    logger.info(f"Found model version {version.version} of '{model.name}' with alias '{production_alias_name}'. Run ID: {version.run_id}")

            except Exception as e:
                 logger.warning(f"Could not check alias '{production_alias_name}' for model '{model.name}': {e}")
                 logger.debug(traceback.format_exc())
                 # Continue to the next model if getting versions by alias fails

        if not models_with_production_alias:
            logger.warning(f"No models found with the '{production_alias_name}' alias.")
            model_list = list_registered_models(client) # Use your helper to list versions/stages/aliases
            if model_list:
                 logger.info("Registered Models and their versions, stages, and aliases:")
                 for m in model_list:
                     aliases_str = ', '.join(m.get('aliases', [])) if m.get('aliases') else 'None'
                     logger.info(f"  - Name: {m['name']}, Version: {m['version']}, Stage: {m['stage']}, Aliases: [{aliases_str}]")
            # Decide if this should be a hard error or just a warning
            # raise ValueError("No models with 'Production' alias found") # Uncomment for hard error
            return False # Indicate no models were converted


        for model_info in models_with_production_alias:
            try:
                logger.info(f"Processing model '{model_info['name']}' (version {model_info['version']}) aliased as '{production_alias_name}'")

                # Load the MLflow model using the constructed URI
                logger.info(f"Loading MLflow model from URI: {model_info['model_uri']}")
                mlflow_model = mlflow.pyfunc.load_model(model_info["model_uri"])
                logger.info("MLflow model loaded successfully.")

                # Generate BentoML model name/tag
                # Split the MLflow name and append the timestamp string for uniqueness
                # Ensure the name is lowercase and safe for BentoML tag
                mlflow_name_suffix = model_info['name'].lower().split('-')[-1] # e.g., 'logisticregression'
                bentoml_model_tag_name = f"{bentoml_model_name_prefix}_{mlflow_name_suffix}_{model_info['timestamp_str']}"
                # BentoML tags are typically <name>:<version>, where <name> is the base name
                # and <version> is the unique identifier.
                # A better BentoML tag structure might be "wine_quality_service_models:<timestamp_str>"
                # Let's use the proposed structure from the entrypoint for consistency:
                # <prefix>_<model_name_suffix>:<timestamp_str>
                bentoml_model_tag = f"{bentoml_model_name_prefix}_{mlflow_name_suffix}:{model_info['timestamp_str']}"
                logger.info(f"Desired BentoML model tag: {bentoml_model_tag}")


                # Check if a BentoML model with this exact tag already exists
                try:
                    existing_bentomodel = bentoml.models.get(bentoml_model_tag)
                    logger.info(f"BentoML model with tag '{bentoml_model_tag}' already exists. Skipping conversion for this model version.")
                    continue # Skip to the next model in the list
                except bentoml.exceptions.NotFound:
                    # Model does not exist, proceed with import
                    logger.info(f"BentoML model with tag '{bentoml_model_tag}' not found. Proceeding with import.")


                # Import the model into the BentoML store
                # This will save it to the volume mounted at /root/.bentoml (~/.bentoml)
                # bentoml.mlflow.import_model handles complex MLflow models
                # The first argument is the desired *BentoML model tag name*, not the full tag.
                # The tag will be `bentoml_model_tag_name:<generated_version>` by default.
                # To specify the full tag, we might need bentoml.models.save instead,
                # or rely on import_model creating a consistent tag based on the URI/name.
                # Let's stick to the simpler import_model and use the returned tag.
                # Update: The first argument to import_model *is* the name part of the tag.
                # The version part is auto-generated unless specified differently or saved manually.
                # Let's use the desired unique identifier as the name part for simplicity here.
                bentoml_model = bentoml.mlflow.import_model(
                    f"{bentoml_model_name_prefix}_{mlflow_name_suffix}", # The desired BentoML model *name*
                    model_info["model_uri"], # The source MLflow model URI
                    signatures={"predict": {"batchable": True}} # Define signatures if needed
                    # Optionally add metadata: metadata={"mlflow_run_id": model_info['run_id'], ...}
                )
                logger.info(f"Converted to BentoML model with tag: {bentomodel.tag}")

                # Log conversion details back to the original MLflow run
                try:
                    # Get the ModelVersion object again to access run_id reliably
                    # (already have run_id in model_info, but getting obj confirms existence)
                    # model_version_obj = client.get_model_version(model_info["name"], model_info["version"]) # No need to get again if we have run_id

                    # Ensure run_id exists and is accessible from the model_info derived earlier
                    if model_info['run_id']:
                         logger.info(f"Logging conversion details to MLflow run ID: {model_info['run_id']}")
                         # Use nested=False to avoid creating nested runs if this script is run within a larger run
                         with mlflow.start_run(run_id=model_info['run_id'], nested=False, log_system_metrics=False): # Disable system metrics for clean log
                             mlflow.set_tag("bentoml_converted", "true")
                             mlflow.set_tag("bentoml_model_tag", bentomodel.tag) # Log the resulting BentoML tag
                             mlflow.set_tag("bentoml_model_name", bentomodel.tag.name) # Log the name part
                             mlflow.set_tag("bentoml_model_version", bentomodel.tag.version) # Log the version part

                             log_data = {
                                 "bentoml_model_tag": bentomodel.tag.human_readable(), # Use human_readable()
                                 "bentoml_model_name": bentomodel.tag.name,
                                 "bentoml_model_version": bentomodel.tag.version,
                                 "mlflow_model_name": model_info['name'],
                                 "mlflow_model_version": model_info['version'],
                                 "mlflow_run_id": model_info['run_id'],
                                 "conversion_timestamp": datetime.now().isoformat(),
                                 "bentoml_import_source": model_info["model_uri"]
                             }

                             # Log the data as a JSON artifact
                             # Get the current run's artifact URI
                             run_artifact_uri = mlflow.active_run().info.artifact_uri
                             log_file_dir = os.path.join(run_artifact_uri.replace("file://", ""), "conversion_logs") # Handle file:// prefix
                             os.makedirs(log_file_dir, exist_ok=True)
                             log_file_name = f"bentoml_conversion_{bentomodel.tag.name}_{bentomodel.tag.version}.json".replace(":", "_") # Safe filename
                             log_file_path = os.path.join(log_file_dir, log_file_name)


                             try:
                                 with open(log_file_path, "w") as f:
                                     json.dump(log_data, f, indent=2)
                                 # Log artifact relative to the run's artifact root
                                 mlflow.log_artifact(log_file_path, artifact_path="conversion_logs")
                                 logger.info(f"Conversion details logged to MLflow run artifact: conversion_logs/{log_file_name}")
                             except Exception as log_err:
                                  logger.error(f"Failed to log conversion artifact to MLflow run {model_info['run_id']}: {log_err}")
                                  logger.debug(traceback.format_exc())
                                  # Decide if failure to log artifact should stop conversion - probably not

                    else:
                         logger.warning(f"Model version {model_info['version']} of '{model_info['name']}' has no associated run_id ({model_info.get('run_id')}). Cannot log conversion details to MLflow run.")

                except Exception as log_start_err:
                     logger.error(f"Error starting MLflow run for logging conversion details: {log_start_err}")
                     logger.debug(traceback.format_exc())


            except Exception as conversion_err:
                logger.error(f"Failed to convert MLflow model '{model_info['name']}' (version {model_info['version']}): {conversion_err}")
                logger.error(traceback.format_exc())
                # Decide if failure to convert one model should stop the script - current logic continues

        return len(models_with_production_alias) > 0 # Return True if at least one model was found with the alias
    except Exception as main_err:
        logger.error(f"Overall conversion process failed: {main_err}")
        logger.error(traceback.format_exc())
        raise # Re-raise to signal failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MLflow Production alias models to BentoML")
    parser.add_argument("--bentoml-model-name-prefix", type=str, default="wine", help="Prefix for BentoML model names (e.g., 'wine' -> 'wine_logisticregression')")
    parser.add_argument("--list-models", action="store_true", help="List registered models, versions, stages, and aliases and exit")
    args = parser.parse_args()

    # Set MLflow tracking URI before creating client for list_models
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = MlflowClient()

    if args.list_models:
        models = list_registered_models(client)
        if not models:
            print("No registered models found.")
        else:
            print("Registered Models:")
            for m in models:
                aliases_str = ', '.join(m.get('aliases', [])) if m.get('aliases') else 'None'
                print(f"  Name: {m['name']}, Version: {m['version']}, Stage: {m['stage']}, Aliases: [{aliases_str}], Run ID: {m['run_id']}")
        exit(0)

    # Run the conversion
    success = convert_mlflow_to_bentoml(args.bentoml_model_name_prefix)
    if not success:
        logger.warning("No 'Production' aliased models were found or converted.")
        # Exit with non-zero code if no production models were found, signalling issue to orchestrator
        # sys.exit(1) # Uncomment if no production models is considered an error
    else:
         logger.info("BentoML conversion script finished.")
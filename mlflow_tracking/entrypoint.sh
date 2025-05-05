#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging - prints commands as they are executed

# This script is the ENTRYPOINT for the BentoML container.
# It's responsible for:
# 1. Ensuring MLflow server is accessible (optional but good practice).
# 2. Importing the desired model from MLflow into the container's BentoML store.
# 3. Building the Bento Service definition.
# 4. Serving the built Bento.

# Expected Environment Variables:
# MLFLOW_TRACKING_URI (set by docker-compose)
# BENTOML_MODEL_TO_SERVE_NAME (e.g., tracking-wine-logisticregression, set by docker-compose)
# BENTOML_MODEL_TO_SERVE_ALIAS (e.g., Production, set by docker-compose)
# BENTOML_MODEL_IMPORT_PREFIX (e.g., wine_service_model, set by docker-compose)
# BENTOML_SERVICE_DEFINITION (e.g., ./bentoml/service.py:svc, hardcoded or set by docker-compose)

# Ensure environment variables are set
if [ -z "$MLFLOW_TRACKING_URI" ]; then echo "Error: MLFLOW_TRACKING_URI not set."; exit 1; fi
if [ -z "$BENTOML_MODEL_TO_SERVE_NAME" ]; then echo "Error: BENTOML_MODEL_TO_SERVE_NAME not set."; exit 1; fi
if [ -z "$BENTOML_MODEL_TO_SERVE_ALIAS" ]; then echo "Error: BENTOML_MODEL_TO_SERVE_ALIAS not set."; exit 1; fi
if [ -z "$BENTOML_MODEL_IMPORT_PREFIX" ]; then echo "Error: BENTOML_MODEL_IMPORT_PREFIX not set."; exit 1; fi

# Set environment variables for python scripts
export MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI"

echo "MLflow Tracking URI set to: $MLFLOW_TRACKING_URI"

# --- Wait for MLflow Server (Optional but Recommended) ---
# Requires a wait-for-it.sh script or similar.
# COPY wait-for-it.sh . and RUN chmod +x wait-for-it.sh in Dockerfile
# /app/wait-for-it.sh mlflow-server:5000 -- echo "MLflow server is up and running."
echo "Skipping wait-for-it for simplicity. Ensure MLflow server is ready before starting container."

# --- Import Model from MLflow into BentoML Store ---
# This script will import the model into the volume mounted at /root/.bentoml (~/.bentoml)
echo "Importing model from MLflow: '$BENTOML_MODEL_TO_SERVE_NAME' with alias '$BENTOML_MODEL_TO_SERVE_ALIAS' using prefix '$BENTOML_MODEL_IMPORT_PREFIX'..."

# Run the python script and capture its stdout (the imported BentoML tag)
# The script `import_mlflow_model.py` is now expected at /app/bentoml/
# Change path from /app/scripts/ to /app/bentoml/
IMPORTED_BENTO_TAG=$(python /app/bentoml/import_mlflow_model.py "$BENTOML_MODEL_TO_SERVE_NAME" "$BENTOML_MODEL_TO_SERVE_ALIAS" "$BENTOML_MODEL_IMPORT_PREFIX")

# Check if the import script succeeded and returned a tag
if [ $? -ne 0 ] || [ -z "$IMPORTED_BENTO_TAG" ]; then
    echo "Error: Model import failed. Check logs above."
    exit 1
fi
echo "Model successfully imported into BentoML store with tag: $IMPORTED_BENTO_TAG"


# --- Build the Bento ---
# Build the service definition. This will package the code and reference the model(s)
# that were just imported into the local BentoML store (/root/.bentoml).
# The service definition (bentoml/service.py:svc) should be designed to load the
# model by tag. We can pass the imported tag to the service via an environment variable.
export BENTOML_SERVE_MODEL_TAG="$IMPORTED_BENTO_TAG" # Make the tag available to service.py

BENTOML_SERVICE_DEFINITION=${BENTOML_SERVICE_DEFINITION:-"./bentoml/service.py:svc"} # Default service path

echo "Building Bento for $BENTOML_SERVICE_DEFINITION using model tag $BENTOML_SERVE_MODEL_TAG..."

# bentoml build automatically assigns a tag based on the service name and a version hash.
# The build output will show the created Bento tag. We don't strictly need to capture it
# if we always serve the latest built version or rely on the service definition.
# However, for robustness, we could try to capture the *actual* built tag.
# Capturing the build output is complex. Let's keep it simple and assume the build
# succeeds and the service definition correctly loads the desired model by tag.
bentoml build "$BENTOML_SERVICE_DEFINITION"

echo "Bento build finished." # Assuming successful build

# --- Serve the Built Bento ---
# Serve the Bento using its tag. We need to figure out the tag BentoML assigned during build.
# The build output contains lines like `Successfully built Bento: wine_quality_service:abcd123456789`
# A simple way is to serve the 'latest' version of the service name,
# assuming the build correctly updated 'latest'. Or, we could parse the build output.
# Let's serve the 'latest' version of the service name defined in service.py.
# The service name is typically the part before ':svc'.
SERVICE_NAME=$(echo "$BENTOML_SERVICE_DEFINITION" | sed 's|.*/\([^:]*\):svc|\1|') # Extract service name
BENTO_TO_SERVE="${SERVICE_NAME}:latest" # Assume the build updates latest

echo "Serving Bento: $BENTO_TO_SERVE on port 3000..."
# The 'exec' command replaces the shell script process with the bentoml serve process,
# ensuring signals (like SIGTERM from docker stop) are correctly handled.
exec bentoml serve "$BENTO_TO_SERVE" --port 3000
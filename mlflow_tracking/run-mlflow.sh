#!/bin/bash
# mlflow_tracking/run-mlflow.sh
set -euo pipefail

IMAGE_NAME="wine-mlflow-app"
DOCKERFILE_NAME="Dockerfile.mlflow"
HOST_OUTPUT_DIR="./wine_outputs"
HOST_ENV_FILE="./.env"
CONTAINER_OUTPUT_DIR="/app/outputs"

cleanup() {
    echo "--- Running cleanup ---"
    if [ -d "${HOST_OUTPUT_DIR}" ]; then
        echo "Cleaning up contents of ${HOST_OUTPUT_DIR}..."
        find "${HOST_OUTPUT_DIR}" -mindepth 1 -delete || echo "Warning: Failed to clean contents of ${HOST_OUTPUT_DIR}. Manual cleanup might be needed."
    fi
    echo "Running docker system prune --force..."
    docker system prune --force || true
    echo "--- Cleanup finished ---"
}

trap cleanup EXIT INT TERM

echo "--- Starting MLflow Wine Pipeline ---"

echo "Checking for environment file: ${HOST_ENV_FILE}"
if [ ! -f "${HOST_ENV_FILE}" ]; then
    echo "Error: Environment file '${HOST_ENV_FILE}' not found!"
    echo "Please create it at the project root with at least 'MLFLOW_TRACKING_URI=http://localhost:5000' (if using --network host)"
    exit 1
fi
echo "Environment file '${HOST_ENV_FILE}' found."

echo "Ensuring host output directory exists: ${HOST_OUTPUT_DIR}"
mkdir -p "${HOST_OUTPUT_DIR}"
echo "Host output directory ready."

echo "Building Docker image '${IMAGE_NAME}' using Dockerfile '${DOCKERFILE_NAME}'..."
docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE_NAME}" .
echo "Docker image built successfully."

echo "Running Docker container from image '${IMAGE_NAME}'..."
# Add --network host here
docker run --rm \
           --network host \
           --env-file "${HOST_ENV_FILE}" \
           -v "$(pwd)/${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
           "${IMAGE_NAME}"

echo "--- MLflow Wine Pipeline finished ---"

exit 0
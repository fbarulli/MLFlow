#!/bin/bash

set -euo pipefail

TRAINING_IMAGE_NAME="wine-mlflow-app"
TRAINING_DOCKERFILE_NAME="Dockerfile.mlflow"
COMPOSE_FILE="./docker-compose.yml"
HOST_ENV_FILE="./.env"

HOST_APP_OUTPUT_DIR="./wine_outputs"

cleanup() {
    echo "--- Running cleanup ---"

    echo "Running docker compose down --volumes --remove-orphans..."
    docker compose -f "${COMPOSE_FILE}" down --volumes --remove-orphans || true

    if [ -d "${HOST_APP_OUTPUT_DIR}" ]; then
        echo "Cleaning up contents of ${HOST_APP_OUTPUT_DIR} on host..."
        find "${HOST_APP_OUTPUT_DIR}" -mindepth 1 -delete || echo "Warning: Failed to clean contents of ${HOST_APP_OUTPUT_DIR}. Manual cleanup might be needed."
    fi

    echo "--- Cleanup finished ---"
}

trap cleanup EXIT INT TERM


echo "--- Starting MLflow Wine Pipeline with Docker Compose ---"

echo "Checking for environment file: ${HOST_ENV_FILE}"
if [ ! -f "${HOST_ENV_FILE}" ]; then
    echo "Error: Environment file '${HOST_ENV_FILE}' not found! Required by docker-compose.yml"
    exit 1
fi
echo "Environment file '${HOST_ENV_FILE}' found."

echo "Ensuring host output directory exists: ${HOST_APP_OUTPUT_DIR}"
mkdir -p "${HOST_APP_OUTPUT_DIR}"
echo "Host output directory ready."

echo "Building Docker image '${TRAINING_IMAGE_NAME}' using Dockerfile '${TRAINING_DOCKERFILE_NAME}'..."
docker build -t "${TRAINING_IMAGE_NAME}" -f "${TRAINING_DOCKERFILE_NAME}" .
echo "Docker image built successfully."

echo "Bringing up services defined in ${COMPOSE_FILE}..."
docker compose -f "${COMPOSE_FILE}" up --build


echo "--- Docker Compose stack operations finished ---"

exit 0
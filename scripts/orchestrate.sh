#!/bin/bash
# ~/MLFlow/scripts/orchestrate.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting orchestration at $(date)"
echo "Script directory: $SCRIPT_DIR"

echo "Step 1: Running shutdown_services.sh..."
bash "$SCRIPT_DIR/shutdown_services.sh" || { echo "Failed shutdown_services.sh"; exit 1; }
echo "Completed shutdown_services.sh"

echo "Step 2: Running configure_git.sh..."
bash "$SCRIPT_DIR/configure_git.sh" || { echo "Failed configure_git.sh"; exit 1; }
echo "Completed configure_git.sh"

echo "Step 3: Running configure_dvc.sh..."
bash "$SCRIPT_DIR/configure_dvc.sh" || { echo "Failed configure_dvc.sh"; exit 1; }
echo "Completed configure_dvc.sh"

echo "Step 4: Running build_docker.sh..."
bash "$SCRIPT_DIR/build_docker.sh" || { echo "Failed build_docker.sh"; exit 1; }
echo "Completed build_docker.sh"

echo "Step 5: Running setup_evidently.sh..."
bash "$SCRIPT_DIR/setup_evidently.sh" || { echo "Failed setup_evidently.sh"; exit 1; }
echo "Completed setup_evidently.sh"

echo "Step 6: Running setup_airflow.sh..."
bash "$SCRIPT_DIR/setup_airflow.sh" || { echo "Failed setup_airflow.sh"; exit 1; }
echo "Completed setup_airflow.sh"

echo "Step 7: Running start_mlflow.sh..."
bash "$SCRIPT_DIR/start_mlflow.sh" || { echo "Failed start_mlflow.sh"; exit 1; }
echo "Completed start_mlflow.sh"

echo "Step 8: Running start_webserver.sh..."
bash "$SCRIPT_DIR/start_webserver.sh" || { echo "Failed start_webserver.sh"; exit 1; }
echo "Completed start_webserver.sh"

echo "Step 9: Running start_scheduler.sh..."
bash "$SCRIPT_DIR/start_scheduler.sh" || { echo "Failed start_scheduler.sh"; exit 1; }
echo "Completed start_scheduler.sh"

echo "Setup complete at $(date)!"
echo "Airflow Web GUI: http://localhost:8080 (admin/admin)"
echo "MLFlow UI: http://localhost:5000"
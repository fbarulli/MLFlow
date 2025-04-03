#!/bin/bash
set -e

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load environment configuration
echo "Loading environment configuration..."
eval "$(python3 "$PROJECT_ROOT/scripts/load_env.py")"
eval "$(python3 "$PROJECT_ROOT/scripts/get_mlflow_config.py")"

# Print configuration
echo "Starting orchestration at $(date)"
echo "Project root: $PROJECT_ROOT"
echo "AIRFLOW_HOME: $AIRFLOW_HOME"
echo "MLflow host: $MLFLOW_HOST:$MLFLOW_PORT"

# Set Airflow variables
export _AIRFLOW_DB_MIGRATE="true"
export AIRFLOW__CORE__DISABLE_DB_UPGRADE=True

# Check for demo mode
DEMO_MODE=false
if [ "$1" = "demo" ]; then
    DEMO_MODE=true
    echo "Running in demo mode: Skipping Git and DVC configuration"
fi

echo "Step 1: Running shutdown_services.sh..."
bash "$PROJECT_ROOT/scripts/shutdown_services.sh" || { echo "Failed shutdown_services.sh"; exit 1; }
echo "Completed shutdown_services.sh"

if [ "$DEMO_MODE" = false ]; then
    echo "Step 2: Running configure_git.sh..."
    bash "$PROJECT_ROOT/scripts/configure_git.sh" || { echo "Failed configure_git.sh"; exit 1; }
    echo "Completed configure_git.sh"

    echo "Step 3: Running configure_dvc.sh..."
    bash "$PROJECT_ROOT/scripts/configure_dvc.sh" || { echo "Failed configure_dvc.sh"; exit 1; }
    echo "Completed configure_dvc.sh"
else
    echo "Skipping Git and DVC configuration in demo mode"
fi

echo "Step 4: Running build_docker.sh..."
bash "$PROJECT_ROOT/scripts/build_docker.sh" || { echo "Failed build_docker.sh"; exit 1; }
echo "Completed build_docker.sh"

echo "Step 5: Running setup_evidently.sh..."
bash "$PROJECT_ROOT/scripts/setup_evidently.sh" || { echo "Failed setup_evidently.sh"; exit 1; }
echo "Completed setup_evidently.sh"

echo "Step 6: Running setup_postgres.sh..."
bash "$PROJECT_ROOT/scripts/setup_postgres.sh" || { echo "Failed setup_postgres.sh"; exit 1; }
echo "Completed setup_postgres.sh"

echo "Step 7: Running setup_airflow.py..."
python3 "$PROJECT_ROOT/setup_airflow.py" || { echo "Failed setup_airflow.py"; exit 1; }
echo "Completed setup_airflow.py"

echo "Step 8: Running start_mlflow.sh..."
bash "$PROJECT_ROOT/scripts/start_mlflow.sh" || { echo "Failed start_mlflow.sh"; exit 1; }
echo "Completed start_mlflow.sh"

echo "Step 9: Running start_webserver.sh..."
bash "$PROJECT_ROOT/scripts/start_webserver.sh" || { echo "Failed start_webserver.sh"; exit 1; }
echo "Completed start_webserver.sh"

echo "Step 10: Running start_scheduler.sh..."
bash "$PROJECT_ROOT/scripts/start_scheduler.sh" || { echo "Failed start_scheduler.sh"; exit 1; }
echo "Completed start_scheduler.sh"

echo "Setup complete at $(date)!"
echo "Services:"
echo "- Airflow Web UI: http://${AIRFLOW__WEBSERVER__WEB_SERVER_HOST}:${AIRFLOW__WEBSERVER__WEB_SERVER_PORT}"
echo "  Username: ${ADMIN_USER}"
echo "  Password: ${ADMIN_PASSWORD}"
echo "- MLflow UI: http://${MLFLOW_HOST}:${MLFLOW_PORT}"
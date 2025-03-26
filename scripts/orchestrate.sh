#!/bin/bash
# ~/MLFlow/scripts/orchestrate.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting orchestration of setup tasks..."

bash "$SCRIPT_DIR/shutdown_airflow.sh"
bash "$SCRIPT_DIR/configure_git.sh"
bash "$SCRIPT_DIR/configure_dvc.sh"
bash "$SCRIPT_DIR/build_docker.sh"
bash "$SCRIPT_DIR/setup_evidently.sh"
bash "$SCRIPT_DIR/setup_airflow.sh"
bash "$SCRIPT_DIR/start_webserver.sh"
bash "$SCRIPT_DIR/start_scheduler.sh"

echo "Airflow setup complete!"
echo "Web GUI should be available at: http://localhost:8080"
echo "Login with username: admin, password: admin"
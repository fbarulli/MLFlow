#!/bin/bash
# ~/MLFlow/scripts/start_webserver.sh
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="$PROJECT_ROOT/miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
fi
echo "Starting Airflow webserver..."
"$PYTHON_EXECUTABLE" -m airflow webserver --port 8080 --hostname 0.0.0.0 &
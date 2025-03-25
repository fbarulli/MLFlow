#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export AIRFLOW_HOME="$PROJECT_ROOT"
PYTHON_EXECUTABLE="./miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "Python3 not found in PATH or local env"
        exit 1
    fi
fi
echo "Setting up Airflow in: $AIRFLOW_HOME"
echo "Using Python: $PYTHON_EXECUTABLE"
echo "Running Airflow setup..."
"$PYTHON_EXECUTABLE" setup_airflow.py
echo "Starting Airflow webserver..."
"$PYTHON_EXECUTABLE" -m airflow webserver --port 8080 --hostname 0.0.0.0 &
echo "Starting Airflow scheduler..."
"$PYTHON_EXECUTABLE" -m airflow scheduler &
echo "Airflow setup complete!"
echo "Web GUI should be available at: http://localhost:8080"
echo "Login with username: admin, password: admin"
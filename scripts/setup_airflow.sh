#!/bin/bash
# ~/MLFlow/scripts/setup_airflow.sh
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="$PROJECT_ROOT/miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "Python3 not found in PATH or local env"
        exit 1
    fi
fi
export AIRFLOW_HOME="$PROJECT_ROOT"
echo "Setting up Airflow in: $AIRFLOW_HOME"
echo "Using Python: $PYTHON_EXECUTABLE"

# Check if airflow.cfg exists
if [ -f "$AIRFLOW_HOME/airflow.cfg" ]; then
    echo "airflow.cfg already exists, skipping config creation."
else
    echo "Creating new airflow.cfg..."
    "$PYTHON_EXECUTABLE" -m airflow config list >/dev/null
fi

# Check if database is initialized
if [ -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Airflow database already initialized, skipping init."
else
    echo "Initializing Airflow database..."
    "$PYTHON_EXECUTABLE" -m airflow db init
fi

echo "Airflow setup complete."
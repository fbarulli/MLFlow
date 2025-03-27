#!/bin/bash
# ~/MLFlow/scripts/setup_airflow.sh
set -e

# Use absolute path for AIRFLOW_HOME
export AIRFLOW_HOME="/home/ubuntu/MLFlow"

PYTHON_EXECUTABLE="$AIRFLOW_HOME/miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "Python3 not found in PATH or local env"
        exit 1
    fi
fi

# This environment variable prevents the interactive prompt
export _AIRFLOW_DB_MIGRATE="true"
export AIRFLOW__CORE__DISABLE_DB_UPGRADE=True

echo "Setting up Airflow in: $AIRFLOW_HOME"
echo "Using Python: $PYTHON_EXECUTABLE"

# Run the custom setup_airflow.py script instead of using default Airflow commands
echo "Running custom setup_airflow.py script..."
"$PYTHON_EXECUTABLE" "$AIRFLOW_HOME/setup_airflow.py"

echo "Airflow setup complete."
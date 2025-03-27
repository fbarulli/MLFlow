#!/bin/bash
# ~/MLFlow/scripts/start_scheduler.sh
set -e

# Use absolute path for AIRFLOW_HOME
export AIRFLOW_HOME="/home/ubuntu/MLFlow"

PYTHON_EXECUTABLE="$AIRFLOW_HOME/miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
fi

# This environment variable prevents the interactive prompt
export _AIRFLOW_DB_MIGRATE="true"
export AIRFLOW__CORE__DISABLE_DB_UPGRADE=True

echo "Starting Airflow scheduler..."
"$PYTHON_EXECUTABLE" -m airflow scheduler &
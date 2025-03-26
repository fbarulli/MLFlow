#!/bin/bash
# ~/MLFlow/scripts/start_scheduler.sh
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="$PROJECT_ROOT/miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
fi
export AIRFLOW_HOME="$PROJECT_ROOT"  # Add this line
echo "Starting Airflow scheduler..."
"$PYTHON_EXECUTABLE" -m airflow scheduler &
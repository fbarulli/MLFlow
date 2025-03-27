#!/bin/bash
set -e


export AIRFLOW_HOME="/home/ubuntu/MLFlow"

PYTHON_EXECUTABLE="$AIRFLOW_HOME/miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
fi


export _AIRFLOW_DB_MIGRATE="true"
export AIRFLOW__CORE__DISABLE_DB_UPGRADE=True

echo "Initializing Airflow database with AIRFLOW_HOME=$AIRFLOW_HOME"
"$PYTHON_EXECUTABLE" -m airflow db init
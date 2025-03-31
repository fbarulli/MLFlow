#!/bin/bash
# ~/MLFlow/scripts/restart_airflow.sh
#
# Quick restart script to pick up Airflow code changes.
# No initialization or database setup needed - just stops and starts the services.

# Ensure we're in the correct directory
cd "$(dirname "$(dirname "$0")")"

# Check if we're in the right conda environment
if [ "$CONDA_DEFAULT_ENV" != "mlflow" ]; then
    echo "Warning: You are not in the 'mlflow' conda environment"
    echo "Please run: conda activate mlflow"
    exit 1
fi

# Set up environment
export AIRFLOW_HOME="$(pwd)"
export _AIRFLOW_DB_MIGRATE="true"
export AIRFLOW__CORE__DISABLE_DB_UPGRADE=True

# Get Python executable from current environment
PYTHON_EXECUTABLE=$(which python3)
if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "Error: Python3 not found in current environment"
    echo "Make sure you have activated the correct conda environment"
    exit 1
fi

# Set path to airflow executable in same environment
AIRFLOW_EXECUTABLE=$(dirname $PYTHON_EXECUTABLE)/airflow

echo "Restarting Airflow services..."
echo "Using Python: $PYTHON_EXECUTABLE"
echo "Using Airflow: $AIRFLOW_EXECUTABLE"
echo "Working directory: $(pwd)"

# Gracefully stop services
pkill -TERM -f "airflow scheduler" 2>/dev/null || true
pkill -TERM -f "airflow webserver" 2>/dev/null || true

# Give services time to shutdown gracefully
sleep 2

# Start services in background with proper environment
echo "Starting services with updated code..."
$AIRFLOW_EXECUTABLE webserver -D
$AIRFLOW_EXECUTABLE scheduler > logs/scheduler.log 2>&1 &

# Wait a moment then check if services started
sleep 5
if pgrep -f "airflow webserver" >/dev/null && pgrep -f "airflow scheduler" >/dev/null; then
    echo "Airflow services started successfully!"
    echo "Web UI available at: http://localhost:8080 (admin/admin)"
else
    echo "Warning: One or more Airflow services may not have started properly"
    echo "Check logs for details"
fi

echo "Airflow restarted! Services are starting up..."
echo "Webserver: http://localhost:8080"
# Start the webserver in background
airflow webserver -D

# Start the scheduler in background
nohup airflow scheduler > logs/scheduler.log 2>&1 &

echo "Airflow services restarted successfully!"
echo "Webserver running at http://localhost:8080"
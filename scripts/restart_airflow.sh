#!/bin/bash
# ~/MLFlow/scripts/restart_airflow.sh
#
# Quick restart script to pick up Airflow code changes.
# No initialization or database setup needed - just stops and starts the services.

# Ensure we're in the correct directory
cd "$(dirname "$(dirname "$0")")"

# Check if we're in the right conda environment
if [ "$CONDA_DEFAULT_ENV" != "mlflow" ]; then
    echo "Error: Not in 'mlflow' conda environment"
    echo "Please run: conda activate mlflow"
    exit 1
fi

# Set up environment
export AIRFLOW_HOME="$(pwd)"
export _AIRFLOW_DB_MIGRATE="true"
export AIRFLOW__CORE__DISABLE_DB_UPGRADE=True

# Get executables from current environment
AIRFLOW_EXECUTABLE=$(which airflow)
if [ -z "$AIRFLOW_EXECUTABLE" ]; then
    echo "Error: airflow not found in current environment"
    exit 1
fi

echo "Restarting Airflow services..."
echo "Using Airflow: $AIRFLOW_EXECUTABLE"
echo "Working directory: $AIRFLOW_HOME"

# Stop existing Airflow services using logic from shutdown_services.sh
echo "Stopping existing Airflow services..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
pkill -f "airflow tasks run" || true # Stop any running tasks

# Check Airflow ports (8080 for webserver, 8793 for workers if used)
AIRFLOW_PORTS=(8080 8793)
for PORT in "${AIRFLOW_PORTS[@]}"; do
    if lsof -i :$PORT >/dev/null 2>&1; then
        echo "Port $PORT in use, attempting to free it..."
        kill -9 $(lsof -t -i :$PORT) || true
    else
        echo "Port $PORT is already free."
    fi
done

# Give services time to shutdown gracefully
echo "Waiting for services to stop..."
sleep 5

# Start services
echo "Starting services..."
$AIRFLOW_EXECUTABLE webserver -D
$AIRFLOW_EXECUTABLE scheduler > logs/scheduler.log 2>&1 &

# Verify services started
echo "Verifying services..."
sleep 5
if pgrep -f "airflow webserver" >/dev/null && pgrep -f "airflow scheduler" >/dev/null; then
    echo "✓ Airflow services started successfully!"
    echo "✓ Web UI available at: http://localhost:8080"
else
    echo "! Warning: One or more services may not have started properly"
    echo "! Check logs for details"
    exit 1
fi
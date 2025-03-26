#!/bin/bash
# ~/MLFlow/scripts/shutdown_services.sh
set -e
echo "Shutting down any lingering Airflow and MLFlow services..."

# Kill Airflow processes
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true

# Kill MLFlow processes with more precision
pkill -f "mlflow server" || true
# Ensure any process on port 5000 is terminated
if lsof -i :5000 >/dev/null 2>&1; then
    echo "Port 5000 in use, attempting to free it..."
    kill -9 $(lsof -t -i :5000) || true
fi

echo "Lingering services terminated."
#!/bin/bash
# ~/MLFlow/scripts/shutdown_services.sh
set -e
echo "Shutting down all Airflow and MLflow services..."


pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
pkill -f "airflow tasks run" || true  


pkill -f "mlflow server" || true


PORTS=(5000 8080 8793)  
for PORT in "${PORTS[@]}"; do
    if lsof -i :$PORT >/dev/null 2>&1; then
        echo "Port $PORT in use, attempting to free it..."
        kill -9 $(lsof -t -i :$PORT) || true
    else
        echo "Port $PORT is already free."
    fi
done


sleep 1
if pgrep -f "airflow|mlflow" >/dev/null; then
    echo "Warning: Some Airflow/MLflow processes still running, forcing termination..."
    pkill -9 -f "airflow|mlflow" || true
fi

echo "All services terminated."
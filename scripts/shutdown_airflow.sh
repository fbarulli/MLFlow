#!/bin/bash
# ~/MLFlow/scripts/shutdown_airflow.sh
set -e
echo "Shutting down any lingering Airflow services..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
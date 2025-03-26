#!/bin/bash
# ~/MLFlow/scripts/start_mlflow.sh
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

# Check if port 5000 is free
if lsof -i :5000 >/dev/null 2>&1; then
    echo "Error: Port 5000 is still in use after cleanup. Please free it manually."
    exit 1
fi

echo "Starting MLFlow server..."
"$PYTHON_EXECUTABLE" -m mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "file://$PROJECT_ROOT/mlruns" \
    --default-artifact-root "file://$PROJECT_ROOT/mlruns" &
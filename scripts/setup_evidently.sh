#!/bin/bash
# ~/MLFlow/scripts/setup_evidently.sh
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
echo "Setting up Evidently..."
"$PYTHON_EXECUTABLE" -m pip install evidently
echo "Evidently installed."
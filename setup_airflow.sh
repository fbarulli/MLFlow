#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export AIRFLOW_HOME="$PROJECT_ROOT"
PYTHON_EXECUTABLE="./miniconda3/envs/mlflow/bin/python3"
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    PYTHON_EXECUTABLE=$(which python3)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "Python3 not found in PATH or local env"
        exit 1
    fi
fi

echo "Shutting down any lingering Airflow services..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true


if [ -z "$(git config --global user.email)" ] || [ -z "$(git config --global user.name)" ]; then
    echo "Git user configuration not found."
    read -p "Enter your Git email: " git_email
    read -p "Enter your Git name: " git_name
    git config --global user.email "$git_email"
    git config --global user.name "$git_name"
    echo "Git configured with email: $git_email, name: $git_name"
else
    echo "Git configuration already set: $(git config --global user.email), $(git config --global user.name)"
fi


if ! dvc remote list | grep -q "myremote"; then
    echo "DVC remote 'myremote' not found."
    dvc remote add -d myremote https://dagshub.com/fbarulli/MLFlow.dvc
    echo "DVC remote 'myremote' added."
    read -p "Enter your DagsHub username (e.g., fbarulli): " dvc_user
    read -s -p "Enter your DagsHub token: " dvc_token
    echo
    dvc remote modify myremote --local auth basic
    dvc remote modify myremote --local user "$dvc_user"
    dvc remote modify myremote --local password "$dvc_token"
    echo "DVC remote 'myremote' configured with username: $dvc_user"
else
    echo "DVC remote 'myremote' already configured."
fi

echo "Building Docker container for weather-collector:light..."
docker build -t weather-collector:light .

echo "Setting up Airflow in: $AIRFLOW_HOME"
echo "Using Python: $PYTHON_EXECUTABLE"
echo "Running Airflow setup..."
"$PYTHON_EXECUTABLE" setup_airflow.py

echo "Starting Airflow webserver..."
"$PYTHON_EXECUTABLE" -m airflow webserver --port 8080 --hostname 0.0.0.0 &

echo "Starting Airflow scheduler..."
"$PYTHON_EXECUTABLE" -m airflow scheduler &

echo "Airflow setup complete!"
echo "Web GUI should be available at: http://localhost:8080"
echo "Login with username: admin, password: admin"
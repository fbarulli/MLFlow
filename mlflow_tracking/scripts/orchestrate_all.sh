#!/bin/bash

# Ensure output directory exists
mkdir -p outputs/tests

# Log file for orchestrator
LOG_FILE="outputs/tests/orchestrate_all.log"
echo "Orchestration started at $(date)" > "$LOG_FILE"

# Function to log and check command status
run_command() {
    local cmd="$1"
    local desc="$2"
    echo "Running: $desc" | tee -a "$LOG_FILE"
    $cmd >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed: $desc" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "Completed: $desc" | tee -a "$LOG_FILE"
}

# Start Docker services
run_command "./run.sh" "Starting Docker services"

# Wait for services to be healthy
echo "Waiting for mlflow-server to be healthy..." | tee -a "$LOG_FILE"
until docker inspect --format='{{.State.Health.Status}}' mlflow_tracking-mlflow-server-1 | grep -q "healthy"; do
    sleep 2
done
echo "mlflow-server is healthy" | tee -a "$LOG_FILE"

# Update model metadata
run_command "docker exec -it mlflow_tracking-app-1 python mlflow/model_metadata.py" "Updating model metadata"

# Convert MLflow models to BentoML
run_command "docker exec -it mlflow_tracking-bentoml-1 python bentoml/convert_to_bentoml.py" "Converting MLflow models to BentoML"

# Run test orchestrator
run_command "./scripts/test_orchestrator.sh" "Running all tests"

echo "Orchestration completed successfully at $(date)" | tee -a "$LOG_FILE"
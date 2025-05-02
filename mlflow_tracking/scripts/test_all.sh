#!/bin/bash
echo "Running all tests..."
docker exec -it mlflow_tracking-bentoml-1 pytest /app/bentoml/tests -v
if [ $? -eq 0 ]; then
    echo "All tests passed. Results in ./outputs/tests."
else
    echo "Tests failed. Check logs and ./outputs/tests."
    exit 1
fi
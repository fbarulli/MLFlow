#!/bin/bash
echo "Running error handling tests..."
docker exec -it mlflow_tracking-bentoml-1 pytest /app/bentoml/tests/test_errors.py -v
if [ $? -eq 0 ]; then
    echo "Error tests passed. Results in ./outputs/tests/test_results.json."
else
    echo "Error tests failed. Check logs and ./outputs/tests."
    exit 1
fi
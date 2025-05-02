#!/bin/bash
echo "Running input validation tests..."
docker exec -it mlflow_tracking-bentoml-1 pytest /app/bentoml/tests/test_validation.py -v
if [ $? -eq 0 ]; then
    echo "Validation tests passed. Results in ./outputs/tests/test_results.json."
else
    echo "Validation tests failed. Check logs and ./outputs/tests."
    exit 1
fi
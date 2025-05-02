#!/bin/bash
echo "Running API endpoint tests..."
docker exec -it mlflow_tracking-bentoml-1 pytest /app/bentoml/tests/test_api.py -v
if [ $? -eq 0 ]; then
    echo "API tests passed. Results in ./outputs/tests/test_results.json."
else
    echo "API tests failed. Check logs and ./outputs/tests."
    exit 1
fi
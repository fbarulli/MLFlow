#!/bin/bash
echo "Running performance tests..."
docker exec -it mlflow_tracking-bentoml-1 pytest /app/bentoml/tests/test_performance.py -v
if [ $? -eq 0 ]; then
    echo "Performance tests passed. Results in ./outputs/tests/performance_metrics.csv."
else
    echo "Performance tests failed. Check logs and ./outputs/tests."
    exit 1
fi
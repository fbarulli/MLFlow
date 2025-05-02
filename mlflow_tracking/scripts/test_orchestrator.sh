#!/bin/bash

# Ensure test output directory exists
mkdir -p outputs/tests

# Initialize results file
RESULTS_FILE="outputs/tests/orchestrator_results.json"
echo "[]" > "$RESULTS_FILE"

# Function to log test result
log_result() {
    local test_name="$1"
    local status="$2"
    local error_msg="$3"
    jq ". += [{\"test\": \"$test_name\", \"status\": \"$status\", \"error\": \"$error_msg\"}]" "$RESULTS_FILE" > tmp.json && mv tmp.json "$RESULTS_FILE"
}

# Start BentoML service for API-dependent tests
echo "Starting BentoML service..."
docker exec -d mlflow_tracking-bentoml-1 bentoml serve /app/bentoml/service:svc --port 3000
sleep 5  # Wait for service to start

# Run tests
TESTS=("test_api.sh" "test_validation.sh" "test_errors.sh" "test_performance.sh")
FAILED=0

for test in "${TESTS[@]}"; do
    echo "Running $test..."
    ./scripts/"$test" > /tmp/test_output 2>&1
    if [ $? -eq 0 ]; then
        log_result "$test" "passed" ""
        echo "$test passed"
    else
        log_result "$test" "failed" "$(cat /tmp/test_output)"
        echo "$test failed"
        FAILED=1
    fi
done

# Stop BentoML service
echo "Stopping BentoML service..."
docker exec mlflow_tracking-bentoml-1 pkill -f "bentoml serve"

# Output summary
echo "Test Summary:"
jq '.' "$RESULTS_FILE"

# Exit with status
if [ $FAILED -eq 0 ]; then
    echo "All tests passed"
    exit 0
else
    echo "Some tests failed"
    exit 1
fi
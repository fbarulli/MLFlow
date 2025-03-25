#!/bin/bash

# This script helps diagnose Airflow configuration issues, particularly with CSRF

echo "===== Airflow Environment Diagnostics ====="
echo "Current directory: $(pwd)"
echo "AIRFLOW_HOME: $AIRFLOW_HOME"

# Check if airflow.cfg exists
if [ -f "./airflow.cfg" ]; then
    echo "✅ airflow.cfg exists"
    
    # Extract important security settings
    echo "==== Security Configuration ===="
    grep "secret_key" ./airflow.cfg
    grep "enable_proxy_fix" ./airflow.cfg
    grep "authenticate" ./airflow.cfg
    grep "auth_backend" ./airflow.cfg
    
    # Check for web_server_host and allowed_origins
    echo "==== Network Configuration ===="
    grep "web_server_host" ./airflow.cfg
    grep "allowed_origins" ./airflow.cfg
    
    # Check webserver settings
    echo "==== Webserver Configuration ===="
    grep "base_url" ./airflow.cfg
    grep "web_server_worker_timeout" ./airflow.cfg
else
    echo "❌ airflow.cfg not found in current directory"
fi

# Check if the webserver is actually running
WEB_PID=$(pgrep -f "airflow webserver")
if [ -n "$WEB_PID" ]; then
    echo "✅ Webserver is running with PID: $WEB_PID"
else
    echo "❌ Webserver is not running"
fi

# Test connection to webserver
echo "==== Testing connection to webserver ===="
curl -s -I http://localhost:8080 | head -n 10

echo "==== Done ===="

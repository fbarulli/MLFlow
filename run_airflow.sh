#!/bin/bash

# Enable error tracing
set -e

# Use Python script to set up Airflow first
echo "Running Airflow setup script..."
python setup_airflow.py

# Set and export AIRFLOW_HOME to a relative path consistently
export AIRFLOW_HOME="."
echo "AIRFLOW_HOME set to: $AIRFLOW_HOME (relative path)"

# Kill any existing Airflow processes
echo "Stopping any running Airflow processes..."
kill -9 $(ps aux | grep '[a]irflow' | awk '{print $2}') 2>/dev/null || true
sudo pkill -9 gunicorn 2>/dev/null || true
sudo netstat -tulpn | grep 8080 && echo "WARNING: Port 8080 is still in use" || echo "Port 8080 is free"

# Wait for processes to terminate
sleep 2
echo "All previous Airflow processes terminated."

# Enhanced database initialization and verification
echo "Verifying database initialization..."
DB_FILE="./airflow.db"
if [ ! -f "$DB_FILE" ] || [ ! -s "$DB_FILE" ]; then
    echo "Database file missing or empty. Initializing..."
    airflow db init
fi

# Check if database is properly initialized by verifying tables
echo "Verifying database tables..."
sqlite3 ./airflow.db "SELECT name FROM sqlite_master WHERE type='table' AND name='dag'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Database tables incomplete. Performing full initialization..."
    airflow db init
    airflow db upgrade
fi

# Verify admin user exists - This is now the single place for user creation
echo "Verifying admin user..."
ADMIN_EXISTS=$(airflow users list | grep -c "admin" || true)
if [ "$ADMIN_EXISTS" -eq 0 ]; then
    echo "Admin user missing. Creating..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname Admin \
        --role Admin \
        --email admin@example.com \
        --password admin
else
    echo "✅ Admin user verified."
fi

# Check if required directories exist, create only if missing
[ -d "./logs" ] || mkdir -p ./logs
[ -d "./plugins" ] || mkdir -p ./plugins
echo "Airflow directories verified."

# Do NOT reset configuration here - use what setup_airflow.py configured

# Print configuration for debugging
echo "Current database connection: $(airflow config get-value core sql_alchemy_conn)"
echo "CSRF enabled: $(airflow config get-value webserver wtf_csrf_enabled)"
echo "Webserver base URL: $(airflow config get-value webserver base_url)"
echo "Secret key configured: $(airflow config get-value webserver secret_key | cut -c1-5)..."

# Create connections if they don't exist
echo "Verifying default connections..."
airflow connections list | grep -q "fs_default" || airflow connections create-default-connections

# Start services with proper environment variables
echo "Starting Airflow scheduler..."
nohup airflow scheduler > ./logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler started with PID: $SCHEDULER_PID"

sleep 3  # Give scheduler time to start

echo "Starting Airflow webserver..."
nohup airflow webserver --port 8080 > ./logs/webserver.log 2>&1 &
WEBSERVER_PID=$!
echo "Webserver started with PID: $WEBSERVER_PID"

# Verify the processes are running
sleep 5
if ps -p $SCHEDULER_PID > /dev/null; then
    echo "✅ Scheduler is running."
else
    echo "❌ ERROR: Scheduler failed to start. Check logs/scheduler.log for details."
    tail -20 ./logs/scheduler.log
fi

if ps -p $WEBSERVER_PID > /dev/null; then
    echo "✅ Webserver is running."
    echo "Testing webserver connection..."
    curl -s -I http://localhost:8080 | head -5 || echo "Failed to connect to webserver"
else
    echo "❌ ERROR: Webserver failed to start. Check logs/webserver.log for details."
    tail -20 ./logs/webserver.log
fi

echo "Airflow services started. Access the web UI at http://localhost:8080"
echo "Username: admin, Password: admin"
echo "Monitor logs with: tail -f ./logs/webserver.log ./logs/scheduler.log"
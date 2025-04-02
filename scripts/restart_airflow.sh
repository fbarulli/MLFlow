#!/bin/bash

# Stop any running Airflow processes
echo "Stopping Airflow processes..."
pkill -f airflow

# Wait for processes to stop
sleep 2

# Check if PostgreSQL is running, start if not
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "Starting PostgreSQL..."
    sudo service postgresql start
    sleep 2
fi

# Run setup script to ensure PostgreSQL and Airflow are configured
echo "Running Airflow setup..."
python setup_airflow.py

# Start Airflow webserver in background
echo "Starting Airflow webserver..."
airflow webserver -D

# Wait for webserver to initialize
sleep 5

# Start Airflow scheduler in background
echo "Starting Airflow scheduler..."
airflow scheduler -D

# Wait a moment
sleep 2

# Check if processes are running
echo "Checking Airflow processes..."
ps aux | grep "airflow" | grep -v "grep"

echo "Airflow restart complete. Web interface available at http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
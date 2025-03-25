#!/bin/bash

# Function to kill processes on port 8793
cleanup() {
    echo "Caught Ctrl+C, killing processes on port 8793..."
    sudo fuser -k 8793/tcp
    exit 0
}

# Trap SIGINT (Ctrl+C)
trap cleanup INT

# Start your Airflow process (adjust command as needed)
airflow webserver -p 8793 &

# Wait for the process to finish
wait
#!/bin/bash

# Get today's date in YYYY-MM-DD format
TODAY=$(date +%Y-%m-%d)

# List of directories that Airflow needs access to
DIRECTORIES=(
    "dags"
    "logs"
    "plugins"
    "data_storage"
    "mlruns"
)

# Create all required Airflow log directories including date-specific ones
mkdir -p "logs/scheduler/${TODAY}"
mkdir -p "logs/scheduler/$(date -d 'tomorrow' +%Y-%m-%d)"  # Also create tomorrow's directory
mkdir -p "logs/dag_processor_manager/${TODAY}"
mkdir -p "logs/webserver/${TODAY}"
mkdir -p "logs/worker/${TODAY}"

# Create directories if they don't exist and set correct permissions
for dir in "${DIRECTORIES[@]}"; do
    # Create directory if it doesn't exist
    mkdir -p "$dir"
    
    # Set ownership to the current user
    sudo chown -R $(id -u):$(id -g) "$dir"
    
    # Set directory permissions to 777 (rwxrwxrwx)
    # This ensures all users (including the airflow user) have full access
    chmod -R 777 "$dir"
done

echo "Directory setup complete. The following directories are ready for Airflow:"
echo -e "\nLogs directory structure:"
ls -R logs/
echo -e "\nMain directories:"
ls -l "${DIRECTORIES[@]}"
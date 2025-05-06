#!/bin/bash

# Stop and remove all Docker containers
echo "Stopping and removing all Docker containers..."
docker-compose down --remove-orphans

# Remove all files except .py files
echo "Removing all files except .py files..."
find . ! -name "*.py" -type f -delete

# Create necessary directories
echo "Setting up directories..."
mkdir -p mlruns outputs outputs/tests
chmod -R 777 mlruns outputs outputs/tests

# Build and run Docker Compose
echo "Building and running Docker Compose..."
docker-compose up --build
#!/bin/bash

# Stop and remove all Docker containers
echo "Stopping and removing all Docker containers..."
docker-compose down --remove-orphans
docker ps -a -q | xargs -r docker rm -f

echo "Removing unused Docker resources..."
docker system prune -a -f

# Remove all Docker images
echo "Removing all Docker images..."
docker images -q | sort -u | xargs -r docker rmi -f

# Remove all Docker networks (except default ones)
echo "Removing all Docker networks..."
docker network ls -q | grep -v "bridge\|host\|none" | xargs -r docker network rm

# Remove all Docker volumes
echo "Removing all Docker volumes..."
docker volume ls -q | xargs -r docker volume rm -f

# Create necessary directories
echo "Setting up directories..."
mkdir -p mlruns outputs outputs/tests
chmod -R 777 mlruns outputs outputs/tests

# Ensure test scripts are executable
echo "Setting script permissions..."
chmod +x scripts/*.sh

# Build and run Docker Compose
echo "Building and running Docker Compose..."
docker-compose up --build
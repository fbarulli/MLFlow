#!/bin/bash

# Remove orphan containers
docker-compose -f mlflow_tracking/docker-compose.yml down --remove-orphans

# Aggressively prune Docker images
docker system prune -a --volumes -f

# Rebuild and start the Docker Compose deployment
docker-compose -f mlflow_tracking/docker-compose.yml up -d --build --force-recreate

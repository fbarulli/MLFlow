#!/bin/bash
# ~/MLFlow/scripts/build_docker.sh
set -e
echo "Building Docker container for weather-collector:light..."
docker build -t weather-collector:light -f Dockerfile .

echo "Building Docker container for weather-monitor:light..."
docker build -t weather-monitor:light -f Dockerfile.monitor .
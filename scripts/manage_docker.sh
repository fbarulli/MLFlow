#!/bin/bash
set -e

function show_help() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  start     Start all services"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  logs      Show logs from all services"
    echo "  clean     Stop services and remove volumes"
    echo "  ps        Show service status"
    exit 1
}

# Determine which docker compose command to use
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "Neither docker-compose nor docker compose is available"
    exit 1
fi

# Export random secret key if not set
if [ -z "$AIRFLOW_SECRET_KEY" ]; then
    export AIRFLOW_SECRET_KEY=$(openssl rand -hex 32)
fi

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

case "$1" in
    start)
        echo "Starting Airflow services..."
        $DOCKER_COMPOSE up -d
        echo "Services are starting. View status with: $DOCKER_COMPOSE ps"
        echo "Web UI will be available at: http://localhost:8080"
        echo "Username: admin"
        echo "Password: admin"
        echo "MLflow UI will be available at: http://localhost:5000"
        ;;
    stop)
        echo "Stopping services..."
        $DOCKER_COMPOSE down
        ;;
    restart)
        echo "Restarting services..."
        $DOCKER_COMPOSE down
        $DOCKER_COMPOSE up -d
        ;;
    logs)
        $DOCKER_COMPOSE logs -f
        ;;
    clean)
        echo "Stopping services and removing volumes..."
        $DOCKER_COMPOSE down -v
        ;;
    ps)
        $DOCKER_COMPOSE ps
        ;;
    *)
        show_help
        ;;
esac
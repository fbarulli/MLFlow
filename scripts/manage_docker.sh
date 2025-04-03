#!/bin/bash
set -e

function show_help() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  start     Start all services"
    echo "  stop      Stop all services"
    echo "  restart   Rebuild images and restart services"
    echo "  build     Build images (use --no-cache for fresh build)"
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
echo "Using: $DOCKER_COMPOSE"

# Export random secret key if not set
if [ -z "$AIRFLOW_SECRET_KEY" ]; then
    export AIRFLOW_SECRET_KEY=$(openssl rand -hex 32)
fi

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

COMMAND=$1
shift # Remove the command from arguments

case "$COMMAND" in
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
    build)
        echo "Building images..."
        # Pass remaining arguments (like --no-cache) to build command
        $DOCKER_COMPOSE build "$@"
        ;;
    restart)
        echo "Rebuilding images and restarting services..."
        $DOCKER_COMPOSE down
        $DOCKER_COMPOSE build # Build without cache by default on restart
        $DOCKER_COMPOSE up -d
        ;;
    logs)
        $DOCKER_COMPOSE logs -f "$@" # Pass extra args like service name
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
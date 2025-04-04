# MLFlow Project with Airflow

This project uses Docker to run Apache Airflow and MLflow services in a reproducible environment.

## Prerequisites

- Docker
- Docker Compose
- sudo privileges (for setting up directory permissions)

## Directory Setup

Before starting the services, you need to set up the required directories with correct permissions:

```bash
# Make the setup script executable
chmod +x scripts/setup_directories.sh

# Run the setup script (requires sudo for permission changes)
./scripts/setup_directories.sh
```

This script creates and configures the following directories with appropriate permissions:
- `dags/`: For Airflow DAG files
- `logs/`: For Airflow logs
- `plugins/`: For Airflow plugins
- `data_storage/`: For storing collected data
- `mlruns/`: For MLflow tracking

## Quick Start

1. Set up directories (first time only):
```bash
chmod +x scripts/setup_directories.sh
./scripts/setup_directories.sh
```

2. Start all services:
```bash
chmod +x scripts/manage_docker.sh
./scripts/manage_docker.sh start
```

2. Access the services:
- Airflow UI: http://localhost:8080
  - Username: admin
  - Password: admin
- MLflow UI: http://localhost:5000

## Project Structure

```
.
├── dags/                  # Airflow DAG files
├── data_storage/          # Data storage directory
├── logs/                  # Airflow logs
├── mlruns/               # MLflow experiment tracking
├── plugins/              # Airflow plugins
├── Dockerfile.airflow    # Custom Airflow image with MLflow
├── docker-compose.yml    # Service orchestration
└── scripts/
    └── manage_docker.sh  # Service management script
```

## Management Commands

```bash
./scripts/manage_docker.sh <command>
```

Available commands:
- `start`: Start all services
- `stop`: Stop all services
- `restart`: Restart all services
- `logs`: Show service logs
- `clean`: Stop services and remove volumes
- `ps`: Show service status

## Environment Variables

You can customize the setup using environment variables:
- `AIRFLOW_SECRET_KEY`: Custom secret key for Airflow (auto-generated if not set)

## Rebuilding Images

To rebuild the images after making changes:
```bash
./scripts/manage_docker.sh stop
docker compose build --no-cache
./scripts/manage_docker.sh start
```

## Service Dependencies

The setup includes:
1. PostgreSQL (Airflow metadata database)
2. Airflow Webserver
3. Airflow Scheduler
4. MLflow Server

## Troubleshooting

1. If you see authentication errors:
   - Stop all services: `./scripts/manage_docker.sh stop`
   - Clean volumes: `./scripts/manage_docker.sh clean`
   - Start fresh: `./scripts/manage_docker.sh start`

2. To view service logs:
   ```bash
   ./scripts/manage_docker.sh logs
   ```

3. To check service status:
   ```bash
   ./scripts/manage_docker.sh ps
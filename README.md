# MLFlow Project

This project combines MLflow, Airflow, and DVC for an end-to-end ML workflow.

## Overview

This project includes the following components:
- **Data Collection**: Scripts to fetch weather data from an API and store it in a CSV file.
- **Data Class**: Pydantic models to structure the weather data.
- **Logging**: Custom logging setup to log events to CSV files.
- **Airflow**: Orchestrates the workflow using DAGs.
- **DVC**: Manages data versioning.

## Setup

1. Clone this repository
2. Run the setup script to create the environment and initialize Airflow:

```bash
chmod +x setup_airflow.sh
```

## Running Airflow

Start Airflow services (scheduler and webserver):

```bash
conda activate mlflow
scripts/orchestrate.sh
```

Access the Airflow UI at http://localhost:8080
- Username: admin
- Password: admin

Stop Airflow services:

```bash
pkill -f "airflow scheduler" && pkill -f "airflow webserver"
```

## Scripts Overview

- **setup_airflow.sh**: Sets up Airflow by configuring the environment and initializing the database.
- **orchestrate.sh**: Orchestrates the setup process by running various setup scripts in sequence.
- **shutdown_services.sh**: Shuts down any lingering Airflow and MLFlow services.
- **configure_git.sh**: Configures Git with user details if not already set.
- **configure_dvc.sh**: Configures DVC remote storage.
- **build_docker.sh**: Builds the Docker container for the project.
- **setup_evidently.sh**: Installs the Evidently package.
- **start_mlflow.sh**: Starts the MLFlow server.
- **start_webserver.sh**: Starts the Airflow webserver.
- **start_scheduler.sh**: Starts the Airflow scheduler.

## DVC Configuration

Set up DVC remote:

```bash
dvc remote add -d myremote https://dagshub.com/fbarulli/MLFlow.dvc
dvc remote modify myremote --local auth basic
dvc remote modify myremote --local user fbarulli
dvc remote modify myremote --local password dhp_yourDagsHubTokenHere
```

## Troubleshooting

If you encounter issues with Airflow:

1. Check the logs in `logs/webserver.log` and `logs/scheduler.log`
2. Run `./check_airflow_config.sh` to diagnose configuration issues
3. Try restarting with `./stop_airflow.sh` followed by `./run_airflow.sh`
# MLFlow Project

This project combines MLflow, Airflow, and DVC for an end-to-end ML workflow.

## Setup

1. Clone this repository
2. Run the setup script to create the environment and initialize Airflow:

```bash
./setup_project.sh
```

## Running Airflow

Start Airflow services (scheduler and webserver):

```bash
conda activate mlflow
./run_airflow.sh
```

Access the Airflow UI at http://localhost:8080
- Username: admin
- Password: admin

Stop Airflow services:

```bash
./stop_airflow.sh
```

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
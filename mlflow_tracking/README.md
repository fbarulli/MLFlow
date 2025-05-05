# mlflow

## Overview

This project trains and evaluates machine learning models (Logistic Regression and Random Forest) to classify wine quality. It uses MLflow for experiment tracking and model management, and KaggleHub to load the dataset. Docker and Docker Compose are used to containerize the application and its dependencies.

## Project Structure

*   `wine_work.py`: Python script that trains and evaluates the machine learning models.
*   `docker-compose.yml`: Defines the services for the MLflow server and the application.
*   `Dockerfile.app`: Dockerfile for building the application image.
*   `Dockerfile.server`: Dockerfile for building the MLflow server image.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `run.sh`: Shell script to stop/remove Docker containers/images/networks/volumes, create the `mlruns` directory, set script permissions, and build/run the Docker Compose project.
*   `mlruns`: Directory where MLflow stores the experiment runs and model artifacts.
*   `outputs`: Directory where the model predictions, plots, and test results are saved.
*   `scripts`: Directory containing shell scripts for testing and orchestration.

## Scripts

The `scripts` directory contains the following shell scripts:

*   `orchestrate_all.sh`: Orchestrates the entire process, including starting Docker services, updating model metadata, converting MLflow models to BentoML, and running tests.
*   `test_all.sh`: Runs all tests using pytest in the BentoML container.
*   `test_api.sh`: Runs API endpoint tests using pytest in the BentoML container.
*   `test_errors.sh`: Runs error handling tests using pytest in the BentoML container.
*   `test_orchestrator.sh`: Runs the test orchestrator, which executes other test scripts and logs the results.
*   `test_performance.sh`: Runs performance tests using pytest in the BentoML container.
*   `test_validation.sh`: Runs input validation tests using pytest in the BentoML container.

## Docker and Docker Compose

Docker is a platform for building, shipping, and running applications in containers. Docker Compose is a tool for defining and running multi-container Docker applications.

In this project, Docker is used to containerize the MLflow server and the application. Docker Compose is used to define the services for the MLflow server and to manage the dependencies between them.
The `docker-compose.yml` file defines the following services:

*   `mlflow-server`: The MLflow server, which is built from `Dockerfile.server`. This service manages MLflow experiments and model artifacts.
*   `bentoml`: The BentoML service, which is built from the base image and converts MLflow models to BentoML for serving.
*   `app`: The application, which is built from `Dockerfile.app` and depends on the `mlflow-server`. This service trains and evaluates machine learning models.

The `Dockerfile.app` and `Dockerfile.server` files define the steps for building the Docker images for the application and the MLflow server, respectively. The `Dockerfile.bentoml` file defines the steps for building the BentoML service.

The `run.sh` script is used to stop/remove Docker containers/images/networks/volumes, create the `mlruns` directory, set script permissions, and build/run the Docker Compose project.

### Docker and Docker Compose Workflow

See [docker_flowchart.md](docker_flowchart.md) for a detailed flowchart of the Docker and Docker Compose workflow.

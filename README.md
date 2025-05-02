# mlflow

## Overview

This project trains and evaluates machine learning models (Logistic Regression and Random Forest) to classify wine quality. It uses MLflow for experiment tracking and model management, and KaggleHub to load the dataset. Docker and Docker Compose are used to containerize the application and its dependencies.

## Project Structure

*   `wine_work.py`: Python script that trains and evaluates the machine learning models.
*   `docker-compose.yml`: Defines the services for the MLflow server and the application.
*   `Dockerfile.app`: Dockerfile for building the application image.
*   `Dockerfile.server`: Dockerfile for building the MLflow server image.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `run.sh`: Shell script to stop/remove Docker containers/images/networks/volumes, create the `mlruns` directory, and build/run the Docker Compose project.
*   `mlruns`: Directory where MLflow stores the experiment runs and model artifacts.
*   `outputs`: Directory where the model predictions and plots are saved.

## Docker and Docker Compose

Docker is a platform for building, shipping, and running applications in containers. Docker Compose is a tool for defining and running multi-container Docker applications.

In this project, Docker is used to containerize the MLflow server and the application. Docker Compose is used to define the services for the MLflow server and to manage the dependencies between them.

The `docker-compose.yml` file defines two services:

*   `mlflow-server`: The MLflow server, which is built from `Dockerfile.server`.
*   `app`: The application, which is built from `Dockerfile.app` and depends on the `mlflow-server`.

The `Dockerfile.app` and `Dockerfile.server` files define the steps for building the Docker images for the application and the MLflow server, respectively.

The `run.sh` script is used to build and run the Docker Compose project.

### Docker and Docker Compose Workflow

See [docker_flowchart.md](docker_flowchart.md) for a detailed flowchart of the Docker and Docker Compose workflow.

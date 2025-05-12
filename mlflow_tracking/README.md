# Wine Quality Classification with MLflow

This project demonstrates how to use MLflow to train and deploy a wine quality classification model.

## Overview

The project uses a dataset from Kaggle to train a model that predicts the quality of wine based on various features. The project uses MLflow to track the training process, log the model, and deploy the model to a BentoML service.

## Components

*   **`MLproject`:** Defines the MLflow project, including the conda environment and entry point.
*   **`conda.yaml`:** Defines the conda environment for the project.
*   **`Dockerfile.app`:** Defines the Docker image for the wine quality classification app.
*   **`Dockerfile.mlflow`:** Defines the Docker image for the MLflow server.
*   **`requirements.txt`:** Defines the pip requirements for the project.
*   **`run_docker.sh`:** A script to build and run the Docker containers.
*   **`mlflow/wine_work.py`:** The main script for training and logging the model.
*   **`mlflow/model_metadata.py`:** A script to update the metadata of the models in the MLflow Model Registry.
*   **`bentoml/service.py`:** Defines the BentoML service for deploying the model.

## Running the project

1.  Build and start the Docker containers:

    ```bash
    bash mlflow_tracking/run_docker.sh
    ```

2.  View the status of the apps:

    ```bash
    docker-compose -f mlflow_tracking/docker-compose.yml ps
    ```

3.  View the logs of the wine\_app service:

    ```bash
    docker-compose -f mlflow_tracking/docker-compose.yml logs wine_app
    ```

## MLflow

This project uses MLflow to track the training process, log the model, and deploy the model to a BentoML service.

*   **MLflow Projects:** The `MLproject` file defines the project, including the conda environment and entry point.
*   **MLflow Models:** The `mlflow.sklearn.log_model` function is used in `mlflow_tracking/mlflow/wine_work.py` to log the trained models.
*   **MLflow Model Registry:** The `mlflow.sklearn.log_model` function also registers the models in the MLflow Model Registry, and the `transition_model_to_staging` function is used to transition the models to the Staging stage.

The `mlflow/model_metadata.py` script is run after the `mlflow/wine_work.py` script to update the metadata of the models in the MLflow Model Registry.
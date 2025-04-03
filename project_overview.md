# MLFlow Project Overview

This document provides an explanation of each file in the MLFlow project repository.

## Configuration Files

### `/home/ubuntu/MLFlow/webserver_config.py`
This file contains the default configuration for the Airflow webserver. It defines authentication types, security settings, and UI theme configurations. The file is based on Flask-AppBuilder and allows for various authentication methods like DB, LDAP, OAuth, etc. Currently configured to use DB authentication (`AUTH_TYPE = AUTH_DB`).

### `/home/ubuntu/MLFlow/airflow.cfg`
The main Airflow configuration file that sets up core components such as:
- Directories for DAGs, plugins, and logs
- Database connection (PostgreSQL)
- Webserver settings
- Executor type (LocalExecutor)
- Scheduler settings
- Security configurations

### `/home/ubuntu/MLFlow/environment.yml`
Conda environment configuration file that specifies all project dependencies. It creates a 'mlflow' environment with Python 3.12 and includes numerous packages like:
- Apache Airflow and its providers
- DVC (Data Version Control)
- MLflow
- Data science tools (pandas, scikit-learn, etc.)
- HTTP request libraries
- Utilities and other dependencies

### `/home/ubuntu/MLFlow/.gitignore`
Standard Git ignore file that specifies which files and directories should not be tracked by version control, particularly data files and directories.

### `/home/ubuntu/MLFlow/.dvcignore`
Specifies patterns of files that DVC should ignore, which can improve performance during data versioning operations.

## Setup Scripts

### `/home/ubuntu/MLFlow/setup_airflow.py`
A comprehensive Python script that automates the setup and configuration of Airflow within the project. It:
1. Sets up PostgreSQL database for Airflow
2. Configures environment variables and creates the airflow.cfg file
3. Initializes the Airflow database
4. Creates an admin user
5. Implements error handling and helpful logging
6. Provides instructions for starting Airflow services after setup

## Data Storage

### `/home/ubuntu/MLFlow/data_storage.dvc`
DVC tracking file for the data_storage directory. This enables version control of data while keeping the actual data files out of Git. Contains metadata about the data directory and its contents.

## Docker Configuration

### `/home/ubuntu/MLFlow/Dockerfile`
Defines a Docker container for data collection. Based on Python 3.12-slim, it installs necessary packages (pandas, requests, tqdm, pydantic) and sets up a non-root user for security. The container is configured to run the weather data collection script.

### `/home/ubuntu/MLFlow/Dockerfile.monitor`
Defines a Docker container for monitoring data. Similar to the main Dockerfile but adds evidently and mlflow packages for data monitoring and ML operations. Configured to run the monitoring script.

## Log and PID Files

### `/home/ubuntu/MLFlow/airflow-webserver.log`
Log file for the Airflow webserver, containing information about server operations, errors, and signals received.

### `/home/ubuntu/MLFlow/airflow-webserver.out`
Standard output log from the Airflow webserver process.

### `/home/ubuntu/MLFlow/airflow-webserver.err`
Standard error log from the Airflow webserver process.

### `/home/ubuntu/MLFlow/airflow-webserver.pid`
Contains the process ID of the running Airflow webserver.

## Documentation

### `/home/ubuntu/MLFlow/README.md`
Main project documentation that provides:
- Project overview and components
- Setup instructions
- Commands for running Airflow
- Script descriptions
- DVC configuration guidance
- Troubleshooting tips

### `/home/ubuntu/MLFlow/previous_errors.md`
Detailed documentation of previously encountered errors and their solutions, including:
- Authentication issues
- Task signal handling problems
- SQLite concurrency issues
- DVC and Git integration problems
- Heartbeat timeout problems

## Airflow DAGs

### `/home/ubuntu/MLFlow/dags/weather_dag.py`
This DAG orchestrates the weather data collection workflow. It runs every minute and consists of two tasks:
1. `collect_weather`: Uses DockerOperator to run a containerized Python script that collects weather data
2. `version_data`: Uses a custom DVCHook to version the collected data with DVC

Key components:
- Validates AIRFLOW_HOME environment variable
- Mounts data storage directory to the Docker container
- Handles data versioning with DVC
- Implements proper error handling

### `/home/ubuntu/MLFlow/dags/weather_monitor_dag.py`
A monitoring DAG that runs every minute to check the status of the weather data collection process. It uses a PythonOperator with a special monitoring function that:
- Implements proper signal handling
- Uses smaller sleep intervals for better responsiveness
- Logs progress every 10 seconds
- Includes comprehensive error handling

This DAG was designed specifically to address task termination issues experienced with SIGTERM signals.

### `/home/ubuntu/MLFlow/dags/airflow_custom_hooks/dvc_hook.py`
A custom Airflow hook for DVC operations. It:
- Initializes DVC without Git integration
- Sets up remote DVC repositories
- Handles authentication for DVC operations
- Provides methods for adding and pushing data to DVC
- Implements retry logic for push operations
- Offers proper logging and error handling

### `/home/ubuntu/MLFlow/dags/setup.py`
Installation script for the custom Airflow hooks, allowing them to be properly imported within the Airflow environment.

## Project Structure

The repository appears to have additional directories that would contain:
- `dags/`: Airflow DAG definitions
- `logs/`: Log files
- `plugins/`: Airflow plugins
- `data_storage/`: Versioned data files
- `src/`: Source code for data collection and monitoring 

## Requirements

### `/home/ubuntu/MLFlow/requirements.txt`
A comprehensive list of Python package dependencies with specific versions for reproducing the project environment. Includes all necessary libraries for Airflow, MLflow, DVC, data science operations, and more.

## Data Monitoring

The project implements a comprehensive data monitoring approach that combines several components:

### Monitoring DAG

The `weather_monitor_dag.py` DAG is responsible for regularly checking the quality and characteristics of collected weather data. This monitoring runs every minute and uses a `PythonOperator` with a dedicated `monitor_function()` that:

1. Performs incremental monitoring through 60-second cycles
2. Provides visibility into the monitoring process with logging every 10 seconds
3. Implements proper exception handling and graceful termination
4. Is designed to handle SIGTERM signals properly to avoid task termination issues

### Docker-based Monitoring

The `Dockerfile.monitor` defines a specialized container for data monitoring that includes:

- Python 3.12 with pandas for data manipulation
- Evidently AI for data quality and drift monitoring
- MLflow for experiment tracking and metric logging

This container runs the `src.monitoring.monitor_script` module, which likely:
- Loads the latest weather data
- Calculates data quality metrics (completeness, validity)
- Detects distribution shifts and anomalies
- Logs monitoring results to MLflow

### Integration with MLflow

The monitoring results are tracked in MLflow, which enables:
- Historical tracking of data quality metrics
- Visualization of metric trends over time
- Correlation of data quality with model performance
- Setting alerts for significant data drift

### Benefits of this Approach

1. **Separation of Concerns**: The monitoring code is isolated in a dedicated container
2. **Reliability**: The monitoring process is designed to be resilient to failures
3. **Visibility**: Regular logging provides insights into the monitoring process
4. **Integration**: Results are tracked in MLflow for comprehensive oversight
5. **Orchestration**: Airflow ensures the monitoring runs consistently

By implementing this monitoring pipeline, the project can detect data quality issues, schema changes, and distribution shifts early, preventing downstream impacts on ML models.

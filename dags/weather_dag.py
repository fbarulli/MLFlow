#dags/weather_dag.py
import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
from pathlib import Path
from docker.types import Mount
from mlflow_custom_hooks.dvc_hook import DVCHook  # Updated import path

def validate_airflow_home():
    """Validate AIRFLOW_HOME environment variable and path."""
    if "AIRFLOW_HOME" not in os.environ:
        raise AirflowException("AIRFLOW_HOME environment variable is not set")
    
    airflow_home = Path(os.environ["AIRFLOW_HOME"])
    if not airflow_home.exists():
        raise AirflowException(f"AIRFLOW_HOME path does not exist: {airflow_home}")
    
    required_dirs = ["dags", "logs", "data_storage"]
    for dir_name in required_dirs:
        if not (airflow_home / dir_name).exists():
            raise AirflowException(f"Required directory '{dir_name}' not found in AIRFLOW_HOME")
    
    return airflow_home

def version_weather_data(run_id):
    """Version weather data using DVC hook"""
    # The hook now manages its own cwd based on AIRFLOW_HOME
    hook = DVCHook()
    hook.add_and_push(
        filepath="data_storage/raw/weather.csv",
        commit=False, # Commit is ignored anyway
        message=f"Update weather data for run {run_id}"
    )

# Get project paths
project_root = validate_airflow_home()
data_storage = project_root / "data_storage" / "raw"
log_folder = project_root / "logs"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'weather_data_collection',
    default_args=default_args,
    description='Collect weather data every minute',
    schedule='* * * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["data"],
    max_active_runs=1,
) as dag:
    collect_weather = DockerOperator(
        task_id='collect_weather_data',
        image='weather-collector:light',
        api_version='auto',
        auto_remove="force",
        command=["python", "-m", "src.data_collection.weather_script"],
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[Mount(target='/data', source=str(data_storage), type='bind')],
        working_dir='/app',
    )

    version_data = PythonOperator(
        task_id='version_data',
        python_callable=version_weather_data,
        op_kwargs={
            'run_id': '{{ run_id }}'
        }
    )

    collect_weather >> version_data
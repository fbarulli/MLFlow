# dags/weather_dag.py
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
from pathlib import Path
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

project_root = Path(__file__).resolve().parent.parent
data_storage = project_root / "data_storage"

with DAG(
    'weather_data_collection',
    default_args=default_args,
    description='Collect weather data every minute',
    schedule='* * * * *',  # Updated to schedule
    start_date=datetime(2025, 3, 25, 10, 0),
    catchup=False,
    tags=["data"]
) as dag:
    collect_weather = DockerOperator(
        task_id='collect_weather_data',
        image='weather-collector:light',
        api_version='auto',
        auto_remove=True,
        command=["python", "-m", "src.data_collection.weather_script"],
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[Mount(target='/data_storage', source=str(data_storage), type='bind')],
    )

    collect_weather
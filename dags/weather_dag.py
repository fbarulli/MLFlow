from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime, timedelta

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
    description='Collect weather data hourly',
    schedule_interval='@hourly',
    start_date=datetime(2025, 3, 24),
    catchup=False,
) as dag:
    collect_weather = DockerOperator(
        task_id='collect_weather_data',
        image='weather-collector:light',
        api_version='auto',
        auto_remove=True,
        command=["python", "-m", "src.data_collection.weather_script"],
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        volumes=['MLFlow/data_storage:MLFlow/data_storage'],
    )

    collect_weather
#dags/weather_dag.py
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
from docker.types import Mount
from airflow_custom_hooks.dvc_hook import DVCHook

def version_weather_data(project_root, run_id):
    """Version weather data using DVC hook"""
    hook = DVCHook()
    hook.add_and_push(
        filepath="data_storage/raw/weather.csv",
        cwd=project_root,
        commit=True,
        message=f"Update weather data for run {run_id}"
    )

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

project_root = Path(__file__).resolve().parent.parent
data_storage = project_root / "data_storage" / "raw"
log_folder = project_root / "logs"

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
            'project_root': str(project_root),
            'run_id': '{{ run_id }}'
        }
    )

    collect_weather >> version_data
# dags/weather_monitor_dag.py
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
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
data_storage = project_root / "data_storage" / "raw"
log_folder = project_root / "logs"

with DAG(
    'weather_data_monitoring',
    default_args=default_args,
    description='Monitor weather data every minute',
    schedule='* * * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["data"],
    max_active_runs=1,
) as dag:
    
    setup_data_dir = BashOperator(
        task_id='setup_data_dir',
        bash_command=f'mkdir -p {data_storage}',
    )

    
    monitor_data = DockerOperator(
        task_id='monitor_data',
        image='weather-monitor:light',
        api_version='auto',
        auto_remove="force",
        command=["python", "-m", "src.data_monitoring.monitor_script"],  # Adjust command as needed
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[Mount(target='/data', source=str(data_storage), type='bind')],
        working_dir='/app',
    )

    
    setup_data_dir >> monitor_data
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
    description='Monitor weather data drift every 30 minutes',
    schedule='* * * * *',  # Every 30 minutes
    start_date=datetime(2025, 3, 25, 10, 0),
    catchup=False,
    tags=["monitoring"]
) as dag:
    clean_logs = BashOperator(
        task_id='clean_logs',
        bash_command="rm -rf {{ params.log_folder }}/dag_id=weather_data_monitoring/run_id=*",
        params={'log_folder': log_folder},
    )

    monitor_data = DockerOperator(
        task_id='monitor_data',
        image='weather-monitor:light',
        api_version='auto',
        auto_remove="force",
        command=["python", "-m", "src.monitoring.monitor_script"],
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[Mount(target='/data', source=str(data_storage), type='bind')],
        working_dir='/app'
    )

    clean_logs >> monitor_data
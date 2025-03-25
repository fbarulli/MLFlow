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
    'weather_data_collection',
    default_args=default_args,
    description='Collect weather data every minute',
    schedule='* * * * *',
    start_date=datetime(2025, 3, 25, 10, 0),
    catchup=False,
    tags=["data"]
) as dag:
    clean_logs = BashOperator(
        task_id='clean_logs',
        bash_command="rm -rf {{ params.log_folder }}/dag_id=weather_data_collection/run_id=*",
        params={'log_folder': log_folder},
    )

    collect_weather = DockerOperator(
        task_id='collect_weather_data',
        image='weather-collector:light',
        api_version='auto',
        auto_remove="force",
        command=["python", "-m", "src.data_collection.weather_script"],
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[Mount(target='/data', source=str(data_storage), type='bind')],
        working_dir='/app'
    )

    version_data = BashOperator(
        task_id='version_data',
        bash_command=f'cd {project_root} && dvc add data_storage/raw/weather.csv && dvc push && git add data_storage/raw/weather.csv.dvc && git commit -m "Update weather data" --allow-empty && git push',    )

    clean_logs >> collect_weather >> version_data
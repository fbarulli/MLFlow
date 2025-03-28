#dags/weather_dag.py
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
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["data"],
    max_active_runs=1,
) as dag:
    
    configure_dvc = BashOperator(
        task_id='configure_dvc',
        bash_command='bash /home/ubuntu/MLFlow/scripts/configure_dvc.sh',
    )

    
    setup_data_dir = BashOperator(
        task_id='setup_data_dir',
        bash_command=f'mkdir -p {data_storage}',
    )

    
    clean_logs = BashOperator(
        task_id='clean_logs',
        bash_command="rm -rf {{ params.log_folder }}/dag_id={{ dag.dag_id }}/run_id={{ run_id }}",
        params={'log_folder': str(log_folder)},
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
        working_dir='/app',
    )

    
    version_data = BashOperator(
        task_id='version_data',
        bash_command=f'''
        cd {project_root} && \
        if [ -f {data_storage}/weather.csv ]; then
            dvc add data_storage/raw/weather.csv && \
            dvc push && \
            git add data_storage/raw/weather.csv.dvc && \
            git commit -m "Update weather data for run {{ run_id }}" --allow-empty && \
            git push
        else
            echo "weather.csv not found, skipping versioning."
            exit 1
        fi
        ''',
    )

    
    configure_dvc >> setup_data_dir >> clean_logs >> collect_weather >> version_data
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import time

def monitor_function(**context):
    """Monitor data with proper signal handling."""
    try:
        print("Starting monitoring...")
        # Use a loop with smaller sleep intervals to be more responsive to signals
        for i in range(60):
            time.sleep(1)
            if i % 10 == 0:  # Log every 10 seconds to show progress
                print(f"Monitoring... {i+1}/60 seconds")
        print("Monitoring complete")
        return "Monitoring successful"
    except Exception as e:
        print(f"Error during monitoring: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    'weather_data_monitoring',
    default_args=default_args,
    description='Monitor weather data collection',
    schedule='* * * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['monitoring']
) as dag:

    monitor_data = PythonOperator(
        task_id='monitor_data',
        python_callable=monitor_function,
        provide_context=True,
        execution_timeout=timedelta(minutes=2),  # Give it more time than needed
        retries=1,
        retry_delay=timedelta(minutes=1)
    )
# MLFlow

# DVC:
dvc remote add -d myremote https://dagshub.com/fbarulli/MLFlow.dvc
dvc remote modify myremote --local auth basic
dvc remote modify myremote --local user fbarulli
dvc remote modify myremote --local password dhp_yourDagsHubTokenHere




# Airflow:
ps aux | grep airflow
kill -9 <scheduler_pid> <webserver_pid>
ps aux | grep airflow

kill -9 $(ps aux | grep '[a]irflow' | awk '{print $2}')

sudo pkill -9 gunicorn


Start Scheduler:
airflow scheduler &

Start WebServer:
airflow webserver -p 8080

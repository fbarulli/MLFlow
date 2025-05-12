
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends libopenblas-dev && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/mlflow/requirements.txt


RUN pip install --no-cache-dir -r /app/mlflow/requirements.txt


COPY mlflow/ /app/mlflow/

RUN chmod +x /app/mlflow/run.sh

ENTRYPOINT ["/app/mlflow/run.sh"]
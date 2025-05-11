FROM python:3.9-slim
WORKDIR /app
RUN rm -rf /app && mkdir -p /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY mlflow/ /app/mlflow/
RUN mkdir -p /app/outputs
CMD ["python", "mlflow/wine_work.py"]
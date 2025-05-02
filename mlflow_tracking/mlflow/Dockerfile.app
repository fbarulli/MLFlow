# Dockerfile.app
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow/wine_work.py .

CMD ["python", "wine_work.py"]
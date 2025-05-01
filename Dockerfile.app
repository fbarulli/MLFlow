FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wine_work.py .
COPY MLproject .

CMD ["mlflow", "run", ".", "--no-conda"]
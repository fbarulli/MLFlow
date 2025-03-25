FROM python:3.12-alpine

WORKDIR /app

COPY requirements.txt .
RUN apk add --no-cache gcc musl-dev linux-headers && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del gcc musl-dev linux-headers

COPY src/ src/

CMD ["python", "-m", "src.data_collection.weather_script"]
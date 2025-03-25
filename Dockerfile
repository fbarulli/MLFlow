# Dockerfile

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y && \
    pip install --no-cache-dir pandas requests tqdm pydantic && \
    apt-get clean && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
COPY src/ src/
CMD ["python", "-m", "src.data_collection.weather_script"]
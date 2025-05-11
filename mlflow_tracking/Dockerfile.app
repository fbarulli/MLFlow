# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install any system-level dependencies (e.g., for SciPy)
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas-dev && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the mlflow_tracking subdirectory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY mlflow/ /app/mlflow/

# Define the command to run your application
CMD ["python", "mlflow/wine_work.py"]
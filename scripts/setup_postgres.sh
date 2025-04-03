#!/bin/bash

# Install PostgreSQL
echo "Installing PostgreSQL..."
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

# Create database and user
echo "Creating Airflow database and user..."
sudo -u postgres psql << EOF
CREATE DATABASE airflow;
CREATE USER airflow WITH PASSWORD 'airflow';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
EOF

# Verify the setup
echo "Verifying PostgreSQL setup..."
sudo -u postgres psql -c "\l" | grep airflow

echo "PostgreSQL setup complete!"
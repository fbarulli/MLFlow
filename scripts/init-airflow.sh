#!/bin/bash
airflow db upgrade
airflow users create \
  --role Admin \
  --username admin \
  --email admin@example.com \
  --firstname admin \
  --lastname admin \
  --password admin || true
#plugins/setup.py
from setuptools import setup, find_packages

setup(
    name="mlflow_custom_hooks",  # Changed from airflow_custom_hooks
    version="0.1",
    packages=find_packages(),
    install_requires=[
        #'apache-airflow',
        'dvc',
    ],
    entry_points={
        'airflow.plugins': [
            'mlflow_plugin = mlflow_custom_hooks.plugin:MLFlowPlugin'
        ]
    },
)

from setuptools import setup, find_packages

setup(
    name="airflow_custom_hooks",  # Match the import name
    version="0.1",
    packages=find_packages(include=['airflow_custom_hooks*']),  # Explicitly include the package
    install_requires=[
        'apache-airflow',
        'dvc',
    ],
)
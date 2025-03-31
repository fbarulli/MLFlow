from setuptools import setup, find_packages

setup(
    name="airflow-custom-hooks",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'apache-airflow',
    ],
)
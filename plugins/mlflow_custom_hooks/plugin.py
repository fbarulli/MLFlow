from airflow.plugins_manager import AirflowPlugin
from mlflow_custom_hooks.dvc_hook import DVCHook

class MLFlowPlugin(AirflowPlugin):
    name = 'mlflow_plugin'
    hooks = [DVCHook]
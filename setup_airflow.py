# setup_airflow.py
import os
from pathlib import Path
import subprocess
import sys

project_root = Path(__file__).resolve().parent
airflow_home = project_root
dags_folder = project_root / "dags"

os.environ["AIRFLOW_HOME"] = str(airflow_home)

try:
    subprocess.run(["airflow", "--version"], check=True, capture_output=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Airflow not found. Installing airflow...")
    subprocess.run([sys.executable, "-m", "pip", "install", "apache-airflow"], check=True)
    subprocess.run(["airflow", "db", "migrate"], check=True)
    subprocess.run(["airflow", "connections", "create-default-connections"], check=True)

if not (airflow_home / "airflow.cfg").exists():
    subprocess.run(["airflow", "db", "migrate"], check=True)
    subprocess.run(["airflow", "connections", "create-default-connections"], check=True)

subprocess.run([
    "airflow", "users", "create",
    "--username", "admin",
    "--firstname", "Admin",
    "--lastname", "Admin",
    "--role", "Admin",
    "--email", "admin@example.com",
    "--password", "admin"
], input="admin\n", text=True, check=True)

with open(airflow_home / "airflow.cfg", "r") as f:
    config = f.read()

config = config.replace(
    f"dags_folder = {airflow_home}/dags",
    "dags_folder = ./dags"
).replace(
    "load_examples = True",
    "load_examples = False"
)

with open(airflow_home / "airflow.cfg", "w") as f:
    f.write(config)

print(f"export AIRFLOW_HOME={airflow_home}")
print(f"Airflow configured: AIRFLOW_HOME={airflow_home.relative_to(Path.cwd())}, dags_folder={dags_folder.relative_to(Path.cwd())}")
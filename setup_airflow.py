import os
from pathlib import Path
import subprocess
import sys
import shutil

# Use project root path for AIRFLOW_HOME
airflow_home = str(Path(__file__).resolve().parent)
dags_folder = str(Path(airflow_home) / "dags")
os.environ["AIRFLOW_HOME"] = airflow_home

# Disable interactive prompts
os.environ["AIRFLOW__DATABASE__LOAD_DEFAULT_CONNECTIONS"] = "False"
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
# This environment variable prevents the interactive prompt
os.environ["_AIRFLOW_DB_MIGRATE"] = "true"
python_executable = sys.executable

print(f"Using Python: {python_executable}")

try:
    logs_folder = os.path.join(airflow_home, "logs")
    db_path = os.path.join(airflow_home, "airflow.db")

    os.makedirs(dags_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    config_content = f"""
    [core]
    dags_folder = {dags_folder}
    load_examples = False

    [database]
    sql_alchemy_conn = sqlite:////{db_path}
    load_default_connections = False

    [webserver]
    base_url = http://localhost:8080
    secret_key = a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
    wtf_csrf_enabled = False
    web_server_host = 0.0.0.0
    web_server_port = 8080

    [auth]
    auth_backend = airflow.auth.backends.password_auth

    [logging]
    base_log_folder = {logs_folder}
    logging_level = INFO
    """

    cfg_path = os.path.join(airflow_home, "airflow.cfg")
    with open(cfg_path, "w") as f:
        f.write(config_content.strip())

    print(f"Configuration written to {cfg_path}")

    print("Initializing Airflow database...")
    subprocess.run(["airflow", "db", "init"], check=True)

    print("Resetting or creating admin user...")
    subprocess.run(["airflow", "users", "delete", "--username", "admin"], check=False)  # Ignore if user doesn’t exist
    subprocess.run([
        "airflow", "users", "create",
        "--username", "admin",
        "--firstname", "Admin",
        "--lastname", "User",
        "--role", "Admin",
        "--email", "admin@example.com",
        "--password", "admin"
    ], check=True)

    print(f"Airflow setup complete.")
    print(f"Airflow Web GUI: http://localhost:8080 (admin/admin)")

    print(f"Creating new airflow.cfg at {cfg_path}")
    FIXED_SECRET_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    # Use four slashes for absolute path as required by Airflow
    correct_db_uri = f"sqlite:////{airflow_home}/airflow.db"

    # Define webserver host and port
    webserver_host = "0.0.0.0"
    webserver_port = 8080
    base_url = f"http://localhost:{webserver_port}"

except Exception as e:
    print(f"Error: {e}")

print(f"Airflow configured with path: AIRFLOW_HOME={airflow_home}")
print(f"Using database at: {correct_db_uri}")
print(f"DAGs folder set to: {dags_folder}")
print(f"Airflow Web GUI: {base_url} (admin/admin) with database URI: {correct_db_uri}")
print(f"Debug: Verify with 'cat {cfg_path} | grep sql_alchemy_conn'")
sys.stdout.flush()
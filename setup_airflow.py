import os
from pathlib import Path
import subprocess
import sys
import shutil
import configparser
import time

# DVC setup will be handled by the DVCHook now
pass

def setup_postgres():
    """Set up PostgreSQL database for Airflow."""
    setup_script = Path(__file__).parent / "scripts" / "setup_postgres.sh"
    if not setup_script.exists():
        raise RuntimeError(f"PostgreSQL setup script not found at {setup_script}")
        
    print("Setting up PostgreSQL...")
    subprocess.run(["bash", str(setup_script)], check=True)
    
    # Install psycopg2 if not already installed
    try:
        import psycopg2
    except ImportError:
        print("Installing psycopg2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psycopg2-binary"], check=True)
    
    # Wait for PostgreSQL to be ready
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname="airflow",
                user="airflow",
                password="airflow",
                host="localhost"
            )
            conn.close()
            print("PostgreSQL is ready")
            break
        except psycopg2.Error:
            if attempt == max_retries - 1:
                raise
            print(f"Waiting for PostgreSQL to be ready (attempt {attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)

def validate_and_setup_airflow_home():
    """Set up and validate AIRFLOW_HOME environment."""
    airflow_home = Path(__file__).resolve().parent
    
    # Create required directories
    required_dirs = ["dags", "logs", "data_storage", "data_storage/raw"]
    for dir_name in required_dirs:
        dir_path = airflow_home / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")
    
    # Set up PostgreSQL
    setup_postgres()
    
    # DVC setup is now handled within the DVCHook
    
    # Set AIRFLOW_HOME environment variable
    os.environ["AIRFLOW_HOME"] = str(airflow_home)
    print(f"Set AIRFLOW_HOME={airflow_home}")
    
    return airflow_home, str(Path(airflow_home) / "dags")

# Set up and validate AIRFLOW_HOME
airflow_home, dags_folder = validate_and_setup_airflow_home()

# Configure Airflow environment
os.environ["AIRFLOW__DATABASE__LOAD_DEFAULT_CONNECTIONS"] = "False"
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = "postgresql+psycopg2://airflow:airflow@localhost/airflow"
os.environ["AIRFLOW__CORE__EXECUTOR"] = "LocalExecutor"
os.environ["_AIRFLOW_DB_MIGRATE"] = "true"

# Create Airflow configuration
config_content = f"""
[core]
dags_folder = {dags_folder}
load_examples = False
executor = LocalExecutor
task_execution_timeout = 120

[database]
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@localhost/airflow
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
base_log_folder = {airflow_home}/logs
logging_level = INFO
"""

try:
    # Write configuration file
    cfg_path = airflow_home / "airflow.cfg"
    cfg_path.write_text(config_content.strip())
    print(f"Configuration written to {cfg_path}")

    # Initialize database and create admin user
    print("Initializing Airflow database...")
    subprocess.run(["airflow", "db", "init"], check=True)
    
    print("Creating admin user...")
    subprocess.run(["airflow", "users", "delete", "--username", "admin"], check=False)
    subprocess.run([
        "airflow", "users", "create",
        "--username", "admin",
        "--firstname", "Admin",
        "--lastname", "User",
        "--role", "Admin",
        "--email", "admin@example.com",
        "--password", "admin"
    ], check=True)

    print("Airflow setup complete")
    print(f"Web GUI: http://localhost:8080 (admin/admin)")
    print(f"Using PostgreSQL database at: postgresql://airflow:airflow@localhost/airflow")

except Exception as e:
    print(f"Error during setup: {e}")
    raise
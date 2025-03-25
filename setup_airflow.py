import os
from pathlib import Path
import subprocess
import sys

# Get absolute path for project root
project_root = Path(__file__).resolve().parent
airflow_home = "."  # Keep relative for AIRFLOW_HOME
dags_folder = "dags"  # Use relative path

# Set the environment variable
os.environ["AIRFLOW_HOME"] = airflow_home

# Calculate absolute paths for database
absolute_airflow_home = project_root
absolute_db_path = f"sqlite:////{absolute_airflow_home}/airflow.db"

try:
    subprocess.run(["airflow", "--version"], check=True, capture_output=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    subprocess.run([sys.executable, "-m", "pip", "install", "apache-airflow"], check=True)
    # Continue even if this fails, as we'll create a new config next
    try:
        subprocess.run(["airflow", "db", "init"], check=True)
    except subprocess.CalledProcessError:
        print("Initial db init failed, but we'll create a custom config next...")

# Force creation of a new config file with absolute SQLite path
airflow_cfg_path = absolute_airflow_home / "airflow.cfg"
if not airflow_cfg_path.exists() or True:  # Always recreate for now
    print(f"Creating new airflow.cfg with absolute SQLite path: {absolute_db_path}")
    
    # Start with defaults
    subprocess.run(["airflow", "config", "list"], check=False)
    
    # Use a fixed, consistent secret key
    FIXED_SECRET_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    
    if airflow_cfg_path.exists():
        with open(airflow_cfg_path, "r") as f:
            config = f.read()
        
        # Fix the database connection to use absolute path
        config = config.replace(
            "sql_alchemy_conn = sqlite:///",
            f"sql_alchemy_conn = {absolute_db_path}"
        ).replace(
            f"dags_folder = {absolute_airflow_home}/dags",
            f"dags_folder = {dags_folder}"
        ).replace(
            "load_examples = True",
            "load_examples = False"
        ).replace(
            "secret_key = changeme",
            f"secret_key = {FIXED_SECRET_KEY}"
        ).replace(
            "base_url = http://0.0.0.0:8080", 
            "base_url = http://localhost:8080"
        ).replace(
            "wtf_csrf_enabled = True",
            "wtf_csrf_enabled = False"
        )
        
        with open(airflow_cfg_path, "w") as f:
            f.write(config)
    else:
        # Create a minimal config file with absolute path for SQLite
        with open(airflow_cfg_path, "w") as f:
            f.write(f"""[core]
sql_alchemy_conn = {absolute_db_path}
dags_folder = {dags_folder}
load_examples = False

[webserver]
base_url = http://localhost:8080
secret_key = {FIXED_SECRET_KEY}
wtf_csrf_enabled = False
web_server_host = 0.0.0.0
web_server_port = 8080
""")

# Now initialize the database with our custom config
subprocess.run(["airflow", "db", "init"], check=True)
subprocess.run(["airflow", "connections", "create-default-connections"], check=True)

# User creation is now handled exclusively in run_airflow.sh

print(f"Airflow configured: AIRFLOW_HOME={airflow_home}, dags_folder={dags_folder}")
print(f"Using database at: {absolute_db_path}")
sys.stdout.flush()
print(f"export AIRFLOW_HOME={airflow_home}", file=sys.stderr)
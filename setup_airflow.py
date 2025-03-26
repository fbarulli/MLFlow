import os
from pathlib import Path
import subprocess
import sys
import shutil

# Use script's directory as the root
script_dir = Path(__file__).parent.resolve()
airflow_home = str(script_dir)
dags_folder = str(script_dir / "dags")
os.environ["AIRFLOW_HOME"] = airflow_home
python_executable = sys.executable

print(f"Using Python: {python_executable}")

try:
    airflow_path = shutil.which("airflow")
    if not airflow_path:
        pip_bin_dir = os.path.dirname(shutil.which("pip"))
        potential_airflow_path = os.path.join(pip_bin_dir, "airflow")
        if os.path.exists(potential_airflow_path):
            airflow_path = potential_airflow_path
        else:
            airflow_path = python_executable
            print(f"Using '{python_executable} -m airflow' as the airflow command")
    print(f"Using Airflow at: {airflow_path}")

    if not os.path.exists(dags_folder):
        raise FileNotFoundError(f"DAGs folder not found at {dags_folder}. Please create it and add your DAGs.")

    # Clean up old state
    default_airflow_dir = Path.home() / "airflow"
    if default_airflow_dir.exists():
        print(f"Removing conflicting default Airflow dir: {default_airflow_dir}")
        shutil.rmtree(default_airflow_dir)
    airflow_cfg_path = Path(airflow_home) / "airflow.cfg"
    if airflow_cfg_path.exists():
        print(f"Removing existing airflow.cfg at {airflow_cfg_path}")
        os.remove(airflow_cfg_path)
    db_path = Path(airflow_home) / "airflow.db"
    if db_path.exists():
        print(f"Removing existing Airflow database at {db_path}")
        os.remove(db_path)

    print(f"Creating new airflow.cfg at {airflow_cfg_path}")
    FIXED_SECRET_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    # Use four slashes for absolute path as required by Airflow
    correct_db_uri = f"sqlite:////{airflow_home}/airflow.db"
    with open(airflow_cfg_path, "w") as f:
        f.write(f"""[core]
dags_folder = {dags_folder}
load_examples = False

[database]
sql_alchemy_conn = {correct_db_uri}

[webserver]
base_url = http://localhost:8080
secret_key = {FIXED_SECRET_KEY}
wtf_csrf_enabled = False
web_server_host = 0.0.0.0
web_server_port = 8080

[auth]
auth_backend = airflow.auth.backends.password_auth

[logging]
base_log_folder = {airflow_home}/logs
logging_level = INFO
""")
    print(f"Config written with sql_alchemy_conn = {correct_db_uri}")

    print("Initializing fresh Airflow database...")
    subprocess.run([python_executable, "-m", "airflow", "db", "init"], 
                   check=True, 
                   env={"AIRFLOW_HOME": airflow_home})

    print("Resetting or creating admin user...")
    subprocess.run([python_executable, "-m", "airflow", "users", "delete", "--username", "admin"],
                   env={"AIRFLOW_HOME": airflow_home}, check=False)
    subprocess.run([python_executable, "-m", "airflow", "users", "create",
                    "--username", "admin",
                    "--firstname", "Admin",
                    "--lastname", "User",
                    "--role", "Admin",
                    "--email", "admin@example.com",
                    "--password", "admin"],
                   check=True,
                   env={"AIRFLOW_HOME": airflow_home})

except Exception as e:
    print(f"Error setting up Airflow: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"Airflow configured with path: AIRFLOW_HOME={airflow_home}")
print(f"Using database at: {correct_db_uri}")
print(f"DAGs folder set to: {dags_folder}")
print("Airflow Web GUI: http://localhost:8080 (admin/admin)")
print(f"Debug: Verify with 'cat {airflow_cfg_path} | grep sql_alchemy_conn'")
sys.stdout.flush()
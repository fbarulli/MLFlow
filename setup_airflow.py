import os
from pathlib import Path
import subprocess
import sys
import shutil

# Use script's directory as the root for reproducibility
script_dir = Path(__file__).parent.resolve()
airflow_home = str(script_dir)
dags_folder = str(script_dir / "dags")  # Absolute path relative to script
os.environ["AIRFLOW_HOME"] = airflow_home
absolute_db_path = f"sqlite:///{airflow_home}/airflow.db"
python_executable = sys.executable

print(f"Using Python: {python_executable}")

try:
    result = subprocess.run([python_executable, "-m", "pip", "list"], capture_output=True, text=True)
    if "apache-airflow" not in result.stdout:
        print("Installing Apache Airflow...")
        subprocess.run([python_executable, "-m", "pip", "install", "apache-airflow"], check=True)
    else:
        print("Apache Airflow is already installed")

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

    # Check if dags folder exists, raise error if not
    if not os.path.exists(dags_folder):
        raise FileNotFoundError(f"DAGs folder not found at {dags_folder}. Please create it and add your DAGs.")

    airflow_cfg_path = Path(airflow_home) / "airflow.cfg"
    print(f"Recreating airflow.cfg at {airflow_cfg_path}")
    FIXED_SECRET_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    with open(airflow_cfg_path, "w") as f:
        f.write(f"""[core]
dags_folder = {dags_folder}
load_examples = False

[database]
sql_alchemy_conn = {absolute_db_path}

[webserver]
base_url = http://localhost:8080
secret_key = {FIXED_SECRET_KEY}
wtf_csrf_enabled = False
web_server_host = 0.0.0.0
web_server_port = 8080

[auth]
auth_backend = airflow.auth.backends.password_auth
""")

    db_path = Path(airflow_home) / "airflow.db"
    if db_path.exists():
        print(f"Removing existing Airflow database at {db_path}")
        os.remove(db_path)
    print("Initializing fresh Airflow database without examples...")
    subprocess.run([python_executable, "-m", "airflow", "db", "init", "--no-load-examples"], 
                   check=True, 
                   env={"AIRFLOW_HOME": airflow_home})

    print("Resetting or creating admin user...")
    subprocess.run([python_executable, "-m", "airflow", "users", "delete", "--username", "admin"],
                   env={"AIRFLOW_HOME": airflow_home}, check=False)  # Ignore if user doesn't exist
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
print(f"Using database at: {absolute_db_path}")
print(f"DAGs folder set to: {dags_folder}")
print("Airflow Web GUI: http://localhost:8080 (admin/admin)")
sys.stdout.flush()
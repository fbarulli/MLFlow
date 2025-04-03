#! /usr/bin/env python

import os
import sys
import time
import subprocess
from pathlib import Path
import warnings

# Import centralized configuration
import config

# These configurations are now imported from config.py
POSTGRES_DB = config.POSTGRES_DB
POSTGRES_USER = config.POSTGRES_USER
POSTGRES_PASSWORD = config.POSTGRES_PASSWORD
POSTGRES_HOST = config.POSTGRES_HOST
POSTGRES_PORT = config.POSTGRES_PORT

AIRFLOW_ADMIN_USER = config.ADMIN_USER
AIRFLOW_ADMIN_PASSWORD = config.ADMIN_PASSWORD
AIRFLOW_ADMIN_EMAIL = config.ADMIN_EMAIL
WEBSERVER_PORT = str(config.WEBSERVER_PORT)

# --- Script Logic ---
# Get Project Root from centralized config
AIRFLOW_HOME_PATH = Path(config.get_project_root())
AIRFLOW_HOME_PATH = Path(__file__).resolve().parent

def run_command(command, check=True, capture_output=False, text=True, env=None, **kwargs):
    """Helper function to run shell commands."""
    print(f"$ {' '.join(command)}")
    effective_env = os.environ.copy()
    if env:
        effective_env.update(env) # Merge provided env vars

    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            text=text,
            env=effective_env, # Use the updated environment
            **kwargs
        )
        if capture_output:
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}: {' '.join(command)}", file=sys.stderr)
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"STDERR:\n{e.stderr}", file=sys.stderr)
        if check: # Re-raise only if check=True
             raise
        return e # Return the error object if check=False
    except Exception as e:
        print(f"ERROR: Unexpected error running command {' '.join(command)}: {e}", file=sys.stderr)
        if check:
            raise
        return None


def setup_postgres():
    """Ensures PostgreSQL is set up and ready."""
    print("\n--- Running PostgreSQL Setup ---")
    setup_script = AIRFLOW_HOME_PATH / "scripts" / "setup_postgres.sh"
    if not setup_script.exists():
        print(f"ERROR: PostgreSQL setup script not found at {setup_script}", file=sys.stderr)
        print("Please ensure scripts/setup_postgres.sh exists and correctly initializes the database and user.", file=sys.stderr)
        raise FileNotFoundError(f"Missing required script: {setup_script}")

    # Make sure the script is executable (useful in some environments)
    run_command(["chmod", "+x", str(setup_script)])
    # Run the user-provided PostgreSQL setup script
    run_command(["bash", str(setup_script)])

    # Ensure psycopg2 is installed for the check below
    try:
        import psycopg2
    except ImportError:
        print("\n--- Installing psycopg2-binary (required for PostgreSQL connection) ---")
        run_command([sys.executable, "-m", "pip", "install", "psycopg2-binary"])

    # Wait for PostgreSQL to be ready after the setup script potentially started/restarted it
    max_retries = 10
    retry_delay = 3
    print(f"\n--- Checking PostgreSQL Connection (Host: {POSTGRES_HOST}) ---")
    for attempt in range(max_retries):
        try:
            import psycopg2 # Re-import in case pip install just ran
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                connect_timeout=5 # Add a connection timeout
            )
            conn.close()
            print("PostgreSQL connection successful.")
            return # Success
        except psycopg2.OperationalError as e:
            print(f"Waiting for PostgreSQL (attempt {attempt + 1}/{max_retries})... Error: {e}")
            if attempt == max_retries - 1:
                print("\nERROR: Could not connect to PostgreSQL after multiple retries.", file=sys.stderr)
                print("Please ensure PostgreSQL is running and accessible with the configured credentials.", file=sys.stderr)
                print(f"Attempted connection: dbname='{POSTGRES_DB}' user='{POSTGRES_USER}' host='{POSTGRES_HOST}'", file=sys.stderr)
                raise ConnectionError("Failed to connect to PostgreSQL") from e
            time.sleep(retry_delay)
        except Exception as e:
             print(f"ERROR: Unexpected error connecting to PostgreSQL: {e}", file=sys.stderr)
             raise


def setup_airflow_environment_and_config():
    """Sets up AIRFLOW_HOME, directories, env vars, and airflow.cfg."""
    print(f"\n--- Setting AIRFLOW_HOME to Project Root: {AIRFLOW_HOME_PATH} ---")
    os.environ["AIRFLOW_HOME"] = str(AIRFLOW_HOME_PATH)

    # Define standard Airflow directories relative to AIRFLOW_HOME
    dags_folder = AIRFLOW_HOME_PATH / "dags"
    logs_folder = AIRFLOW_HOME_PATH / "logs"
    plugins_folder = AIRFLOW_HOME_PATH / "plugins" # Even if empty, good practice
    data_storage_folder = AIRFLOW_HOME_PATH / "data_storage" # As seen in your tree

    required_dirs = [dags_folder, logs_folder, plugins_folder, data_storage_folder]
    print("\n--- Ensuring required Airflow directories exist ---")
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Ensured: {dir_path}")

    # Call PostgreSQL setup *before* setting DB connection strings in config/env
    setup_postgres()

    # Set Airflow Environment Variables for configuration
    # These often override airflow.cfg settings, making them reliable
    print("\n--- Setting Airflow environment variables for configuration ---")
    sql_alchemy_conn = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = sql_alchemy_conn
    os.environ["AIRFLOW__CORE__EXECUTOR"] = "LocalExecutor"
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    os.environ["AIRFLOW__DATABASE__LOAD_DEFAULT_CONNECTIONS"] = "False"
    os.environ["AIRFLOW__WEBSERVER__EXPOSE_CONFIG"] = "True" # Useful for debugging
    # Use Password authentication (FAB = Flask-AppBuilder, the default web UI framework)
    os.environ["AIRFLOW__AUTH__AUTH_BACKEND"] = "airflow.providers.fab.auth_manager.fab_auth_manager.PasswordUserAuthManager"
    # Set webserver details via env vars too
    os.environ["AIRFLOW__WEBSERVER__WEB_SERVER_HOST"] = "0.0.0.0"
    os.environ["AIRFLOW__WEBSERVER__WEB_SERVER_PORT"] = WEBSERVER_PORT
    # Disable CSRF for local dev convenience (REMOVE THIS FOR PRODUCTION)
    os.environ["AIRFLOW__WEBSERVER__WTF_CSRF_ENABLED"] = "False"
    if os.environ["AIRFLOW__WEBSERVER__WTF_CSRF_ENABLED"] == "False":
        warnings.warn("WTF_CSRF_ENABLED is set to False. This is insecure and should ONLY be used for local development.", UserWarning)

    # For Python 3.12 compatibility with pendulum, if needed (often helps)
    os.environ["PENDULUM_PREFER_ZONEINFO"] = "True"

    print(f"  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://{POSTGRES_USER}:***@{POSTGRES_HOST}/{POSTGRES_DB}")
    print(f"  AIRFLOW__CORE__EXECUTOR={os.environ['AIRFLOW__CORE__EXECUTOR']}")
    print(f"  AIRFLOW__CORE__LOAD_EXAMPLES={os.environ['AIRFLOW__CORE__LOAD_EXAMPLES']}")
    print(f"  AIRFLOW__AUTH__AUTH_BACKEND={os.environ['AIRFLOW__AUTH__AUTH_BACKEND']}")
    # ... add other important env vars if needed ...

    # Generate/Overwrite airflow.cfg
    # NOTE: This WILL overwrite your existing airflow.cfg.
    # Environment variables set above will generally take precedence.
    print(f"\n--- Generating/Overwriting airflow.cfg at {AIRFLOW_HOME_PATH / 'airflow.cfg'} ---")
    config_content = f"""[core]
# Setting AIRFLOW_HOME is preferred over setting this in the cfg
# airflow_home = {AIRFLOW_HOME_PATH}
dags_folder = {dags_folder}
plugins_folder = {plugins_folder}
load_examples = {os.environ['AIRFLOW__CORE__LOAD_EXAMPLES']}
executor = {os.environ['AIRFLOW__CORE__EXECUTOR']}
# Don't log sensitive variable values
hide_sensitive_variable_fields = True

[database]
sql_alchemy_conn = {sql_alchemy_conn}
load_default_connections = {os.environ['AIRFLOW__DATABASE__LOAD_DEFAULT_CONNECTIONS']}
# Pool settings can be adjusted if needed under heavy load
# sql_alchemy_pool_size = 5
# sql_alchemy_pool_recycle = 1800

[webserver]
base_url = http://localhost:{WEBSERVER_PORT}
# Generate a strong secret key for production: python -c 'import os; print(os.urandom(16))'
secret_key = {os.environ.get('AIRFLOW__WEBSERVER__SECRET_KEY', 'fallback_secret_key_for_dev_only_change_in_prod')}
web_server_host = {os.environ['AIRFLOW__WEBSERVER__WEB_SERVER_HOST']}
web_server_port = {os.environ['AIRFLOW__WEBSERVER__WEB_SERVER_PORT']}
expose_config = {os.environ['AIRFLOW__WEBSERVER__EXPOSE_CONFIG']}
# Disable CSRF protection. Set to True in production!
wtf_csrf_enabled = {os.environ['AIRFLOW__WEBSERVER__WTF_CSRF_ENABLED']}

[auth]
# Using env var AIRFLOW__AUTH__AUTH_BACKEND is preferred
# auth_backend = {os.environ['AIRFLOW__AUTH__AUTH_BACKEND']}

[logging]
base_log_folder = {logs_folder}
logging_level = INFO
# Customize logging further if needed
# fab_logging_level = WARN
# werkzeug_logging_level = WARN

[operators]
default_timezone = utc

[scheduler]
# How often (seconds) to check for new DAGs
dag_dir_list_interval = 30
"""
    try:
        cfg_path = AIRFLOW_HOME_PATH / "airflow.cfg"
        # Check for existing SQLite file and warn if switching
        sqlite_db_path = AIRFLOW_HOME_PATH / "airflow.db"
        if sqlite_db_path.exists():
            print(f"\nWARNING: Found existing SQLite database '{sqlite_db_path.name}'.")
            print("This setup script configures Airflow to use PostgreSQL.")
            print(f"The '{sqlite_db_path.name}' file will no longer be used by Airflow with this configuration.")
            print("You may want to remove it manually after successful setup to avoid confusion.")

        cfg_path.write_text(config_content)
        print(f"Configuration written to {cfg_path}")
    except Exception as e:
        print(f"ERROR: Failed to write airflow.cfg: {e}", file=sys.stderr)
        raise

    return AIRFLOW_HOME_PATH # Return the path for use in the next step


def initialize_airflow_db_and_user(airflow_home):
    """Initializes the Airflow DB (using PostgreSQL) and creates the admin user."""
    print(f"\n--- Ensuring Airflow is run with AIRFLOW_HOME={airflow_home} ---")
    # Set environment specifically for the airflow commands
    airflow_cmd_env = os.environ.copy()
    airflow_cmd_env["AIRFLOW_HOME"] = str(airflow_home)

    # Install necessary providers, especially FAB for auth, BEFORE db migrate/init
    # Ensure postgres provider is also included
    print("\n--- Installing/Updating Airflow core and necessary providers (fab, postgres) ---")
    # Using a flexible constraint allows pip to resolve dependencies better. Adjust if specific versions are needed.
    # Consider adding other providers your DAGs need here (e.g., dvc, http, etc.)
    run_command([
        sys.executable, "-m", "pip", "install",
        # Ensure core airflow is installed/updated too
        "apache-airflow>=2.6.0,<3",
        # Ensure providers match the core version range roughly
        "apache-airflow-providers-fab",
        "apache-airflow-providers-postgres"
        ], env=airflow_cmd_env)
    # You might want to add 'apache-airflow-providers-dvc' here if your dvc_hook needs it

    print("\n--- Initializing/Migrating Airflow Database (PostgreSQL) ---")
    # Use 'db migrate' - it's idempotent and preferred over 'db init' for setup and upgrades.
    # It ensures the schema matches the installed Airflow version.
    run_command(["airflow", "db", "migrate"], env=airflow_cmd_env)

    print("\n--- Creating/Updating Airflow Admin User ---")
    # Try deleting first to make user creation idempotent (ignore error if user doesn't exist)
    # Use check=False so the script doesn't fail if the user wasn't there initially
    print(f"Attempting to delete existing user '{AIRFLOW_ADMIN_USER}' (if exists)...")
    run_command(["airflow", "users", "delete", "--username", AIRFLOW_ADMIN_USER], check=False, env=airflow_cmd_env)

    # Create the admin user
    print(f"Creating user '{AIRFLOW_ADMIN_USER}' with role 'Admin'...")
    run_command([
        "airflow", "users", "create",
        "--username", AIRFLOW_ADMIN_USER,
        "--firstname", "Admin",
        "--lastname", "User",
        "--role", "Admin",
        "--email", AIRFLOW_ADMIN_EMAIL,
        "--password", AIRFLOW_ADMIN_PASSWORD
    ], env=airflow_cmd_env) # check=True is default, will raise error if creation fails


# --- Main Execution ---
if __name__ == "__main__":
    print("========================================")
    print("=== Starting Airflow PostgreSQL Setup ===")
    print("========================================")
    start_time = time.time()
    airflow_home_path_used = None
    try:
        # 1. Setup Environment, Directories, Config, and ensure PostgreSQL is ready
        airflow_home_path_used = setup_airflow_environment_and_config()

        # 2. Initialize DB Schema and Create Admin User
        initialize_airflow_db_and_user(airflow_home_path_used)

        end_time = time.time()
        print("\n======================================")
        print("=== Airflow Setup Complete! ===")
        print("======================================")
        print(f"Setup took {end_time - start_time:.2f} seconds.")
        print(f"\nAIRFLOW_HOME configured at: {airflow_home_path_used}")
        print(f"Database configured: PostgreSQL on {POSTGRES_HOST} (DB: {POSTGRES_DB})")
        print(f"\nAirflow UI should be available at: http://localhost:{WEBSERVER_PORT}")
        print(f"Login with: {AIRFLOW_ADMIN_USER} / {AIRFLOW_ADMIN_PASSWORD}")
        print("\n--- IMPORTANT NEXT STEPS ---")
        print("1. **CLEAR BROWSER CACHE/COOKIES** for localhost/your-airflow-domain.")
        print("   This is crucial to avoid login issues ('NoneType' object error) from old sessions.")
        print("2. Start Airflow services (in separate terminals):")
        print(f"   cd {airflow_home_path_used}")
        print(f"   # Activate your conda/virtual environment if needed (e.g., conda activate mlflow)")
        print(f"   export AIRFLOW_HOME=\"{airflow_home_path_used}\"")
        print(f"   airflow webserver -p {WEBSERVER_PORT}")
        print("   --- and in another terminal ---")
        print(f"   cd {airflow_home_path_used}")
        print(f"   # Activate environment again")
        print(f"   export AIRFLOW_HOME=\"{airflow_home_path_used}\"")
        print("   airflow scheduler")
        print("\n   (Alternatively, use your existing scripts: ./scripts/start_webserver.sh and ./scripts/start_scheduler.sh if they set AIRFLOW_HOME correctly)")

        # Reminder about the SQLite file
        sqlite_db_file = airflow_home_path_used / "airflow.db"
        if sqlite_db_file.exists():
             print(f"\nReminder: The old SQLite file '{sqlite_db_file.name}' is still present but unused.")
             print("  You may want to delete it: rm airflow.db")


    except (FileNotFoundError, ConnectionError, subprocess.CalledProcessError) as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print("!!!   AIRFLOW SETUP FAILED            !!!", file=sys.stderr)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print(f"\nError Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        if airflow_home_path_used:
             logs_dir = airflow_home_path_used / 'logs'
             print(f"\nPlease check for more details in setup output above and potentially in Airflow logs: {logs_dir}", file=sys.stderr)
        else:
            print("\nSetup failed early, possibly before AIRFLOW_HOME was fully established.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print("!!!   UNEXPECTED ERROR DURING SETUP   !!!", file=sys.stderr)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print(f"\nError Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        # Print traceback for unexpected errors
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
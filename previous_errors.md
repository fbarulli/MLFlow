# Previous Errors and Fixes

## Webserver Authentication Error (`AttributeError: 'NoneType' object has no attribute 'is_active'`)

**Error Message:**
```
AttributeError: 'NoneType' object has no attribute 'is_active'
  File ".../flask_login/login_manager.py", line 364, in _load_user
    user = self._user_callback(user_id)
  File ".../airflow/providers/fab/auth_manager/security_manager/override.py", line 1625, in load_user
    if user.is_active:
```

**Origin:**
This error appears in the Airflow webserver logs (`airflow-webserver.log` or console output) when accessing the UI after the database backend was switched (e.g., from SQLite to PostgreSQL).

**Root Cause:**
The user's browser session cookie contains an ID referencing a user from the *old* database. When the webserver, now connected to the *new* database (PostgreSQL), tries to load this user ID via `load_user`, it finds no matching user, resulting in `user` being `None`. The subsequent attempt to access `user.is_active` causes the `AttributeError`.

**Solution:**
1.  **Clear Browser Cookies:** The primary solution is to clear cookies for the Airflow webserver domain (e.g., `localhost`) in the browser. This removes the invalid session cookie.
2.  **Ensure Clean Restart:** Make sure old Airflow processes are stopped before restarting. The `restart_airflow.sh` script attempts this with `pkill`, but may require manual intervention or `sudo` if permissions are insufficient.
    ```bash
    # Manually stop if needed (replace PIDs)
    # kill <pid1> <pid2>
    # OR potentially use sudo
    # sudo pkill -f airflow
    
    # Then restart
    bash scripts/restart_airflow.sh
    ```
3.  **Log In Again:** After clearing cookies and restarting, access the Airflow UI. You should be prompted to log in again, creating a new, valid session tied to the PostgreSQL database.

## Stale DAG File Error

**Error Message:**
```
TypeError: DVCHook.add_and_push() got an unexpected keyword argument 'cwd'
```

**Origin:**
This error reappeared in the `version_data` task even after the code in `weather_dag.py` was corrected to remove the `cwd` argument.

**Root Cause:**
The Airflow scheduler was likely still using a cached or older version of the `weather_dag.py` file. Changes made to DAG files are not always picked up immediately by the scheduler, leading to tasks executing with outdated code.

**Troubleshooting:**
1. Verified the code in `dags/weather_dag.py` was correct using `read_file`.
2. Confirmed the `cwd` argument was indeed removed from the `hook.add_and_push` call.

**Solution:**
Restarting the Airflow scheduler and webserver forces a reload of all DAG files, ensuring the latest code is used.
```bash
bash scripts/restart_airflow.sh
```
This is a common step required after modifying DAG definitions to ensure the scheduler recognizes the changes.

## DVCHook Argument Error

**Error Message:**
```
TypeError: DVCHook.add_and_push() got an unexpected keyword argument 'cwd'
```

**Origin:**
The error occurred in the `version_data` task within `weather_dag.py` when calling the `add_and_push` method of the `DVCHook`.

**Root Cause:**
We refactored the `DVCHook` to manage its own working directory (`cwd`) during initialization based on `AIRFLOW_HOME`. This meant the `add_and_push` method no longer needed the `cwd` argument. However, the calling code in `weather_dag.py` was still passing the `project_root` as the `cwd` argument.

**Fix:**
Updated the `version_weather_data` function and the `PythonOperator` call in `weather_dag.py` to remove the `cwd` argument:
```python
# In version_weather_data function:
hook = DVCHook() # Hook now determines cwd internally
hook.add_and_push(
    filepath="data_storage/raw/weather.csv",
    commit=False # commit is ignored
)

# In PythonOperator op_kwargs:
op_kwargs={
    'run_id': '{{ run_id }}' # Removed 'project_root'
}
```
This aligns the DAG code with the updated `DVCHook` signature, ensuring the hook manages its working directory consistently.

## DVC Push Failure (401 Unauthorized)

**Error Message:**
```
aiohttp.client_exceptions.ClientResponseError: 401, message='Unauthorized', url='https://dagshub.com/fbarulli/MLFlow.dvc/files/md5/...'
subprocess.CalledProcessError: Command '['dvc', 'push', '-v']' returned non-zero exit status 1.
```

**Origin:**
The error occurs in the `version_data` task when attempting to push data to the DVC remote. The `dvc push` command fails with a 401 Unauthorized error.

**Root Cause Analysis:**
Attempts to configure DVC credentials using `dvc config` within the hook (`_setup_dvc`) did not reliably solve the authentication issue for `dvc push` in the Airflow task environment. It seems the most reliable way to authenticate `dvc push` in this context is via environment variables.

**Fix (Simplified Authentication):**
Further simplified the `DVCHook`:

1.  **`_setup_dvc` Method:**
    *   Only runs `dvc init --no-scm` if needed.
    *   Ensures the remote URL is set using `dvc remote add --force -d origin ...`.
    *   **Removed** all `dvc config` commands for setting `auth`, `user`, and `password`.

2.  **`add_and_push` Method:**
    *   **Relies solely on setting `DVC_USERNAME` and `DVC_PASSWORD` environment variables** for the `dvc push` subprocess call.
    ```python
       # Push to DVC with retries (using env with explicit credentials)
       env_with_creds = os.environ.copy()
       env_with_creds['DVC_USERNAME'] = self.dvc_user
       env_with_creds['DVC_PASSWORD'] = self.dvc_password
       # ... retry loop ...
           result = subprocess.run(
               ["dvc", "push", "-v"],
               env=env_with_creds, # Use env with explicit credentials
               cwd=self.cwd,
               # ... rest of subprocess args ...
           )
       # ... success/retry logic ...
    ```

This approach:
*   Avoids potential conflicts or inconsistencies from writing to `.dvc/config` within the hook.
*   Uses the most direct method (environment variables) to authenticate the `dvc push` command, which appears necessary in the Airflow task environment.
*   Still ensures DVC is initialized and the remote URL is configured.

## Task Signal Handling Improvement

**Error Message:**
```
airflow.exceptions.AirflowTaskTerminated: Task received SIGTERM signal
```

**Origin:**
Even after switching to PostgreSQL, the monitor_data task was still being terminated by SIGTERM after about 15 seconds, despite using BashOperator with a simple sleep command.

**Root Cause Analysis:**
1. BashOperator's sleep command isn't gracefully handling interrupts
2. Long-running sleep command doesn't provide visibility into task progress
3. No proper signal handling implementation in the bash command

**Fix:**
Replaced BashOperator with PythonOperator for better control:
```python
def monitor_function(**context):
    """Monitor data with proper signal handling."""
    try:
        print("Starting monitoring...")
        # Use a loop with smaller sleep intervals
        for i in range(60):
            time.sleep(1)
            if i % 10 == 0:  # Log progress
                print(f"Monitoring... {i+1}/60 seconds")
        print("Monitoring complete")
        return "Monitoring successful"
    except Exception as e:
        print(f"Error during monitoring: {e}")
        raise
```

This approach:
- Uses smaller sleep intervals for better responsiveness
- Provides progress logging every 10 seconds
- Implements proper exception handling
- Gives better visibility into task status
- More controllable execution flow

## Task Supervisor Timeout Issue

**Error Message:**
```
airflow.exceptions.AirflowTaskTerminated: Task received SIGTERM signal
```

**Origin:**
Tasks are being terminated by SIGTERM exactly 15 seconds after starting, regardless of database type (SQLite or PostgreSQL).

**Root Cause:**
Airflow's Task Supervisor is configured with a default timeout of 15 seconds (`task_supervisor_timeout`). This is causing tasks to be killed prematurely before they can complete their execution, regardless of database configuration.

**Key Findings from Logs:**
```
[2025-04-02T16:04:04.300+0000] Started task execution
[2025-04-02T16:04:19.596+0000] Received SIGTERM. Terminating subprocesses
```
The consistent ~15 second gap between start and SIGTERM indicates this is a Task Supervisor timeout issue, not a database or resource problem.

**Fix:**
Updated Airflow configuration to have appropriate timeouts:
```ini
[core]
task_supervisor_timeout = 120
dag_file_processor_timeout = 600
task_execution_timeout = 120

[scheduler]
scheduler_heartbeat_sec = 5
scheduler_health_check_threshold = 30
parsing_processes = 2
```

This configuration:
- Allows tasks proper time to complete (120s)
- Maintains scheduler health with frequent checks
- Sets appropriate resource limits
- Addresses the actual root cause (Task Supervisor timeout)
- Works regardless of database backend

## SQLite Concurrency Issue

**Error Message:**
```
self.connection._commit_impl()
File ".../sqlalchemy/engine/base.py", line 1094, in _commit_impl
self.engine.dialect.do_commit(self.connection)
```

**Origin:**
The error occurs during SQLite database commit operations in the monitor_data task. The task fails when trying to commit changes to the SQLite database due to concurrent access issues.

**Root Cause:**
1. SQLite has limitations with concurrent access
2. Using LocalExecutor with SQLite is not recommended for production environments
3. Database lock contentions occur during parallel task execution

**Initial Fix Attempt (Not Recommended):**
```python
# Adding SQLite-specific parameters and limiting concurrency
os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = \
    f"sqlite:////{airflow_home}/airflow.db?cache=shared&uri=true&timeout=60"
os.environ["AIRFLOW__CORE__PARALLELISM"] = "1"
```

**Better Solution:**
Replace SQLite with PostgreSQL:
1. Install PostgreSQL:
   ```bash
   sudo apt-get install postgresql postgresql-contrib
   ```

2. Create Airflow database and user:
   ```sql
   CREATE DATABASE airflow;
   CREATE USER airflow WITH PASSWORD 'airflow';
   GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
   ```

3. Update Airflow configuration:
   ```python
   os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = \
       "postgresql+psycopg2://airflow:airflow@localhost/airflow"
   ```

This approach:
- Provides proper concurrent access handling
- Enables true parallel task execution
- Is more suitable for production environments
- Follows Airflow's recommended database setup

## Git Integration Error with DVC

**Error Message:**
```
subprocess.CalledProcessError: Command '['git', 'add', 'data_storage/raw/weather.csv.dvc']' returned non-zero exit status 128.
```

**Origin:**
The error occurred when the DVCHook attempted to use git commands after adding files to DVC. This happened because DVC was initialized with git SCM integration by default.

**Root Cause:**
1. DVC by default integrates with git for version control
2. Our Airflow environment doesn't need git integration for DVC
3. The git commands in DVCHook were failing because either:
   - Git wasn't properly configured in the Airflow environment
   - Git operations weren't necessary for our DVC usage

**Fix:**
1. Modified setup_airflow.py to initialize DVC without git:
   ```python
   # Remove any existing DVC setup
   if dvc_dir.exists():
       shutil.rmtree(dvc_dir)
   
   print("Initializing DVC without git...")
   subprocess.run(["dvc", "init", "--no-scm"], cwd=airflow_home, check=True)
   ```

2. Simplified DVCHook to remove git operations:
   - Removed all git add/commit/push commands
   - Made commit parameter optional and ignored
   - Focus only on DVC operations (add/push)

This approach:
- Eliminates dependency on git configuration
- Simplifies the versioning process
- Focuses on DVC's data versioning capabilities without SCM integration

## DagFileProcessorManager Heartbeat Timeout

**Error Message:**

```
DagFileProcessorManager (PID=26099) last sent a heartbeat 263.09 seconds ago! Restarting it
```

**Origin:**

This error is found in the scheduler logs (`logs/scheduler.log`). It indicates that the DagFileProcessorManager, a component of the Airflow scheduler responsible for processing DAG files, is not sending heartbeats to the main scheduler process within the expected interval. This leads to the scheduler restarting the DagFileProcessorManager.

**Possible Cause:**

The issue might be related to resource contention or deadlocks within the DagFileProcessorManager, preventing it from sending heartbeats in a timely manner. It's also possible that the DockerOperator is contributing to this issue, as suspected earlier.

**Current Fix Attempt:**

To investigate if the DockerOperator is the root cause, we have replaced the `DockerOperator` in the `weather_monitor_dag.py` DAG with a simple `BashOperator` that just sleeps. This will help us determine if the issue persists without the DockerOperator.

**Next Steps:**

After applying this change, we need to monitor the scheduler logs again to see if the `DagFileProcessorManager` heartbeat timeout error still occurs. If the error disappears, it would strongly suggest that the DockerOperator is indeed related to the problem. If the error persists, we will need to investigate other potential causes for the heartbeat timeout.

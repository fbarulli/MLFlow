# Previous Errors and Fixes

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

## DVC Remote 'origin' Not Configured

**Error Message:**

```
ValueError: DVC remote 'origin' not configured in .dvc/config
```

**Origin:**

This error is found in the task logs for the `version_data` task in the `weather_data_collection` DAG (`logs/dag_id=weather_data_collection/run_id=scheduled__2025-03-31T17:45:00+00:00/task_id=version_data/attempt=1.log`). It occurs because the DVC remote named 'origin' is not configured in the `.dvc/config` file. The error originates from the `DVCHook` class in `dags/airflow_custom_hooks/dvc_hook.py`, specifically in the `_read_dvc_config` method.

**Fix:**

To fix this error, you need to configure the DVC remote 'origin'. This typically involves running the command `dvc remote add origin <remote_storage_url>` in the terminal within the project directory. Replace `<remote_storage_url>` with the actual URL of your remote storage (e.g., cloud storage bucket or SSH server). After adding the remote, DVC will store the configuration in `.dvc/config`, and the `DVCHook` will be able to read it successfully.

## Task Terminated by SIGTERM

**Error Message:**

```
airflow.exceptions.AirflowTaskTerminated: Task received SIGTERM signal
```

**Origin:**

This error is found in the task logs for the `monitor_data` task in the `weather_data_monitoring` DAG (`logs/dag_id=weather_data_monitoring/run_id=scheduled__2025-03-31T17:48:00+00:00/task_id=monitor_data/attempt=1.log`). It indicates that the task was terminated by a SIGTERM signal.

**Possible Cause:**

The SIGTERM signal is a termination signal that can be sent by the Airflow scheduler or the operating system to stop a running process. Possible causes include:

1. **Task Timeout:** Airflow might have a default task timeout, and if the task exceeds this timeout, it will be terminated.
2. **Scheduler Restart/Shutdown:** If the Airflow scheduler is restarted or shut down, it will send SIGTERM to all running tasks.
3. **Resource Contention:** In cases of high system load, the scheduler might terminate tasks to free up resources.
4. **External Signal:** An external process or user might have manually sent a SIGTERM signal to the task process.

**Current Status:**

We are currently investigating why the `monitor_data` task, which is now a simple `BashOperator` sleeping for 60 seconds, is being terminated by SIGTERM. This issue might be related to the previous DagFileProcessorManager heartbeat timeout error, or it could be a separate problem.

**Next Steps:**

We need to further investigate the scheduler logs and task logs to understand why the SIGTERM signal is being sent. We should also check Airflow configurations for any task timeout settings.


**Investigation into SIGTERM for `monitor_data` Task**

After observing the SIGTERM signal for the `monitor_data` task, we investigated the possible causes. We checked the Airflow configurations in `airflow.cfg` to see if there are any task timeout settings that might be causing this behavior.

**Findings:**

Upon reviewing `airflow.cfg`, we did not find any explicit task timeout configurations set. Airflow does have a default `task_execution_timeout` setting, but it is usually set to a longer duration (e.g., hours or days, depending on the Airflow version). Without a specific timeout configured, it's unlikely that the default task timeout is causing the `monitor_data` task (which only sleeps for 60 seconds) to be terminated after approximately 33 seconds.

**Possible Root Cause:**

Given that the `DagFileProcessorManager` heartbeat timeout issue was observed earlier, and now we are seeing SIGTERM signals for tasks, it is possible that the scheduler itself is becoming unstable or overloaded. When the scheduler becomes unresponsive or overloaded, it might be sending SIGTERM signals to running tasks as part of its attempt to recover or manage resources.

**Next Steps:**

1. **Monitor Scheduler Stability:** We need to closely monitor the scheduler logs (`logs/scheduler.log`) for any signs of instability, errors, or warnings. We should look for patterns or recurring issues that might indicate scheduler overload or crashes.
2. **Resource Usage:** We should check the resource usage of the Airflow scheduler and worker processes (CPU, memory, disk I/O) to see if there are any resource bottlenecks. If the scheduler is running out of resources, it could explain the SIGTERM signals and heartbeat timeouts.
3. **Increase Scheduler Resources (if applicable):** If resource contention is identified as a potential issue, we might need to increase the resources allocated to the Airflow scheduler (e.g., more CPU, memory). However, since we are using a SequentialExecutor with SQLite, resource limitations might be inherent to this setup.
4. **Revert to DockerOperator (with caution):** As the initial suspicion was related to the DockerOperator, and we have now ruled out task timeouts as the immediate cause of SIGTERM, we might need to cautiously revert back to using the DockerOperator in `weather_monitor_dag.py` to see if the original `DagFileProcessorManager` heartbeat timeout issue reappears. If it does, it would strengthen the hypothesis that the DockerOperator is somehow related to the scheduler instability.

**Current Focus:**


**Attempt to Resolve SIGTERM and Scheduler Instability**

Based on the analysis of scheduler logs and the suspicion of scheduler instability, we are attempting to switch from `SequentialExecutor` to `LocalExecutor` in `airflow.cfg`. Additionally, we are correcting the `sql_alchemy_conn` path in `airflow.cfg` to ensure it is properly formatted for SQLite.

**Changes Made:**

1. **Switched Executor to LocalExecutor:**
   - Modified `airflow.cfg` to set `executor = LocalExecutor` under the `[core]` section.
2. **Corrected `sql_alchemy_conn` Path:**
   - Modified `airflow.cfg` to correct the `sql_alchemy_conn` path to `sqlite:////home/ubuntu/MLFlow/airflow.db` under the `[database]` section.

**Reasoning:**

- `SequentialExecutor` is known to have limitations in terms of parallelism and might become a bottleneck, especially with SQLite. Switching to `LocalExecutor` can potentially improve performance and stability, as it allows for parallel task execution within a single process.
- Correcting the `sql_alchemy_conn` path ensures that Airflow can properly connect to the SQLite database, which is crucial for scheduler operations.

**Next Steps:**

1. **Restart Airflow Services:** We need to restart the Airflow scheduler and webserver for the configuration changes to take effect. We will use the `scripts/restart_airflow.sh` script for this purpose.
2. **Monitor Scheduler Logs:** After restarting Airflow, we will closely monitor the scheduler logs (`logs/scheduler.log`) to see if the SIGTERM signals and scheduler instability issues are resolved.
3. **Observe Task Execution:** We will observe the execution of the `weather_data_monitoring` DAG and its tasks to ensure they are running smoothly without termination errors.

**Expected Outcome:**

By switching to `LocalExecutor` and correcting the database connection path, we hope to improve the stability and performance of the Airflow scheduler and resolve the SIGTERM issue for the `monitor_data` task.

**If the issue persists:**

If the SIGTERM issue continues after these changes, we might need to investigate further into resource limitations, task configurations, or consider using a more robust executor and database setup (e.g., CeleryExecutor with PostgreSQL).

**Increased Task Execution Timeout**

To further mitigate potential task termination issues, we have added an explicit `task_execution_timeout` setting in `airflow.cfg`.

**Changes Made:**

1. **Set `task_execution_timeout`:**
   - Modified `airflow.cfg` to include `task_execution_timeout = 120` under the `[core]` section. This sets a 2-minute timeout for task execution, which should be sufficient for the `monitor_data` task.

**Reasoning:**

- While we didn't find any immediate evidence of task timeouts in the logs, setting an explicit `task_execution_timeout` can help prevent unexpected task terminations due to potential internal timeouts or resource contention.

**Next Steps:**

1. **Restart Airflow Services:** We have already restarted Airflow services after switching to `LocalExecutor` and correcting the database path. No further restart is needed at this moment unless new changes are made.
2. **Monitor Task Execution:** We will continue to monitor the execution of the `weather_data_monitoring` DAG and its tasks, paying close attention to the `monitor_data` task, to see if the SIGTERM issue is resolved with the increased timeout and executor change.
3. **Review Scheduler Logs (if needed):** If the issue persists, we will re-examine the scheduler logs for any new error patterns or warnings.

**Current Status:**

We have implemented several changes to address the SIGTERM issue, including switching to `LocalExecutor`, correcting the database path, and increasing the task execution timeout. We are now monitoring the system to observe if these changes have resolved the problem.

**Further Investigation (if issue persists):**

If the SIGTERM issue persists despite these efforts, we will need to delve deeper into potential resource constraints, task configurations, or consider more advanced debugging techniques.
For now, the primary focus should be on monitoring the scheduler logs and system resource usage to understand the root cause of the SIGTERM signals and potential scheduler instability.

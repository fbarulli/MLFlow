from typing import Optional
from pathlib import Path
from airflow.hooks.base import BaseHook
import subprocess
import os
import time
import shutil

class DVCHook(BaseHook):
    """
    Hook for interacting with DVC.
    Ensures DVC is initialized and configured before running commands.
    """
    
    def __init__(self, cwd: Optional[str] = None) -> None:
        super().__init__()
        self.cwd = cwd or os.environ.get("AIRFLOW_HOME")
        if not self.cwd:
            raise ValueError("Could not determine working directory (cwd or AIRFLOW_HOME)")
        self.cwd_path = Path(self.cwd)
        self._setup_dvc()
        self._read_dvc_config() # Keep this to set instance variables if needed

    def _setup_dvc(self) -> None:
        """Initialize DVC and configure remote if not already set up."""
        dvc_dir = self.cwd_path / ".dvc"
        
        if not dvc_dir.exists():
            self.log.info(f"Initializing DVC in {self.cwd} without git...")
            try:
                subprocess.run(
                    ["dvc", "init", "--no-scm"], 
                    cwd=self.cwd, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
            except subprocess.CalledProcessError as e:
                self.log.error(f"DVC init failed: {e.stderr}")
                raise
        else:
            self.log.info(f"DVC already initialized in {self.cwd}")

        # Configure DVC remote using dvc config commands
        try:
            self.log.info("Configuring DVC remote 'origin'...")
            subprocess.run(
                ["dvc", "remote", "add", "-d", "origin", "https://dagshub.com/fbarulli/MLFlow.dvc"],
                cwd=self.cwd, check=True, capture_output=True, text=True
            )
            subprocess.run(
                ["dvc", "remote", "modify", "origin", "--local", "auth", "basic"],
                cwd=self.cwd, check=True, capture_output=True, text=True
            )
            subprocess.run(
                ["dvc", "remote", "modify", "origin", "--local", "user", "fbarulli"],
                cwd=self.cwd, check=True, capture_output=True, text=True
            )
            subprocess.run(
                ["dvc", "remote", "modify", "origin", "--local", "password", "dhp_PF4M7kHEXdFW8WGDGXrXnRuP"],
                cwd=self.cwd, check=True, capture_output=True, text=True
            )
            self.log.info("DVC remote 'origin' configured successfully.")
        except subprocess.CalledProcessError as e:
            # Check if the error is because the remote already exists
            if "already exists" in e.stderr:
                 self.log.warning(f"DVC remote 'origin' already configured: {e.stderr}")
            else:
                self.log.error(f"DVC remote configuration failed: {e.stderr}")
                raise

    def _read_dvc_config(self) -> None:
        """Set hardcoded DVC credentials for environment variables."""
        # These are used to set environment variables during command execution
        self.dvc_user = "fbarulli"
        self.dvc_password = "dhp_PF4M7kHEXdFW8WGDGXrXnRuP"

    def add_and_push(self, filepath: str, commit: bool = False, message: Optional[str] = None) -> None:
        """
        Adds a file to DVC and pushes changes.
        
        :param filepath: Path to the file to add relative to cwd
        :type filepath: str
        :param commit: Ignored since we're using DVC without git
        :type commit: bool
        :param message: Ignored since we're using DVC without git
        :type message: Optional[str]
        """
        env = os.environ.copy()
        # DVC should pick up credentials from the config file now
        # env['DVC_USERNAME'] = self.dvc_user
        # env['DVC_PASSWORD'] = self.dvc_password
        
        # Verify DVC remote is set up correctly
        try:
            result = subprocess.run(
                ["dvc", "remote", "list"],
                env=env,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            self.log.info(f"DVC remotes: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.log.error(f"DVC remote list error: {e.stderr}")
            raise
        
        # Add to DVC
        try:
            result = subprocess.run(
                ["dvc", "add", filepath],
                env=env,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            self.log.info(f"DVC add output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.log.error(f"DVC add error: {e.stderr}")
            raise
        
        # Push to DVC with retries
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["dvc", "push", "-v"],
                    env=env,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60
                )
                self.log.info(f"DVC push output: {result.stdout}")
                break
            except subprocess.CalledProcessError as e:
                self.log.error(f"DVC push error (attempt {attempt + 1}): {e.stderr}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
            except subprocess.TimeoutExpired:
                self.log.error(f"DVC push timeout (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
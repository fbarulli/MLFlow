from typing import Optional
from pathlib import Path
from airflow.hooks.base import BaseHook
import subprocess
import os
import time

class DVCHook(BaseHook):
    """
    Hook for interacting with DVC.
    Ensures DVC is initialized and configured before running commands.
    Configures credentials directly in the remote settings.
    """
    
    def __init__(self, cwd: Optional[str] = None) -> None:
        super().__init__()
        self.cwd = cwd or os.environ.get("AIRFLOW_HOME")
        if not self.cwd:
            raise ValueError("Could not determine working directory (cwd or AIRFLOW_HOME)")
        self.cwd_path = Path(self.cwd)
        self._setup_dvc()
        self._configure_remote_credentials()

    def _setup_dvc(self) -> None:
        """Initialize DVC and set remote URL if not already set up."""
        dvc_dir = self.cwd_path / ".dvc"
        
        if not dvc_dir.exists():
            self.log.info(f"Initializing DVC in {self.cwd} without git...")
            subprocess.run(
                ["dvc", "init", "--no-scm"], 
                cwd=self.cwd, check=True, capture_output=True, text=True
            )
        else:
            self.log.info(f"DVC already initialized in {self.cwd}")

        # Set the remote URL
        subprocess.run(
            ["dvc", "remote", "add", "--force", "-d", "origin", "https://dagshub.com/fbarulli/MLFlow.dvc"],
            cwd=self.cwd, check=True, capture_output=True, text=True
        )
        self.log.info("Ensured DVC remote 'origin' URL is set.")

    def _configure_remote_credentials(self) -> None:
        """Set DVC remote credentials using 'dvc remote modify'."""
        try:
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
            self.log.info("DVC remote credentials set successfully.")
        except subprocess.CalledProcessError as e:
            self.log.error(f"Failed to set DVC remote credentials: {e.stderr}")
            raise

    def add_and_push(self, filepath: str, commit: bool = False, message: Optional[str] = None) -> None:
        """
        Adds a file to DVC and pushes changes to the remote.
        
        :param filepath: Path to the file to add relative to cwd
        :param commit: Ignored since we're using DVC without git
        :param message: Ignored since we're using DVC without git
        """
        # Add the file to DVC
        try:
            result = subprocess.run(
                ["dvc", "add", filepath],
                cwd=self.cwd, 
                capture_output=True,
                text=True,
                check=True
            )
            self.log.info(f"DVC add output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.log.error(f"DVC add error: {e.stderr}")
            raise

        # Push to DVC remote with retries
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log.info(f"Attempt {attempt + 1} to push...")
                result = subprocess.run(
                    ["dvc", "push", "-v", "--remote", "origin"],
                    cwd=self.cwd, 
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.log.info(f"DVC push output: {result.stdout}")
                break  # Exit loop on success
            except subprocess.CalledProcessError as e:
                self.log.error(f"DVC push error (attempt {attempt + 1}): {e.stderr}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
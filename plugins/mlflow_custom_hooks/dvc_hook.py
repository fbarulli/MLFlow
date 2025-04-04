from typing import Optional
from pathlib import Path
from airflow.hooks.base import BaseHook
import subprocess
import os
import time

class DVCHook(BaseHook):
    def __init__(self, cwd: Optional[str] = None) -> None:
        super().__init__()
        self.cwd = cwd or os.environ.get("AIRFLOW_HOME")
        if not self.cwd:
            raise ValueError("Could not determine working directory (cwd or AIRFLOW_HOME)")
        self.cwd_path = Path(self.cwd)
        self._setup_dvc()
        self._configure_remote_credentials()

    def _setup_dvc(self) -> None:
        """Initialize DVC and set the remote URL if not already configured."""
        dvc_dir = self.cwd_path / ".dvc"
        if not dvc_dir.exists():
            self.log.info(f"Initializing DVC in {self.cwd} without git...")
            subprocess.run(["dvc", "init", "--no-scm"], cwd=self.cwd, check=True)
        else:
            self.log.info(f"DVC already initialized in {self.cwd}")
        # Set the remote URL (adjust the URL as needed)
        subprocess.run(
            ["dvc", "remote", "add", "--force", "-d", "origin", "https://dagshub.com/fbarulli/MLFlow.dvc"],
            cwd=self.cwd,
            check=True
        )
        self.log.info("Ensured DVC remote 'origin' URL is set.")

    def _configure_remote_credentials(self) -> None:
        """Configure authentication credentials for the 'origin' remote."""
        try:
            # Set authentication method to basic
            subprocess.run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"], cwd=self.cwd, check=True)
            # Replace 'fbarulli' and the password with your actual credentials
            subprocess.run(["dvc", "remote", "modify", "origin", "--local", "user", "fbarulli"], cwd=self.cwd, check=True)
            subprocess.run(
                ["dvc", "remote", "modify", "origin", "--local", "password", "1de275ede522e8bd56e558a81ecd32a803b7ba64"],
                cwd=self.cwd,
                check=True
            )
            self.log.info("DVC remote credentials set successfully.")
        except subprocess.CalledProcessError as e:
            self.log.error(f"Failed to set DVC remote credentials: {e.stderr}")
            raise

    def add_and_push(self, filepath: str, commit: bool = False, message: Optional[str] = None) -> None:
        """Add a file to DVC and push it to the remote."""
        try:
            # Add the file to DVC
            result = subprocess.run(["dvc", "add", filepath], cwd=self.cwd, capture_output=True, text=True, check=True)
            self.log.info(f"DVC add output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.log.error(f"DVC add error: {e.stderr}")
            raise

        # Retry logic for pushing to remote
        max_retries = 3
        retry_delay = 5
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
                break
            except subprocess.CalledProcessError as e:
                self.log.error(f"DVC push error (attempt {attempt + 1}): {e.stderr}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
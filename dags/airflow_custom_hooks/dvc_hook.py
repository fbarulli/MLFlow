from typing import Optional
import configparser
from pathlib import Path
from airflow.hooks.base import BaseHook
import subprocess
import os

class DVCHook(BaseHook):
    """
    Hook for interacting with DVC using credentials from DVC config file.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._read_dvc_config()
    
    def _read_dvc_config(self) -> None:
        """Read credentials from DVC config file"""
        config = configparser.ConfigParser()
        config_path = Path('.dvc/config')
        
        if not config_path.exists():
            raise ValueError("DVC config file not found at .dvc/config")
            
        config.read(config_path)
        
        if 'remote "origin"' not in config:
            raise ValueError("DVC remote 'origin' not configured in .dvc/config")
            
        remote_config = config['remote "origin"']
        self.dvc_user = remote_config.get('user')
        self.dvc_password = remote_config.get('password')
        
        if not self.dvc_user or not self.dvc_password:
            raise ValueError("DVC credentials not found in .dvc/config")
    
    def add_and_push(self, filepath: str, cwd: Optional[str] = None, commit: bool = True, message: Optional[str] = None) -> None:
        """
        Adds a file to DVC, pushes changes, and optionally commits to git
        
        :param filepath: Path to the file to add
        :type filepath: str
        :param cwd: Working directory where DVC commands will be executed
        :type cwd: Optional[str]
        :param commit: Whether to commit changes to git
        :type commit: bool
        :param message: Commit message if committing to git
        :type message: Optional[str]
        """
        env = os.environ.copy()
        
        # No need for setup_authentication as credentials are in .dvc/config
        
        # Add and push to DVC
        subprocess.run(["dvc", "add", filepath], env=env, cwd=cwd, check=True)
        subprocess.run(["dvc", "push"], env=env, cwd=cwd, check=True)
        
        if commit:
            if not message:
                message = f"Update {filepath}"
            
            # Add DVC file to git and commit
            dvc_file = f"{filepath}.dvc"
            subprocess.run(["git", "add", dvc_file], env=env, cwd=cwd, check=True)
            subprocess.run(["git", "commit", "-m", message, "--allow-empty"], env=env, cwd=cwd, check=True)
            subprocess.run(["git", "push"], env=env, cwd=cwd, check=True)
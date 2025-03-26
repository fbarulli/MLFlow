# /home/ubuntu/MLFlow/src/monitoring/monitor_script.py
import pandas as pd
from pathlib import Path
from typing import Optional
import mlflow
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.logger.logger import setup_logger, CSVLogHandler  

logger = setup_logger()

DATA_DIR = Path(os.getenv("DATA_DIR", "~/MLFlow/data_storage/raw")).expanduser()
DATA_PATH: Path = DATA_DIR / "weather.csv"
REFERENCE_PATH: Path = DATA_DIR / "weather_reference.csv"
REPORT_PATH: str = str(DATA_DIR / "drift_report.html")

mlflow.set_tracking_uri("http://localhost:5000")

def monitor_data() -> None:
    try:
        with mlflow.start_run():
            logger.info("Starting data monitoring")
            logger.info(f"Looking for data at: {DATA_PATH}")
            current_df: pd.DataFrame = pd.read_csv(DATA_PATH)
            current_df['datetime'] = pd.to_datetime(current_df['datetime'], errors='coerce')

            if not REFERENCE_PATH.exists():
                ref_df: pd.DataFrame = current_df.head(200)
                ref_df.to_csv(REFERENCE_PATH, index=False)
                logger.info(f"Reference data saved to {REFERENCE_PATH}")
                mlflow.log_artifact(str(REFERENCE_PATH))
                return
            
            ref_df: pd.DataFrame = pd.read_csv(REFERENCE_PATH)
            ref_df['datetime'] = pd.to_datetime(ref_df['datetime'], errors='coerce')
            
            drift_report: Report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=ref_df, current_data=current_df)
            drift_report.save_html(REPORT_PATH)
            logger.info(f"Drift report saved to {REPORT_PATH}")
            mlflow.log_artifact(REPORT_PATH)
            
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}", exc_info=True)
        raise  

if __name__ == "__main__":
    monitor_data()
import pandas as pd
from pathlib import Path
from typing import Optional
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.logger.logger import setup_logger, CSVLogHandler  

logger = setup_logger()

DATA_PATH: Path = Path("/data/weather.csv")
REFERENCE_PATH: Path = Path("/data/weather_reference.csv")
REPORT_PATH: str = "/data/drift_report.html"

def monitor_data() -> None:
    """Monitor weather data for drift and save a report."""
    try:
        with mlflow.start_run():  # Start an MLFlow run
            logger.info("Starting data monitoring")
            current_df: pd.DataFrame = pd.read_csv(DATA_PATH)
            current_df['datetime'] = pd.to_datetime(current_df['datetime'], errors='coerce')

            if not REFERENCE_PATH.exists():
                ref_df: pd.DataFrame = current_df.head(100)  # Increased for better stats
                ref_df.to_csv(REFERENCE_PATH, index=False)
                logger.info(f"Reference data saved to {REFERENCE_PATH}")
                mlflow.log_artifact(REFERENCE_PATH)  # Log reference data as an artifact
                return
            
            ref_df: pd.DataFrame = pd.read_csv(REFERENCE_PATH)
            ref_df['datetime'] = pd.to_datetime(ref_df['datetime'], errors='coerce')
            
            drift_report: Report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=ref_df, current_data=current_df)
            drift_report.save_html(REPORT_PATH)
            logger.info(f"Drift report saved to {REPORT_PATH}")
            mlflow.log_artifact(REPORT_PATH)  # Log the drift report
            
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}", exc_info=True)
        raise  

if __name__ == "__main__":
    monitor_data()
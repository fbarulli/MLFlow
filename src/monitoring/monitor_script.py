# src/monitoring/monitor_script.py
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from ..logger.logger import setup_logger

logger = setup_logger()
DATA_PATH = Path("/data/weather.csv")
REFERENCE_PATH = Path("/data/weather_reference.csv")

def monitor_data():
    logger.info("Starting data monitoring")
    current_df = pd.read_csv(DATA_PATH)
    if not REFERENCE_PATH.exists():
        current_df.head(5).to_csv(REFERENCE_PATH, index=False)  # Use first 5 rows as ref
        logger.info(f"Reference data saved to {REFERENCE_PATH}")
        return
    
    ref_df = pd.read_csv(REFERENCE_PATH)
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_df, current_data=current_df)
    drift_results = drift_report.as_dict()
    drift_score = drift_results["metrics"][0]["result"]["drift_score"]
    drifted_columns = drift_results["metrics"][1]["result"]["number_of_drifted_columns"]
    logger.info(f"Data Drift Score: {drift_score}, Drifted Columns: {drifted_columns}")

if __name__ == "__main__":
    monitor_data()
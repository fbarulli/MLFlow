# src/data_collection/weather_script.py
import pandas as pd
from typing import Dict, List, Optional
from ..logger.logger import setup_logger
from tqdm import tqdm
from ..data_class.data_class import WeatherData
from .weather_api import fetch_weather_data
from pathlib import Path

logger = setup_logger()

CITIES: List[str] = ["Paris", "London", "New York", "Berlin", "Tokyo"]
COUNTRIES: List[str] = ["FR", "GB", "US", "DE", "JP"]
OUTPUT_PATH: Path = Path("/home/ubuntu/MLFlow/data_storage/raw/weather.csv")

def to_weather_data(raw_data: Dict) -> Optional[WeatherData]:
    try:
        if not raw_data:
            raise ValueError("Empty response received from API")
        weather = WeatherData(
            city=str(raw_data["name"]),
            country=str(raw_data["sys"]["country"]),
            main_temp=float(raw_data["main"]["temp"]),
            main_pressure=int(raw_data["main"]["pressure"]),
            main_humidity=int(raw_data["main"]["humidity"]),
            weather_description=str(raw_data["weather"][0]["description"])
        )
        logger.debug(f"Parsed weather data for {weather.city}")
        return weather
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to parse weather data: {str(e)}", exc_info=True)
        return None

def process_weather_data(cities: List[str] = CITIES, countries: List[str] = COUNTRIES) -> pd.DataFrame:
    logger.info("Starting weather data processing")
    df: pd.DataFrame = pd.DataFrame({"city": cities, "country": countries})
    logger.debug("Initialized DataFrame with cities and countries")
    tqdm.pandas()
    logger.debug("Enabled tqdm for progress tracking")
    logger.info("Fetching weather data")
    df["data"] = df.progress_apply(lambda row: fetch_weather_data(row["city"], row["country"]), axis=1)
    logger.info("Parsing weather data")
    weather_objects: pd.Series = df["data"].progress_apply(to_weather_data)
    logger.debug("Converting parsed data to DataFrame")
    df_expanded: pd.DataFrame = pd.DataFrame([obj.model_dump() if obj else {} for obj in weather_objects])
    df = pd.concat([df[["city", "country"]], df_expanded], axis=1)
    logger.info(f"DataFrame shape: {df.shape}")
    na_counts = df.isna().sum().to_dict()
    logger.info(f"NA counts per column: {na_counts}")
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Data saved to {OUTPUT_PATH}")
    except (PermissionError, FileNotFoundError) as e:
        logger.error(f"Failed to save data to {OUTPUT_PATH}: {str(e)}", exc_info=True)
        raise
    logger.info("Weather data processing completed")
    return df

if __name__ == "__main__":
    result_df = process_weather_data()
    
# src/weather_script.py
import pandas as pd
from typing import Dict, List, Optional
from ..logger.logger import setup_logger
from tqdm import tqdm
from ..data_class.data_class import WeatherData  # Relative import from src/data_class
from .weather_api import fetch_weather_data

# Setup logger
logger = setup_logger()

# Constants
CITIES: List[str] = ["Paris", "London", "New York", "Berlin", "Tokyo"]
COUNTRIES: List[str] = ["FR", "GB", "US", "DE", "JP"]

# Convert API data to Pydantic model
def to_weather_data(raw_data: Dict) -> Optional[WeatherData]:
    """Parse raw API data into WeatherData model."""
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

# Main processing function
def process_weather_data(cities: List[str] = CITIES, countries: List[str] = COUNTRIES) -> pd.DataFrame:
    """Process weather data for given cities and return a DataFrame."""
    logger.info("Starting weather data processing")
    
    # Create initial DataFrame
    df: pd.DataFrame = pd.DataFrame({"city": cities, "country": countries})
    logger.debug("Initialized DataFrame with cities and countries")
    
    # Enable tqdm for pandas
    tqdm.pandas()
    logger.debug("Enabled tqdm for progress tracking")
    
    # Fetch data
    logger.info("Fetching weather data")
    df["data"] = df.progress_apply(lambda row: fetch_weather_data(row["city"], row["country"]), axis=1)
    
    # Parse data
    logger.info("Parsing weather data")
    weather_objects: pd.Series = df["data"].progress_apply(to_weather_data)
    
    # Convert to DataFrame
    logger.debug("Converting parsed data to DataFrame")
    df_expanded: pd.DataFrame = pd.DataFrame([obj.dict() if obj else {} for obj in weather_objects])
    df = pd.concat([df[["city", "country"]], df_expanded], axis=1)
    
    logger.info("Weather data processing completed")
    return df

# Execute and print
if __name__ == "__main__":
    result_df = process_weather_data()
    print(result_df)
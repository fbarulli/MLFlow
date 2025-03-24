# src/weather_api.py
import requests
from typing import Dict
from ..logger.logger import setup_logger  

logger = setup_logger()

API_URL: str = "https://api.openweathermap.org/data/2.5/weather"
API_KEY: str = "fcf2e035cc53b836623ac91cccd8848d"

def fetch_weather_data(city: str, country: str) -> Dict:
    """Fetch weather data for a given city and country."""
    try:
        query: str = f"{city},{country}"
        params: Dict[str, str] = {"q": query, "appid": API_KEY, "units": "metric"}
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        logger.info(f"Successfully fetched data for {query}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data for {query}: {str(e)}", exc_info=True)
        return {}
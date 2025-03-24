# models.py
from pydantic import BaseModel

class WeatherData(BaseModel):
    city: str
    country: str
    main_temp: float
    main_pressure: int
    main_humidity: int
    weather_description: str
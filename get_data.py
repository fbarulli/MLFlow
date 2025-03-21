# get_data.py
import requests
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from typing import List, Dict, Optional, Any
import traceback
import csv
import logging
import sys
from pydantic import BaseModel, ValidationError
from datetime import timedelta






CITIES = ["Paris", "London", "New York", "Berlin", "Tokyo"]
COUNTRIES = ["FR", "GB", "US", "DE", "JP"]
API_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "fcf2e035cc53b836623ac91cccd8848d" 



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


class WeatherData(BaseModel):
    city : str
    country : str
    main_temp : float
    main_pressure : int
    main_humidity : int

def log_to_csv(function_name: str,
               message : str,
               status : str) ->None:
    '''Logs to log_file in logs'''
    
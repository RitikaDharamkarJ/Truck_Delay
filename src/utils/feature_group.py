import os
import os.path 
import sys
import configparser
import hopsworks
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sqlalchemy import create_engine, MetaData
import warnings


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.configutils import read_config
#from src.utils.feature_group import fetch_df_from_feature_groups # Import whatever is needed

# # Add the 'src' directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from src.utils.configutils import read_config, get_connection, load_all_tables


CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
config = read_config(CONFIG_FILE_PATH)

def feature_group(engine, dataframes):
        config = read_config(CONFIG_FILE_PATH)
        api_key = config['API']['hopswork_api_key']
        config = get_connection(config)
        dataframes = load_all_tables(engine)
        engine = engine
        dataframes = dataframes
        

feature_descriptions = {
    "city_weather_fg": [
        {"name": "id", "description": "unique identification for each weather record"},
        {"name": "city_id", "description": "unique identification for each city"},
        {"name": "date", "description": "date of the weather observation"},
        {"name": "hour", "description": "hour of the weather observation (military time, 0-2300)"},
        {"name": "temp", "description": "temperature at the time of the weather observation, in Fahrenheit"},
        {"name": "wind_speed", "description": "wind speed during the observation, in miles per hour"},
        {"name": "description", "description": "brief description of the weather condition (e.g., Sunny, Cloudy)"},
        {"name": "precip", "description": "precipitation level during the observation, in inches"},
        {"name": "humidity", "description": "humidity percentage during the observation"},
        {"name": "visibility", "description": "visibility in miles at the time of the observation"},
        {"name": "pressure", "description": "atmospheric pressure at the time of the observation, in millibars"},
        {"name": "chanceofrain", "description": "chance of rain during the observation, as a percentage"},
        {"name": "chanceoffog", "description": "chance of fog during the observation, as a percentage"},
        {"name": "chanceofsnow", "description": "chance of snow during the observation, as a percentage"},
        {"name": "chanceofthunder", "description": "chance of thunder during the observation, as a percentage"},
        {"name": "event_time", "description": "dummy event time for this weather record"}
    ],
    "drivers_table_fg": [
        {"name": "driver_id", "description": "unique identification for each driver"},
        {"name": "name", "description": "name of the truck driver"},
        {"name": "gender", "description": "gender of the truck driver"},
        {"name": "age", "description": "age of the truck driver"},
        {"name": "experience", "description": "experience of the truck driver in years"},
        {"name": "driving_style", "description": "driving style of the truck driver, conservative or proactive"},
        {"name": "ratings", "description": "average rating of the truck driver on a scale of 1 to 10"},
        {"name": "vehicle_no", "description": "the number of the driver’s truck"},
        {"name": "average_speed_mph", "description": "average speed of the truck driver in miles per hour"},
        {"name": "event_time", "description": "dummy event time"}
    ],
    "trucks_table_fg": [
        {"name": "id", "description": "unique identification for each truck record"},
        {"name": "truck_id", "description": "unique identification for each truck"},
        {"name": "truck_age", "description": "age of the truck in years"},
        {"name": "load_capacity_pounds", "description": "maximum load capacity of the truck in pounds (some values may be missing)"},
        {"name": "mileage_mpg", "description": "truck's fuel efficiency measured in miles per gallon"},
        {"name": "fuel_type", "description": "type of fuel used by the truck (e.g., gas, diesel)"},
        {"name": "event_time", "description": "the timestamp when the event or record was created"}
    ],
    "routes_table_fg": [
        {"name": "id", "description": "unique identification for each route record"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "origin_id", "description": "unique identification for the origin city or location"},
        {"name": "destination_id", "description": "unique identification for the destination city or location"},
        {"name": "distance", "description": "distance between origin and destination in miles"},
        {"name": "average_hours", "description": "average travel time between origin and destination in hours"},
        {"name": "event_time", "description": "the timestamp when the event or record was created"}
    ],
    "traffic_table_fg": [
        {"name": "id", "description": "unique identification for each route activity record"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "date", "description": "date of the route activity"},
        {"name": "hour", "description": "hour of the activity (military time, e.g., 500 = 5:00 AM)"},
        {"name": "no_of_vehicles", "description": "number of vehicles on the route during the recorded hour"},
        {"name": "accident", "description": "whether an accident occurred (0 for no accident, 1 for accident)"},
        {"name": "event_time", "description": "the timestamp when the event or record was created"}
    ],
    "truck_schedule_table_fg": [
        {"name": "id", "description": "unique identification for each truck schedule record"},
        {"name": "truck_id", "description": "unique identification for each truck"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "departure_date", "description": "the departure date and time of the truck"},
        {"name": "estimated_arrival", "description": "the estimated arrival date and time of the truck"},
        {"name": "delay", "description": "whether the truck was delayed (0 for no delay, 1 for delayed)"},
        {"name": "event_time", "description": "the timestamp when the event or record was created"}
    ],
    "routes_weather_fg": [
        {"name": "id", "description": "unique identification for each weather record on the route"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "date", "description": "date and time of the weather observation"},
        {"name": "temp", "description": "temperature at the time of the weather observation, in Fahrenheit"},
        {"name": "wind_speed", "description": "wind speed during the observation, in miles per hour"},
        {"name": "description", "description": "brief description of the weather condition (e.g., Sunny, Rain Shower)"},
        {"name": "precip", "description": "precipitation level during the observation, in inches"},
        {"name": "humidity", "description": "humidity percentage during the observation"},
        {"name": "visibility", "description": "visibility in miles at the time of the observation"},
        {"name": "pressure", "description": "atmospheric pressure at the time of the observation, in millibars"},
        {"name": "chanceofrain", "description": "chance of rain during the observation, as a percentage"},
        {"name": "chanceoffog", "description": "chance of fog during the observation, as a percentage"},
        {"name": "chanceofsnow", "description": "chance of snow during the observation, as a percentage"},
        {"name": "chanceofthunder", "description": "chance of thunder during the observation, as a percentage"},
        {"name": "event_time", "description": "the timestamp when the event or record was created"}
    ]
}

def hopswork_login():
    api_key = read_config(CONFIG_FILE_PATH)['API']['hopswork_api_key']
    return hopsworks.login(api_key_value = api_key)

def fetch_df_from_feature_groups(feature_store, table_names, ver):
    """Fetch feature groups for the provided table names."""
    feature_groups = [table + '_fg' for table in table_names]
    feature_dataframes = {}
    
    for fg_name in feature_groups:
        fg = feature_store.get_feature_group(fg_name, version=ver)
        df = fg.read()
        feature_dataframes[fg_name] = df
        print(f"Fetched data for {fg_name}, number of rows: {df.shape[0]}")
    
    return feature_dataframes
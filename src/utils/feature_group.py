import os
import os.path 
import sys
import configparser
import hopsworks
import pandas as pd
import numpy as np
# import time
from datetime import datetime
from sqlalchemy import create_engine, MetaData
# import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.configutils import read_config, get_connection, load_all_tables
from src.utils.dbutils import *   # Import whatever is needed

# # Add the 'src' directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from src.utils.configutils import read_config, get_connection, load_all_tables


CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
config = read_config(CONFIG_FILE_PATH)


class fgIngestion:
    def __init__(self, engine, dataframes):
        self.config = read_config(CONFIG_FILE_PATH)
        self.api_key = self.config['API']['hopswork_api_key']
        self.config = get_connection(self.config)
        self.dataframes = load_all_tables(self.engine)
        self.engine = engine
        self.dataframes = dataframes
        self.project = None

def hopswork_login(self):
        return hopsworks.login(api_key_value = self.api_key)
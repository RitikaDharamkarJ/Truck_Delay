import os
import sys
import pandas as pd
import numpy as np
import configparser
import hopsworks
from hsfs.client.exceptions import RestAPIError
# from src.utils.feature_group import hopswork_login, fetch_df_from_feature_groups

class DataTransformation:
    def __init__(self, project):
        self.config = self.load_config()
        # self.project = hopsworks.login(api_key_value=self.config['HOPSWORK']['hopsworks_api_key'])
        # self.fs = self.project.get_feature_store()
        # self.feature_group_data = self.get_features()
        self.project = project
        self.fs = self.project.get_feature_store()
        
        
    def load_config(self):
        """Load configuration from the config file."""
        config = configparser.RawConfigParser()
        CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
        config.read(CONFIG_FILE_PATH)
        return config
    
    def hopswork_login(config):
        """Logs into Hopsworks using the API key from the config."""
        api_key = config['HOPSWORK']['hopsworks_api_key']
        return hopsworks.login(api_key_value=api_key)
    
    
    def connect_and_download_feature_group(api_key, feature_group_name, version):
        try:
            # Establish connection to Hopsworks
            project = hopsworks.login(api_key_value=api_key)
            print("Successfully connected to Hopsworks.") 
            
            # Access the Feature Store
            fs = project.get_feature_store()

            # Retrieve the feature group by name and version
            feature_group = fs.get_feature_group(feature_group_name, version=version)

            # Read the feature group as a DataFrame
            df = feature_group.read()

            # Print confirmation and return the DataFrame
            print(f"Downloaded feature group: {feature_group_name} (version {version})")
            return df

        except RestAPIError as e:
            print(f"Error downloading feature group: {feature_group_name} (version {version})")
            print(e)
        except Exception as e:
            print("An unexpected error occurred.")
            print(e)
            return None
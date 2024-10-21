import configparser
import sys
import os
import hopsworks
import pandas as pd

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

from src.components.data_transformation import DataTransformation
from src.utils.configutils import read_config

# Initialize configparser
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'

# Class for Data Merging Pipeline
class DataTransformationPipeline:
    def __init__(self):
        # Load config and initialize DataMerge
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE_PATH)
        self.db_config = read_config(CONFIG_FILE_PATH)

        # Log into Hopsworks
        self.project = hopsworks.login(api_key_value=self.config['HOPSWORK']['hopsworks_api_key'])

        # Initialize DataMerge object
        self.data_transformation_obj = DataTransformation(self.project)
    
    def main(self):
        print(">>> Stage started: Data Transformation <<<")
        
        #Step 1
        # Fetch feature groups via DataMerge class
        print("Fetch feature groups via DataTransformation class...")
        feature_group_data = self.data_transformation_obj.get_features()
        

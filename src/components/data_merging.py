import os
import sys
import pandas as pd
import requests
from sqlalchemy import create_engine
import configparser
import hopsworks
from hsfs.client.exceptions import RestAPIError

config = configparser.RawConfigParser()  # Initialize config object


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.configutils import read_config, get_connection, load_all_tables


# Read configuration from the config file
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
STAGE_NAME = "Data Merging"
config = read_config(CONFIG_FILE_PATH)


# Fetch API key and project name from the config file
hopsworks_api_key = config.get('API', 'hopsworks_api_key')
project_name = config.get('API', 'project_name')

# Establish connection to Hopsworks
project = hopsworks.login(
    api_key_value=hopsworks_api_key  # Using the API key from the config file
)
# Set your project name
project_name = project_name 
# # Set your project name
# project_name = "TruckDelay_Pipelines"  # Replace with your actual project name

class DataMerge:
    def __init__(self, engine, dataframes):
        self.engine = engine
        self.dataframes = dataframes
        self.project = project_name
    
        # Hopsworks project login
    project = hopsworks.login()
    fs = project.get_feature_store()

    
    #Step 1
    def get_feature_store(project_name, api_key):
        print('importing the feature group')
        
      
    #Step 2  
    def get_features():
         
        # Get the feature store
        # Access the Feature Store
        fs = project.get_feature_store()

        # List of feature groups and their versions
        feature_groups = [
            {"name": "city_weather", "version": 1},
            {"name": "drivers_table", "version": 1},
            {"name": "routes_table", "version": 1},
            {"name": "routes_weather", "version": 1},
            {"name": "traffic_table", "version": 1},  # Corrected typo
            {"name": "truck_schedule_table", "version": 1},
            {"name": "trucks_table", "version": 1},
        ]
        # Initialize an empty dictionary to hold DataFrames for each feature group
        feature_group_data = {}

        # Loop over the list of feature groups and download them
        for fg in feature_groups:
            try:
                # Retrieve the feature group by its name and version
                feature_group = fs.get_feature_group(fg['name'], version=fg['version'])
                df = feature_group.read()  # Read the feature group as a DataFrame
                
                # Save the DataFrame in the dictionary with the feature group's name as the key
                feature_group_data[fg['name']] = df

                print(f"Downloaded feature group: {fg['name']} (version {fg['version']})")
                print(df.head())  # Optionally display the first few rows of the DataFrame

            except RestAPIError as e:
                print(f"Error downloading feature group: {fg['name']} (version {fg['version']})")
                print(e)
 
 
    #Step 3
    def assign_feature_group_data_to_dfs(feature_group_data):
        # Create a dictionary to store all DataFrames
        dfs = {
            'city_weather': feature_group_data['city_weather'],
            'drivers_table': feature_group_data['drivers_table'],
            'routes_table': feature_group_data['routes_table'],
            'routes_weather': feature_group_data['routes_weather'],
            'traffic_table': feature_group_data['traffic_table'],
            'truck_schedule_table': feature_group_data['truck_schedule_table'],
            'trucks_table': feature_group_data['trucks_table']
        }
        
        # Return the dfs dictionary
        return dfs
    
    
    # def extract_tables(feature_group_data):
        
    #         city_weather = feature_group_data['city_weather']
    #         drivers_table = feature_group_data['drivers_table']
    #         routes_table = feature_group_data['routes_table']
    #         routes_weather = feature_group_data['routes_weather']
    #         traffic_table = feature_group_data['traffic_table']
    #         truck_schedule_table = feature_group_data['truck_schedule_table']
    #         trucks_table = feature_group_data['trucks_table']
            
    #         return {
    #             'city_weather': city_weather,
    #             'drivers_table': drivers_table,
    #             'routes_table': routes_table,
    #             'routes_weather': routes_weather,
    #             'traffic_table': traffic_table,
    #             'truck_schedule_table': truck_schedule_table,
    #             'trucks_table': trucks_table
    # }
    
    
    # def assign_feature_group_data(feature_group_data, table_names):
    #         # Dynamically assign the DataFrame from feature_group_data to the variable names provided in table_names
    #         for table_name in table_names:
    #             globals()[table_name] = feature_group_data[table_name]
    #             print(f"Assigned {table_name} to variable.")
                
    #         # List of table names to be assigned
    #         table_names = ['city_weather', 'routes_weather', 'traffic_table', 'truck_schedule_table', 'trucks_table']
     
     
    #Step 4 
    # Updated function to drop multiple columns from the DataFrames
    def drop_columns_from_dfs(dfs, columns):
        for df_name, df in dfs.items():
            columns_to_drop = [col for col in columns if col in df.columns]
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                print(f"Columns {columns_to_drop} dropped from {df_name}")
            else:
                print(f"No specified columns found in {df_name}")

    
    #Step 5
    #Drop the duplicate values from the specified columns in the dataframes
    def drop_duplicates_from_dfs(dfs, columns_to_check):   
        for df_name, df in dfs.items():
            if df_name in columns_to_check:
                subset_columns = columns_to_check[df_name]
                before_dropping = df.shape[0]  # Count the number of rows before dropping duplicates
                df.drop_duplicates(subset=subset_columns, inplace=True)
                after_dropping = df.shape[0]  # Count the number of rows after dropping duplicates
                print(f"Dropped {before_dropping - after_dropping} duplicate rows from {df_name} based on columns: {subset_columns}")
            else:
                print(f"No duplicate check columns provided for {df_name}")
            
     
     #Step 6
     #code to drop the unnecessary columns in the city_weather dataframe
     
    def drop_weather_columns_from_dfs(dfs, table_names, columns_to_drop):
        for table_name in table_names:
            if table_name in dfs:
                dfs[table_name].drop(columns=columns_to_drop, inplace=True)
                print(f"Columns {columns_to_drop} dropped from {table_name}.")
            else:
                print(f"Table {table_name} not found in dfs.")
    
                
    #Step 7 
    
    def rename(dfs,table_names):
       for table_name in table_names:
            if table_name in dfs:
              table_name.rename(columns={'date': 'custom_date'}, inplace=True)
            else:
                print(f"Table {table_name} not found in dfs.")
     
     
    #Step 8
    
   # Merge the resulant data frame with route_weather on route_id and date (left)

    def process_dates(df, estimated_arrival_column, departure_date_column):
        # Function to convert to UTC if not already timezone-aware
        def convert_to_utc(series):
            if series.dt.tz is None:  # Check if the series is not timezone-aware
                return series.dt.tz_localize('UTC')
            else:
                return series.dt.tz_convert('UTC')  # Convert to UTC if it already has a timezone

        # Step 1: Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Step 2: Convert both columns to datetime format and set to UTC timezone
        df_copy[estimated_arrival_column] = convert_to_utc(pd.to_datetime(df_copy[estimated_arrival_column], errors='coerce'))
        df_copy[departure_date_column] = convert_to_utc(pd.to_datetime(df_copy[departure_date_column], errors='coerce'))

        # Step 3: Apply ceil operation to estimated arrival
        df_copy[estimated_arrival_column] = df_copy[estimated_arrival_column].dt.ceil("6H")

        # Step 4: Floor the 'departure_date' column to the nearest 6 hours
        df_copy[departure_date_column] = df_copy[departure_date_column].dt.floor("6H")

        # Step 5: Create a new column 'date' with date ranges between 'departure_date' and 'estimated_arrival'
        df_copy['date'] = [
            pd.date_range(start=row[departure_date_column], end=row[estimated_arrival_column], freq='6H')
            for index, row in df_copy.iterrows()
        ]
        
        # Step 6: Explode the 'date' column to separate rows for each date in the range
        df_copy = df_copy.explode('date').reset_index(drop=True)

        return df_copy 
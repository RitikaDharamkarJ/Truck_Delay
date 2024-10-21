import configparser  # Ensure this is imported
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import hopsworks  # Import Hopsworks
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.configutils import *

CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
# db_config = read_config(CONFIG_FILE_PATH)


class DataClean:
    def __init__(self, engine, dataframes):
        self.engine = engine
        self.dataframes = dataframes
        self.project = None

    def fill_missing_values(self, df):
        """Fill missing values in the DataFrame based on data type."""
        if df.empty:
            print("Warning: The DataFrame is empty. No missing values to fill.")
            return df
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            
            if df[column].dtype in [np.int64, np.float64]:  # Numerical columns
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                print(f"Filled {missing_count} missing values in numerical column '{column}' with mean value {mean_value}.")
            
            elif df[column].dtype == 'object':  # Categorical columns
                mode_value = df[column].mode()[0] if not df[column].mode().empty else None
                df[column].fillna(mode_value, inplace=True)
                print(f"Filled {missing_count} missing values in categorical column '{column}' with mode value '{mode_value}'.")
            
            elif np.issubdtype(df[column].dtype, np.datetime64):  # Date columns
                df[column].fillna(pd.Timestamp('1970-01-01'), inplace=True)
                print(f"Filled {missing_count} missing values in datetime column '{column}' with default date '1970-01-01'.")
        
        return df

    def remove_outliers(self, df, outlier_columns):
        """Remove outliers using the IQR method."""
        for column in outlier_columns:
            if column in df.columns:
                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1  # Interquartile range

                # Determine bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter the DataFrame to remove outliers
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        return df

    def add_rowindex_event_time(self, df):
        """Adds RowIndex and event_time columns to the DataFrame."""
        df.insert(0, 'rowindex', range(1, 1 + len(df)))
        # df['event_time'] = pd.to_datetime('2024-09-24')
        return df

    def fix_negative_values(self, df, column):
        """Fix negative values by replacing them with the mean."""
        mean_value = df[df[column] >= 0][column].mean()
        df[column] = df[column].apply(lambda x: mean_value if x < 0 else x)
        return df

    def merge_date_hour(self, df, table_name, date, hour):
        """Merge date and hour columns into a single datetime column for the specified table."""
        # Check if the DataFrame is for city_weather table
            # Ensure hour column is formatted as two digits
        #df[hour] = df[hour].apply(lambda x: f'{int(x):02d}' if pd.notnull(x) else '00')
        df[hour] = df[hour].apply(lambda x: f"{x // 100:02}:{x % 100:02}:00")
            
            # Create datetime column by combining date and formatted hour
        df[date] = pd.to_datetime(df[date] + ' ' + df[hour], errors='coerce')
            
            # Drop the hour column as it's no longer needed
        df.drop(columns=[hour], inplace=True)
        
        return df
    
    # def convert_date_to_datetime(df, self, Date, table_name):
    #     # Check if the DataFrame is for routes_weather table
    #     if table_name == 'routes_weather':
    #         df[Date] = pd.to_datetime(df[Date], errors='coerce')
    #     return df
    
    def convert_date_to_datetime(self, df, table_name, date_col):
       df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
       return df


    def converted_estimated_arrival(self,df, table_name, estimated_arrival):
            
        # Convert 'estimated_arrival' to datetime format if not already
            df[estimated_arrival] = pd.to_datetime(df[estimated_arrival], errors='coerce')

         # Set minutes and seconds to '00:00:00' and floor the datetime to the nearest hour
            df[estimated_arrival] = df[estimated_arrival].dt.floor('H')
      
            return df


    def create_feature_groups(self, cleaned_dataframes, config):
        """Function to create feature groups from cleaned DataFrames in Hopsworks, including eventtime column."""
        # Log into Hopsworks using API key from the config
        if self.project is None:  # Log in if not already done
            self.project = hopsworks.login(api_key_value=config['API']['hopsworks_api_key'])
        fs = self.project.get_feature_store()  # Access the feature store

        table_names = [
            'city_weather',
            'drivers_table',
            'routes_table',
            'routes_weather',
            'traffic_table',
            'truck_schedule_table',
            'trucks_table'
        ]

        # Iterate over cleaned DataFrames and create feature groups
        for table_name in table_names:
            if table_name in cleaned_dataframes:
                print(f"Creating feature group for table: {table_name}")
                
                df = cleaned_dataframes[table_name]

                #Ensure the DataFrame has an eventtime column. If not, add it manually.
                if 'eventtime' not in df.columns:
                    df['eventtime'] = pd.to_datetime('now')

                #Convert date and eventtime to proper datetime format
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if 'eventtime' in df.columns:
                    df['eventtime'] = pd.to_datetime(df['eventtime'], errors='coerce')

                # Ensure 'hour' is of type bigint (int64)
                # if 'hour' in df.columns:
                #     df['hour'] = df['hour'].astype('int64')

                # # List of required columns based on feature group schema
                # required_columns = ['date',  'eventtime']
                
                # # Check for missing columns and keep only the required columns
                # for col in required_columns:
                #     if col not in df.columns:
                #         print(f"Warning: Column {col} is missing from {table_name}. It will be dropped.")
                        
                # df = df[required_columns]  # Retain only the required columns

                try:     # Create feature group
                    feature_group = fs.get_or_create_feature_group(name=table_name,version=1)
                    print(f"Updating feature group '{table_name}' with new data.")
                    feature_group.insert(df, write_options={"upsert": True})
                except Exception as e:
                      print(f"Creating new feature group for '{table_name}'.")
                      feature_group = fs.create_feature_group(
                                name=table_name,
                                version=1,
                                primary_key=['rowindex'],
                                description=f"Features for {table_name}",
                                event_time='eventtime',
                            )
                
                
                #     primary_key=['rowindex'],  # Adjust primary key based on your data
                #     event_time='eventtime',
                #     description=f"{table_name} feature group with eventtime"
                # )

                # Insert the DataFrame into the feature group
                try:
                    feature_group.insert(df, write_options={"wait_for_job": False})
                    print(f"Successfully inserted data into feature group: {table_name}")
                except Exception as e:
                    print(f"Error inserting data into feature group {table_name}: {e}")


    # def create_or_update_feature_groups_in_hopsworks(self, cleaned_data, config):
    #         try:
    #             # Log into Hopsworks using the API key
    #                 #         self.project = hopsworks.login(api_key_value=config['API']['hopsworks_api_key'])

    #             project = hopsworks.login(api_key_value=config.hopsworks_api_key)
    #             fs = project.get_feature_store()

    #             for table_name, df in cleaned_data.items():
    #                 if 'eventtime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['eventtime']):
    #                     df['eventtime'] = df['eventtime'].dt.date

    #                     try:
    #                         feature_group = fs.get_feature_group(name=table_name, version=1)
    #                         print(f"Updating feature group '{table_name}' with new data.")
    #                         feature_group.insert(df, write_options={"upsert": True})
    #                     except Exception as e:
    #                         print(f"Creating new feature group for '{table_name}'.")
    #                         feature_group = fs.create_feature_group(
    #                             name=table_name,
    #                             version=1,
    #                             primary_key=['index'],
    #                             description=f"Features for {table_name}",
    #                             event_time='eventtime',
    #                             online_enabled='False'
    #                         )
    #                         feature_group.insert(df)
    #                         print(f"Feature group '{table_name}' created with initial data.")
    #         except Exception as e:
    #             print(f"Error during feature group management: {e}")
    #             raise
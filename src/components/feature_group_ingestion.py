# data_cleaning_pipeline.py

import configparser
import os.path as path
import sys
import os

parent_directory = os.path.abspath(path.join(__file__, "../../"))
sys.path.append(parent_directory)

from src.components.data_cleaning import DataClean
from src.components.feature_group_component import create_feature_group  # Assuming you have this import

# Read configuration from the config file
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
config = configparser.RawConfigParser()  # Initialize config object

STAGE_NAME = "Data Cleaning and Feature Group Creation"

class DataCleaningPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize DataClean component
            cleaner = DataClean(CONFIG_FILE_PATH)

            # Define the outlier columns for each table
            outlier_columns = {
                'city_weather': ['temp', 'humidity', 'visibility', 'pressure'],
                'routes_weather': ['temp', 'wind_speed', 'precip','humidity', 'pressure'],
                'drivers_table': ['experience','age','ratings','average_speed_mph'],
                'traffic_table': ['no_of_vehicles'],
                'routes_tables': ['average_hours','distance'],
                'truck_schedule': ['delay'],
                'trucks_table': ['mileage_mpg','load_capacity_pounds','truck_age']
            }

            # Define columns to drop for each table
            columns_to_drop_dict = {
                'city_weather': ['chanceofrain','chanceoffog','chanceofsnow','chanceofthunder'],
                'routes_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder'],
            }

            # Perform data cleaning
            cleaned_dataframes = cleaner.clean_data(outlier_columns, columns_to_drop_dict)

            # Pass the cleaned dataframes to the ingestion step
            self.create_feature_groups(cleaned_dataframes)

        except Exception as e:
            print(f"Error during data cleaning or feature group creation: {e}")
            raise e

    def create_feature_groups(self, cleaned_dataframes):
        """
        Function to create feature groups from cleaned DataFrames.
        """
        table_names = [
            'city_weather',
            'drivers_table',
            'routes_tables',
            'routes_weather',
            'traffic_table',
            'truck_schedule',
            'trucks_table'
        ]

        # Create feature groups using the cleaned DataFrames
        for table_name in table_names:
            if table_name in cleaned_dataframes:
                create_feature_group(
                    table_name=table_name,
                    df=cleaned_dataframes[table_name],
                    primary_key=['RowIndex'],  # Adjust primary key if necessary
                    version=1,
                    description=f"{table_name} feature group"
                )


if __name__ == '__main__':
    try:
        print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")
        obj = DataCleaningPipeline()
        obj.main()
        print(f">>>>>> Stage completed <<<<<< : {STAGE_NAME}")
    except Exception as e:
        print(e)
        raise e

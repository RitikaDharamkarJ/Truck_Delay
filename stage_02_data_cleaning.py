import configparser
import os.path as path
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.configutils import read_config, get_connection, load_all_tables
from src.components.data_cleaning import DataClean

# Read configuration from the config file
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'

STAGE_NAME = "Data Cleaning"

class DataCleaningPipeline:
    def __init__(self):
        # Load config and database connection
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE_PATH)
        self.db_config = read_config(CONFIG_FILE_PATH)
        self.engine = get_connection(self.db_config)

        # Load all tables into DataFrames
        self.dataframes = load_all_tables(self.engine)


        # Initialize the DataClean component
        self.cleaner = DataClean(self.engine, self.dataframes)

    def main(self):
        try:
            # Step 1: Load data from database (already loaded in __init__)
            print("Data loaded successfully from database.")
            
            # Step 2: Replace missing values
            print("Replacing missing values...")
            for table_name, df in self.dataframes.items():
                self.dataframes[table_name] = self.cleaner.fill_missing_values(df)
                


            # Step 3: Add RowIndex and event_time columns
            print("Adding RowIndex and event_time columns...")
            for table_name, df in self.dataframes.items():
                self.dataframes[table_name] = self.cleaner.add_rowindex_event_time(df)
               


            # Step 4: Fix negative values in certain columns
            print("Fixing negative values in specific columns...")
            # Specify which columns in each table to check for negative values
            negative_value_columns = {
                'drivers_table': 'experience',
                'trucks_table': 'mileage_mpg'
            }
            for table_name, column in negative_value_columns.items():
                if table_name in self.dataframes:
                    self.dataframes[table_name] = self.cleaner.fix_negative_values(self.dataframes[table_name], column)
            
                    
           
            # # Step 5: merging the row date and hour column
            print("Merging the row date and hour columns and converting them into datetime format...")

            # Loop over the dataframes
            for table_name, df in self.dataframes.items():
                # Define the date and hour column names based on the table
                if table_name == 'city_weather':
                    date = 'date'  # Specify the correct date column for city_weather
                    hour = 'hour'  # Specify the correct hour column for city_weather
                elif table_name == 'traffic_table':
                    date = 'date'  # Specify the correct date column for traffic_table
                    hour = 'hour'  # Specify the correct hour column for traffic_table
                else:
                    # Skip tables that do not require date/hour merging
                    print(f"Skipping table: {table_name}")
                    continue  # Skip the current iteration for tables that don't require merging
             
                # Call the merge_date_hour function with all required arguments
                self.dataframes[table_name] = self.cleaner.merge_date_hour(df, table_name, date, hour)

            #Step 6: 
            print("Converting the date column to datetime format...")
             # Loop over the dataframes
            for table_name, df in self.dataframes.items():
                if table_name == 'routes_weather':
                    # Specify the correct date column for routes_weather
                    date_col = 'Date'
                    # Call the convert_date_to_datetime function
                    self.dataframes[table_name] = self.cleaner.convert_date_to_datetime(df, table_name, date_col)
                else:
                    # Skip tables that don't require this conversion
                    print(f"Skipping table: {table_name}")
                    
            for table_name, df in self.dataframes.items():
               print(f"Table: {table_name}")
               print(df.head())  # Print the first 5 rows of each DataFrame
               print("\n")  # Add a newline for better readability
        
                    
            # #Step 7: 
            print("Converting the 'estimated_arrival' column to datetime format...")
            # Define a list of tables that require the conversion
            target_tables = ['truck_schedule_table']
            for table_name, df in self.dataframes.items():
                if table_name in target_tables:
                    estimated_arrival = 'estimated_arrival'  
                    self.dataframes[table_name] = self.cleaner.converted_estimated_arrival(df, table_name, estimated_arrival)
                else:
                    # Skip tables that don't require this conversion
                    print(f"Skipping table: {table_name}")

 
            # Step 8: Remove outliers from specific columns
            print("Removing outliers from specific columns...")
            outlier_columns = {
                    'city_weather': ['temp','wind_speed', 'humidity', 'pressure'],
                    'routes_weather': ['temp', 'wind_speed', 'humidity', 'pressure'],
                    'drivers_table': ['experience', 'age', 'ratings', 'average_speed_mph'],
                    'traffic_table': ['no_of_vehicles'],
                    'routes_table': ['average_hours', 'distance'],
                    'trucks_table': ['mileage_mpg', 'load_capacity_pounds', 'truck_age'], 
            }

            # Iterate over the outlier columns
            for table_name, columns in outlier_columns.items():
                 if table_name in self.dataframes:
                # Save the original DataFrame for comparison
                   original_shape = self.dataframes[table_name].shape

                # Remove outliers using the IQR method
                   cleaned_df = self.cleaner.remove_outliers(self.dataframes[table_name], columns)

                # Check if the cleaned DataFrame is empty
                 if cleaned_df.empty:
                   print(f"Warning: All values turned to NaN for {table_name} after outlier removal.")
                 else:
                   self.dataframes[table_name] = cleaned_df
                   print(f"Removed outliers from {table_name}. Original shape: {original_shape}, New shape: {self.dataframes[table_name].shape}")
 

            # Step 9: Create feature groups in Hopsworks
            print(self.dataframes)
            print("Creating feature groups in Hopsworks...")
            # self.cleaner.create_feature_groups(self.dataframes, self.config)
            self.cleaner.create_feature_groups(self.dataframes, self.config)

        except Exception as e:
            print(f"Error during data cleaning pipeline: {e}")
            raise e


if __name__ == '__main__':
    try:
        print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")
        pipeline = DataCleaningPipeline()
        pipeline.main()
        print(f">>>>>> Stage completed <<<<<< : {STAGE_NAME}")
    except Exception as e:
        print(e)
        raise e

import configparser
import sys 
import os
import hopsworks
from hsfs.client.exceptions import RestAPIError
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.configutils import read_config, get_connection
from src.components.data_merging import DataMerge

# Read configuration from the config file
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
datamerge_obj=DataMerge()
 # Initialize the configparser to read the configuration
config = configparser.ConfigParser()
config.read(CONFIG_FILE_PATH)

# Fetch API key and project name from the config file
hopsworks_api_key = config.get('API', 'hopsworks_api_key')
project_name = config.get('API', 'project_name')

# Establish connection to Hopsworks
project = hopsworks.login(
    api_key=hopsworks_api_key  # Using the API key from the config file
)

# Set your project name
project_name = project_name  

# # Establish connection to Hopsworks
# project = hopsworks.login(
#     api_key_value="O4IOxWozstKu0BFQ.07C1tbvgVI5C4XNLbLrGH4PS4t0EqBYN00ex8318TNIkl82WwDi3Vh9MidMrCA83"
# )
# # Set your project name
# project_name = "TruckDelay_Pipelines"  # Replace with your actual project name

STAGE_NAME = "Data Merging"


class DataMergingPipeline:
    def __init__(self):
        # Load config and database connection
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE_PATH)
        self.db_config = read_config(CONFIG_FILE_PATH)


    def main(self):
        
        #Step 1
        #connect with the feature store
        print("Feature store connection started. Entering pipeline.")

        # Call the function to get the feature store
        feature_group = datamerge_obj.get_feature_store(project_name, api_key)
         
        #Step 2 
        #get the feature groups
        
        feature_group= datamerge_obj.get_features()
        
        #Step 3        
        #assign the tables in feature_group_data to the different names
         
        # Call the function and assign it to dfs
        dfs = datamerge_obj.assign_feature_group_data_to_dfs(feature_group_data)        
        
        # Step 4
        #drop the event_time, eventtime and rowindex columns in all the data frames
        # List of columns to drop
        columns_to_remove = ['rowindex', 'event_time', 'eventtime']
        
        # Call the function to drop the specified columns from all DataFrames
        feature_group= datamerge_obj.drop_columns_from_dfs(dfs, columns_to_remove)
 
        
        #Step 5
        #Drop the duplicate values from the specified columns in the dataframes
        # Dictionary where the key is the DataFrame name, and the value is the subset of columns to check for duplicates
        columns_to_check = {
            'city_weather': ['city_id', 'date'],
            'routes_weather': ['route_id', 'date'],
            'trucks_table': ['truck_id'],
            'drivers_table': ['driver_id'],
            'routes_table': ['route_id', 'destination_id', 'origin_id'],
            'truck_schedule': ['truck_id','route_id','departure_date']
        }
        
        # Call the function to drop the specified columns from all DataFrames
        feature_group=datamerge_obj.drop_duplicates_from_dfs(dfs, columns_to_check)
        # Check the result by printing the modified DataFrames

        
        #Step 6
        #code to drop the unnecessary columns in the city_weather dataframe
        # List of columns to drop
        columns_to_drop = ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder']
        # List of tables from which you want to drop columns
        table_names = ['city_weather', 'routes_weather']
        
        # Call the function
        feature_group=datamerge_obj.drop_weather_columns_from_dfs(dfs, table_names, columns_to_drop)
        
        #Step 7
        # Rename the 'date' column to 'custom_date'
        table_names = ['city_weather']
         
        #Call the function
        feature_group=datamerge_obj.rename(dfs, table_names) 
        
        
        #Step 8
        
        # Merge the resulant data frame with route_weather on route_id and date (left)
        
        # Define target tables
        target_tables = ['truck_schedule_table']
        departure_date_column = 'departure_date'
        estimated_arrival_column = 'estimated_arrival'

        # Process the DataFrames
        for table_name in list(dfs.keys()):  # Use list(dfs.keys()) to avoid changing the dictionary during iteration
            df = dfs[table_name]
            if table_name in target_tables:
                # Create a copy and process dates
                result_df_copy = datamerge_obj.process_dates(df, estimated_arrival_column, departure_date_column)
                
                # Merge with route_weather on 'route_id' and 'date'
                scheduled_weather = pd.merge(result_df_copy, routes_weather, on=['route_id', 'date'], how='left')

                # Display the merged DataFrame
                print(scheduled_weather)

                # Save the processed DataFrame under a new key
                dfs[f'{table_name}_processed'] = result_df_copy

        
        

if __name__ == '__main__':
    try:
        print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")
        pipeline = DataMergingPipeline()
        pipeline.main()
        print(f">>>>>> Stage completed <<<<<< : {STAGE_NAME}")
    except Exception as e:
        print(e)
        raise e

  
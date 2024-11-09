import configparser
import sys
import os
import hopsworks
import pandas as pd

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

from src.components.data_merging import DataMerge
from src.utils.configutils import read_config

# Initialize configparser
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'

# Class for Data Merging Pipeline
class DataMergingPipeline:
    def __init__(self):
        # Load config and initialize DataMerge
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE_PATH)
        self.db_config = read_config(CONFIG_FILE_PATH)

        # Log into Hopsworks
        self.project = hopsworks.login(api_key_value=self.config['HOPSWORK']['hopsworks_api_key'])

        # Initialize DataMerge object
        self.data_merge_obj = DataMerge(self.project)
    
    def main(self):
        print(">>> Stage started: Data Merging <<<")
        
        #Step 1
        # Fetch feature groups via DataMerge class
        print("Fetch feature groups via DataMerge class...")
        feature_group_data = self.data_merge_obj.get_features()
        
        #Step 2
        # Assign each DataFrame to the 'dfs' dictionary with a new name
        print("Assign feature groups to DataFrames and save them in a dictionary")
        dfs = self.data_merge_obj.assign_feature_group_data(feature_group_data)

        #Step 3
        # Drop unnecessary columns
        print("Dropping unnecessary columns like rowindex, event_time and eventtime")
        columns_to_remove = ['rowindex', 'event_time', 'eventtime']
        dfs = self.data_merge_obj.drop_columns_from_dfs(feature_group_data, columns_to_remove)

        #Step 4
        # Drop duplicates based on specified columns
        print("Dropping duplicates in specified rows...")

        columns_to_check = {
            'city_weather': ['city_id', 'date'],
            'routes_weather': ['route_id', 'date'],
            'trucks_table': ['truck_id'],
            'drivers_table': ['driver_id'],
            'routes_table': ['route_id', 'destination_id', 'origin_id'],
            'truck_schedule_table': ['truck_id', 'route_id', 'departure_date']
        }
        dfs = self.data_merge_obj.drop_duplicates_from_dfs(dfs, columns_to_check)

        #Step 5
        # Define weather columns to drop
        print("Dropping unnecessary columns like in weather tables")
        weather_columns_to_drop = {
            'city_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder'],
            'routes_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder']
        }
        # Drop weather-related columns
        dfs = self.data_merge_obj.drop_weather_columns_from_dfs(dfs, weather_columns_to_drop)

        #Step 6
        # Rename 'date' to 'custom_date' in city_weather
        print("Renaming the date column in city_weather...")
        dfs = self.data_merge_obj.rename(dfs, ['city_weather'])
        
        #Step 7
         # Process dates and merge DataFrames
        print("Merging the tables routes_weather and truck_schedule_table...")
                # Define target tables
        target_tables = ['truck_schedule_table']
        departure_date_column = 'departure_date'
        estimated_arrival_column = 'estimated_arrival'

        # Process the DataFrames
        for table_name in list(dfs.keys()):  # Use list(dfs.keys()) to avoid changing the dictionary during iteration
            df = dfs[table_name]
            if table_name in target_tables:
                # Create a copy and process dates
                result_df_copy = self.data_merge_obj.process_dates(df, estimated_arrival_column, departure_date_column)
                
                # Merge with route_weather on 'route_id' and 'date'
                scheduled_weather = pd.merge(result_df_copy, dfs['routes_weather'], on=['route_id', 'date'], how='left')

                # Display the merged DataFrame
                # print(scheduled_weather)

                # # Save the processed DataFrame under a new key
                # dfs[f'{table_name}_processed'] = result_df_copy
 
        #Step 8
        # Define a custom function to calculate mode with error handling
        print("Define a custom function to calculate mode with error handling...")
        group_by_columns = ['truck_id','route_id']
        # Call the group_and_aggregate_weather_data function
        scheduled_weather_grp = self.data_merge_obj.group_and_aggregate_weather_data(scheduled_weather, group_by_columns)

        #Step 8.5
        #Merge schedule df with schedule_weather_grp df
        print("Merge schedule df with schedule_weather_grp df...")
        schedule_weather_merge=dfs['truck_schedule_table'].merge(scheduled_weather_grp,on=['truck_id','route_id'],how='left')

        #Step 9
        # Find Origin and Destination city Weather
        print("Find Origin and Destination city Weather..")
        truck_schedule_df=dfs['truck_schedule_table']
        routes_df = dfs['routes_table']
        dfs['routes_table']
        nearest_hour_schedule_route_df = self.data_merge_obj.round_schedule_times_and_merge(truck_schedule_df, routes_df)

        #Step 9.5
        # Assume 'city_weather' is your original DataFrame
        city_weather=dfs['city_weather']
        origin_weather_data= city_weather.copy()  # First copy
        destination_weather_data = city_weather.copy()  # Second copy

        ###****** Create a copy of the 'weather_df' DataFrame for manipulation
        #             * Drop the 'date' and 'hour' columns from 'origin_weather_data'
        #             * Create a copy of the 'weather_df' DataFrame for manipulation
        #             * Drop the 'date' and 'hour' columns from 'destination_weather_data'
        #             * Merge 'nearest_hour_schedule_route_df' with 'origin_weather_data' based on specified columns
        #             * Merge 'origin_weather_merge' with 'destination_weather_data' based on specified columns

        # so basically here I already dropped hour column by merging it into date column and later renamed it to customdate column. Hence nothing left here to drop #####*****  
        #Step 10
        ##Merge 'nearest_hour_schedule_route_df' with 'origin_weather_data' based on specified columns
        print("Merge 'nearest_hour_schedule_route_df' with 'origin_weather_data' based on specified columns...")
        origin_weather_merge = self.data_merge_obj.merge_schedule_with_weather(nearest_hour_schedule_route_df, origin_weather_data)


        #Step 11
        print("Merge 'origin_weather_merge' with 'destination_weather_data' based on specified columns...")
        schedule_data_merge = self.data_merge_obj.merge_with_destination_weather(origin_weather_merge, destination_weather_data)

        #Step 11.5
        # Create a copy of the schedule_data_merge DataFrame
        schedule_data_copy = schedule_data_merge.copy()
        
        # * Round 'estimated_arrival' times to the nearest hour
        # * Round 'departure_date' times to the nearest hour
        #Step 12
        print("Rounding the scheduled dates to nearest hour...")
        nearest_hour_schedule_df = self.data_merge_obj.round_schedule_dates_to_nearest_hour(schedule_data_copy)
        
        #Step 13
        print("exploding the schedule...")
        hourly_exploded_scheduled_df = self.data_merge_obj.explode_schedule(nearest_hour_schedule_df,
            departure_col='departure_date',
            arrival_col='estimated_arrival'
        )
        
        #Step 13.5
        #Merge the traffic_table and hourly_exploded_scheduled_df DataFrames
        print("Merge the traffic_table and hourly_exploded_scheduled_df DataFrames...")
        traffic_table=dfs['traffic_table']
        scheduled_traffic = hourly_exploded_scheduled_df.merge(
            traffic_table,
            left_on=['route_id', 'custom_date'],  # Columns from hourly_exploded_scheduled_df
            right_on=['route_id', 'date'],        # Corresponding columns from traffic_table
            how='left'
        )
        
        #Step 14
        # Group by 'unique_id', 'truck_id', and 'route_id', and apply custom aggregation
        print("# Group by 'unique_id', 'truck_id', and 'route_id', and apply custom aggregation...")
        scheduled_route_traffic = self.data_merge_obj.aggregate_scheduled_traffic(
            df=scheduled_traffic,
            group_columns=['truck_id', 'route_id'],
            vehicle_col='no_of_vehicles',
            accident_col='accident'
        )
        
       
        #renaming the schedule_data_merge with origin_destination_weather
        origin_destination_weather=schedule_data_merge
        
         #Step 14.5
        #Merging the data frames
        #Merge schedule_route_traffic with origin_destination_weather
        origin_destination_weather_traffic_merge=origin_destination_weather.merge(scheduled_route_traffic,on=['truck_id','route_id'],how='left')

        #Step 15
        #merge weather & traffic
        print("merge weather & traffic...")
        # Define the columns to merge on
        merge_columns = ['truck_id', 'route_id', 'departure_date', 'estimated_arrival', 'delay']

        # Call the function to merge the data
        merged_data_weather_traffic = self.data_merge_obj.merge_schedule_weather_traffic(
            schedule_weather_df=schedule_weather_merge,
            weather_traffic_df=origin_destination_weather_traffic_merge,
            merge_columns=merge_columns
        ) 
        # # Convert 'departure_date' to datetime in the schedule_weather_merge DataFrame
        # schedule_weather_merge['departure_date'] = pd.to_datetime(schedule_weather_merge['departure_date'], errors='coerce')

        # # If origin_destination_weather_traffic_merge has 'departure_date' as object, convert it as well
        # origin_destination_weather_traffic_merge['departure_date'] = pd.to_datetime(origin_destination_weather_traffic_merge['departure_date'], errors='coerce')
        # #merged_data_weather_traffic=pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['truck_id', 'route_id', 'departure_date','estimated_arrival', 'delay'], how='left')

        # # Merge the DataFrames
        # print("Merging the dataframes...")
        # merged_data_weather_traffic = pd.merge(
        #     schedule_weather_merge, 
        #     origin_destination_weather_traffic_merge, 
        #     on=['truck_id', 'route_id', 'departure_date', 'estimated_arrival', 'delay'], 
        #     how='left'
        # )

        #merge weather traffic trucks
        trucks_table = dfs['trucks_table']
        merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_table, on='truck_id', how='left')

        #Merge merged_data with truck_data based on 'truck_id' column (Left Join)
        drivers_table = dfs['drivers_table']
        final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_table, left_on='truck_id', right_on = 'vehicle_no', how='left')
        
        #Function to check if there is nighttime involved between arrival and departure time
        print(" Adding function to see if there is mighttime between departure time and arrival time")
        def has_midnight(start, end):
            return int(start.date() != end.date())
        final_merge['is_midnight'] = final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)
        
        #Dropping the duplicates
        print("Dropping the duplicates and last cleaning")
        final_merge = final_merge.drop_duplicates()
        final_merge = final_merge.drop(columns=['city_id_x','city_id_y','custom_date_x','custom_date_y'])
        final_merge['unique_id'] = final_merge.index
        final_merge.dropna(inplace=True)
        final_merge['event_time']= pd.to_datetime('2024-10-8')
        print(final_merge.shape)

        # Call the function to process the DataFrame
        print(" processing the final merge dataframe")
        final_merge = self.data_merge_obj.process_final_merge(final_merge)
        
        
        # Now call the function as in the usage example
        print("Inserting the final_merge into the hopswork...")
        final_merge = self.data_merge_obj.insert_dataframe_to_feature_group(final_merge
            # dataframe=final_merge,  # Ensure you’re using the right parameter name here
            # feature_group_name="final_data_feature_group",
            # primary_key=["unique_id"],
            # description="Updated truck delay data with additional features",
            # event_time="event_time"
        )
            
    


        print("Data processing completed.")

if __name__ == '__main__':
    try:
        pipeline = DataMergingPipeline()
        pipeline.main()
        print(">>> Stage completed: Data Merging <<<")
    except Exception as e:
        print(e)
        raise e

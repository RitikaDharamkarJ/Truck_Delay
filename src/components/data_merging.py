import os
import sys
import pandas as pd
import numpy as np
import configparser
import hopsworks
from hsfs.client.exceptions import RestAPIError
# from src.utils.feature_group import hopswork_login, fetch_df_from_feature_groups

class DataMerge:
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
    
    
    #Step 1
    def get_features(self):
        """Fetch feature groups from Hopsworks."""
        feature_groups = [
            {"name": "city_weather", "version": 1},
            {"name": "drivers_table", "version": 1},
            {"name": "routes_table", "version": 1},
            {"name": "routes_weather", "version": 1},
            {"name": "traffic_table", "version": 1},
            {"name": "truck_schedule_table", "version": 1},
            {"name": "trucks_table", "version": 1}
        ]

        feature_group_data = {}
        for fg in feature_groups:
            try:
                feature_group = self.fs.get_feature_group(fg['name'], version=fg['version'])
                df = feature_group.read()
                feature_group_data[fg['name']] = df
                print(f"Downloaded feature group: {fg['name']}")
            except RestAPIError as e:
                print(f"Error downloading feature group: {fg['name']}")
                print(e)
        return feature_group_data

    #Step 2
    def assign_feature_group_data(self, feature_group_data):
        """Assign feature groups to DataFrames and store them in a dictionary."""
        dfs = {}
        
        # Assign each DataFrame to the 'dfs' dictionary with a new name
        dfs['city_weather'] = feature_group_data['city_weather']
        dfs['drivers_table'] = feature_group_data['drivers_table']
        dfs['routes_table'] = feature_group_data['routes_table']
        dfs['routes_weather'] = feature_group_data['routes_weather']
        dfs['traffic_table'] = feature_group_data['traffic_table']
        dfs['truck_schedule_table'] = feature_group_data['truck_schedule_table']
        dfs['trucks_table'] = feature_group_data['trucks_table']
        
        return dfs

    #Step 3
    def drop_columns_from_dfs(self, dfs, columns_to_remove):
        """Drop specified columns from all DataFrames."""
        for name, df in dfs.items():
            dfs[name] = df.drop(columns=columns_to_remove, errors='ignore')
        return dfs

    #Step 4
    def drop_duplicates_from_dfs(self, dfs, columns_to_check):
        """Drop duplicate rows from DataFrames based on specified columns."""
        for df_name, subset in columns_to_check.items():
            dfs[df_name] = dfs[df_name].drop_duplicates(subset=subset)
        return dfs
    
    #Step 5
    def drop_weather_columns_from_dfs(self, dfs, weather_columns_to_drop):
        """Drop weather-related columns from specific DataFrames."""
        for table_name, columns_to_drop in weather_columns_to_drop.items():
            if table_name in dfs:
                dfs[table_name] = dfs[table_name].drop(columns=columns_to_drop, errors='ignore')
        return dfs

    #Step 6
    def rename(self, dfs, table_names):
        """Rename 'date' column to 'custom_date' in specified DataFrames."""
        for table_name in table_names:
            dfs[table_name] = dfs[table_name].rename(columns={'date': 'custom_date'})
        return dfs
     
    #Step 7
    def process_dates(self, df, estimated_arrival_column, departure_date_column):
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


        # * Define a custom function to calculate mode
        #              def custom_mode(x):
        #                  return x.mode().iloc[0]
    #Step 8
    def group_and_aggregate_weather_data(self, df, group_by_columns):
            """
            Group by specified columns and perform aggregation on weather data.

            Parameters:
            df (pd.DataFrame): The DataFrame containing the weather data.
            group_by_columns (list): Columns to group by, e.g., ['truck_id', 'route_id'].

            Returns:
            pd.DataFrame: A DataFrame with aggregated weather data.
            """
            # Define a custom function to calculate mode with error handling
            def custom_mode(x):
                mode_values = x.mode()
                if len(mode_values) > 0:
                    return mode_values.iloc[0]
                else:
                    return np.nan  # or a default value, such as '' or np.nan

            # Group by specified columns and aggregate
            aggregated_df= df.groupby(group_by_columns, as_index=False).agg(
                route_avg_temp=('temp', 'mean'),
                route_avg_wind_speed=('wind_speed', 'mean'),
                route_avg_precip=('precip', 'mean'),
                route_avg_humidity=('humidity', 'mean'),
                route_avg_visibility=('visibility', 'mean'),
                route_avg_pressure=('pressure', 'mean'),
                route_description=('description', custom_mode)
            )
            
            return aggregated_df
     
     
    #Step 9
    def round_schedule_times_and_merge(self, truck_schedule_df, routes_df):
        """
        This function takes two DataFrames, rounds the 'estimated_arrival' and 'departure_date' columns to the nearest hour
        in the truck schedule DataFrame, and merges it with the routes DataFrame on 'route_id'.
        
        Parameters:
        truck_schedule_df (pd.DataFrame): DataFrame containing the truck schedule information.
        routes_df (pd.DataFrame): DataFrame containing the routes information.
        
        Returns:
        pd.DataFrame: A merged DataFrame with rounded 'estimated_arrival' and 'departure_date' times.
        """
        # Create a copy of the truck schedule DataFrame
        nearest_hour_schedule_df = truck_schedule_df.copy()

        # Convert 'estimated_arrival' and 'departure_date' columns to datetime (if they aren't already)
        nearest_hour_schedule_df['estimated_arrival'] = pd.to_datetime(nearest_hour_schedule_df['estimated_arrival'], errors='coerce')
        nearest_hour_schedule_df['departure_date'] = pd.to_datetime(nearest_hour_schedule_df['departure_date'], errors='coerce')

        # Round 'estimated_arrival' and 'departure_date' to the nearest hour
        nearest_hour_schedule_df['estimated_arrival_nearest_hour'] = nearest_hour_schedule_df['estimated_arrival'].dt.round("H")
        nearest_hour_schedule_df['departure_date_nearest_hour'] = nearest_hour_schedule_df['departure_date'].dt.round("H")

        # Merge the result with routes DataFrame on 'route_id'
        merged_df = pd.merge(nearest_hour_schedule_df, routes_df, on='route_id', how='left')

        return merged_df
    
    
    
    #Step 10
    def merge_schedule_with_weather(self,nearest_hour_schedule_route_df, origin_weather_data):
        """
        This function merges the truck schedule and routes DataFrame with weather data based on specific columns.
        It ensures the datetime columns are in UTC format before merging.

        Parameters:
        nearest_hour_schedule_route_df (pd.DataFrame): Merged DataFrame of truck schedule and routes with nearest hour times.
        origin_weather_data (pd.DataFrame): DataFrame containing weather data with 'custom_date' and 'city_id'.

        Returns:
        pd.DataFrame: Merged DataFrame containing truck schedule, routes, and corresponding weather information.
        """
        
        # Ensure 'departure_date_nearest_hour' is in UTC
        if nearest_hour_schedule_route_df['departure_date_nearest_hour'].dt.tz is None:
            nearest_hour_schedule_route_df['departure_date_nearest_hour'] = nearest_hour_schedule_route_df['departure_date_nearest_hour'].dt.tz_localize('UTC')

        # Ensure 'custom_date' in origin_weather_data is in UTC
        if origin_weather_data['custom_date'].dt.tz is None:
            origin_weather_data['custom_date'] = pd.to_datetime(origin_weather_data['custom_date'], errors='coerce').dt.tz_localize('UTC')

        # Specify the columns to merge on
        left_merge_columns = ['origin_id', 'departure_date_nearest_hour']
        right_merge_columns = ['city_id', 'custom_date']

        # Perform the merge
        merged_df = pd.merge(
            nearest_hour_schedule_route_df,
            origin_weather_data,
            left_on=left_merge_columns,
            right_on=right_merge_columns,
            how='left'
        )
        
        return merged_df
    
    
    #Step 11
    def merge_with_destination_weather(self,origin_weather_merge, destination_weather_data):
        """
        This function merges the schedule data (already merged with origin weather) with destination weather data 
        based on specific columns, ensuring that datetime columns are in UTC format.

        Parameters:
        origin_weather_merge (pd.DataFrame): Merged DataFrame of truck schedule, routes, and origin weather data.
        destination_weather_data (pd.DataFrame): DataFrame containing destination weather data with 'custom_date' and 'city_id'.

        Returns:
        pd.DataFrame: Merged DataFrame containing schedule, routes, origin weather, and destination weather information.
        """
        
        # Ensure 'estimated_arrival_nearest_hour' is in UTC
        if origin_weather_merge['estimated_arrival_nearest_hour'].dt.tz is None:
            origin_weather_merge['estimated_arrival_nearest_hour'] = origin_weather_merge['estimated_arrival_nearest_hour'].dt.tz_localize('UTC')

        # Ensure 'custom_date' in destination_weather_data is in UTC
        if destination_weather_data['custom_date'].dt.tz is None:
            destination_weather_data['custom_date'] = pd.to_datetime(destination_weather_data['custom_date'], errors='coerce').dt.tz_localize('UTC')

        # Specify the columns to merge on
        left_merge_columns = ['destination_id', 'estimated_arrival_nearest_hour']
        right_merge_columns = ['city_id', 'custom_date']

        # Perform the merge
        merged_df = pd.merge(
            origin_weather_merge,
            destination_weather_data,
            left_on=left_merge_columns,
            right_on=right_merge_columns,
            how='left'
        )
        
        return merged_df
    
    
    #Step 12
    def round_schedule_dates_to_nearest_hour(self, schedule_data):
        """
        This function rounds the 'estimated_arrival' and 'departure_date' columns in a DataFrame to the nearest hour.

        Parameters:
        schedule_data (pd.DataFrame): The DataFrame containing the 'estimated_arrival' and 'departure_date' columns.

        Returns:
        pd.DataFrame: A copy of the input DataFrame with 'estimated_arrival' and 'departure_date' rounded to the nearest hour.
        """
        
        # Create a copy of the DataFrame to avoid modifying the original data
        schedule_data_copy = schedule_data.copy()

        # Round 'estimated_arrival' to the nearest hour
        schedule_data_copy['estimated_arrival'] = schedule_data_copy['estimated_arrival'].dt.round('H')

        # Round 'departure_date' to the nearest hour
        schedule_data_copy['departure_date'] = schedule_data_copy['departure_date'].dt.round('H')

        return schedule_data_copy
    
    #Step 13
    def explode_schedule(self, schedule_df, departure_col, arrival_col):
        """
        Ensures the specified datetime columns are in UTC and creates custom date ranges between them.

        Parameters:
        schedule_df (pd.DataFrame): The DataFrame containing the schedule data.
        departure_col (str): The column name for departure date.
        arrival_col (str): The column name for estimated arrival date.

        Returns:
        pd.DataFrame: A DataFrame with custom date ranges created between departure and arrival columns.
        """

        # Function to ensure datetime columns are in UTC
        def ensure_utc(column):
            if column.dt.tz is None:  # Check if the column is naive
                return column.dt.tz_localize('UTC')  # Localize to UTC
            else:
                return column.dt.tz_convert('UTC')  # Convert to UTC if already aware

        # Ensure both datetime columns are in UTC
        schedule_df[departure_col] = ensure_utc(schedule_df[departure_col])
        schedule_df[arrival_col] = ensure_utc(schedule_df[arrival_col])

        # Create the custom date ranges and explode the DataFrame
        exploded_schedule_df = (
            schedule_df.assign(
                custom_date=[
                    pd.date_range(start, end, freq='H')  # Create custom date ranges
                    for start, end in zip(
                        schedule_df[departure_col], schedule_df[arrival_col]
                    )  # Using departure and estimated arrival times
                ]
            ).explode('custom_date', ignore_index=True)  # Explode the DataFrame based on the custom date range
        )

        return exploded_schedule_df

    #Step 14
    def aggregate_scheduled_traffic(self, df, group_columns, vehicle_col, accident_col):
        """
        Aggregates traffic data by calculating the average number of vehicles and detecting accidents.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the traffic data.
        group_columns (list): List of columns to group by.
        vehicle_col (str): The column name for the number of vehicles.
        accident_col (str): The column name for accident data.

        Returns:
        pd.DataFrame: A DataFrame with aggregated traffic data, including average number of vehicles and accident detection.
        """
        
        # Define a custom aggregation function for accidents
        def custom_agg(values):
            return 1 if 1 in values else 0

        # Group by specified columns and apply custom aggregation
        aggregated_df = df.groupby(group_columns, as_index=False).agg(
            avg_no_of_vehicles=(vehicle_col, 'mean'),
            accident=(accident_col, custom_agg)
        )

        return aggregated_df


#Step 15

    def merge_schedule_weather_traffic(self, schedule_weather_df, weather_traffic_df, merge_columns):
        """
        Merges schedule weather data with weather-traffic data on specified columns.

        Parameters:
        schedule_weather_df (pd.DataFrame): The schedule weather DataFrame (schedule_weather_merge).
        weather_traffic_df (pd.DataFrame): The weather-traffic merged DataFrame (origin_destination_weather_traffic_merge).
        merge_columns (list): List of columns to merge on.

        Returns:
        pd.DataFrame: Merged DataFrame containing both schedule weather and weather-traffic data.
        """
        # Convert 'departure_date' to datetime in both DataFrames
        schedule_weather_df['departure_date'] = pd.to_datetime(schedule_weather_df['departure_date'], errors='coerce')
        weather_traffic_df['departure_date'] = pd.to_datetime(weather_traffic_df['departure_date'], errors='coerce')

        # Merge the DataFrames
        print("Merging the dataframes...")
        merged_df = pd.merge(
            schedule_weather_df, 
            weather_traffic_df, 
            on=merge_columns, 
            how='left'
        )
        
        return merged_df


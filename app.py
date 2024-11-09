
import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import logging
import os
import joblib
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import configparser

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up your Streamlit page
st.set_page_config(page_title="Truck Delay Prediction", layout="wide")

# Load configuration for Hopsworks and MLflow
CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'

# Function to connect to Hopsworks and fetch the final merged dataset
def fetch_final_merge_dataset(api_key, feature_group_name, version=1):
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    feature_group = fs.get_feature_group(feature_group_name, version=version)
    df = feature_group.read()
    logging.info("Configuration read successfully.")
    return df

# Function to load the model, encoder, and scaler from MLflow
def load_mlflow_assets(experiment_name, model_name):
    # Set MLflow tracking URI
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    
    # Get the latest run for the model
    client = mlflow.tracking.MlflowClient() 
    print("MLflow client initialized")  # Debug
 
    # Load the experiment
    experiment =mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        st.error(f"Experiment '{experiment_name}' not found.")
        return None, None, None
    else:
        print(f"Experiment '{experiment_name}' found with ID: {experiment.experiment_id}")  # Debug
        

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"])
    print(f"Found {len(runs)} runs for experiment '{experiment_name}'")  # Debug

    # Get the latest run ID
    latest_run = runs[0]
    run_id = latest_run.info.run_id 
    
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        print(f"Loaded model: {model_name}")  # Debug
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    # Load scaler and encoder as artifacts if they are saved in the model directory
    scaler = None
    encoder = None
    
    artifact_location = client.get_run(run_id).info.artifact_uri.replace("file:///", "")
    print(artifact_location)
    try:
        scaler_path =  os.path.join(artifact_location, 'scaler.pkl')
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            print("Scaler loaded")  # Debug
    except Exception as e:
        print(f"Scaler not found or error loading: {e}")

    try:
        encoder_path = os.path.join(artifact_location, 'onehot_encoder.pkl')
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
            print("Encoder loaded")  # Debug
    except Exception as e:
        print(f"Encoder not found or error loading: {e}")

    return model, scaler, encoder


# Function to preprocess data (encoding + scaling)
def preprocess_data(data, encoder, scaler, cts_cols, cat_cols):
    # Ensure user_input_df has the required columns
    data = data[cts_cols + cat_cols]
 
    # Apply One-Hot Encoding only to the categorical columns
    encoded_df = pd.DataFrame(encoder.transform(data[cat_cols]), 
                              columns=encoder.get_feature_names_out(), 
                              index=data.index)
#check here 
    # Concatenate encoded columns with continuous columns
    full_df = pd.concat([data[cts_cols], encoded_df], axis=1)

    # Apply Scaling only to the continuous + encoded columns
    scaled_df = pd.DataFrame(scaler.transform(full_df), columns=full_df.columns, index=full_df.index)

    return scaled_df
# Function to make a prediction
def make_prediction(model, user_input_df):
    # Only return predictions
    prediction = model.predict(user_input_df)
    return prediction

# Streamlit application
def main():
    # Load config for Hopsworks API key
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)
    api_key = config['HOPSWORK']['hopsworks_api_key']

    # Streamlit UI setup
    st.title('Truck Delay Classification')
    
    # Fetch the final merged dataset
    final_merge = fetch_final_merge_dataset(api_key, "final_merge_df")
    final_merge = final_merge.drop(columns=['unique_id', 'event_time'], errors='ignore')

    model, scaler, encoder = load_mlflow_assets("ML Models with Hyperparameter Tuning", "XGBoost_Model")

    # Define the continuous and categorical columns
    cts_cols = ['route_avg_temp', 'route_avg_wind_speed',
                    'route_avg_precip', 'route_avg_humidity', 'route_avg_visibility',
                    'route_avg_pressure', 'distance', 'average_hours',
                    'temp_origin', 'wind_speed_origin', 'precip_origin', 'humidity_origin',
                    'visibility_origin', 'pressure_origin',
                    'temp_destination','wind_speed_destination','precip_destination',
                    'humidity_destination', 'visibility_destination','pressure_destination',
                    'avg_no_of_vehicles', 'truck_age','load_capacity_pounds', 'mileage_mpg',
                    'age', 'experience','average_speed_mph']  # Define your continuous columns here
    cat_cols = ['route_description',
                    'origin_description', 'destination_description',
                    'accident', 'fuel_type',
                    'gender', 'driving_style', 'ratings','is_midnight']  # Define your categorical columns here
    encode_columns = ['route_description', 'origin_description', 'destination_description', 'fuel_type', 'gender', 'driving_style']

    # Sidebar UI for filtering
    st.sidebar.header("Filter Options")
    options = ['date_filter', 'truck_id_filter', 'route_id_filter']
    
    # Use radio button to allow the user to select only one option for filtering
    selected_option = st.sidebar.radio("Choose a filtering option:", options)
    
    filter_query = None
    sentence = ""
    flag = False
    
    # Date filter
    if selected_option == 'date_filter':
        st.write("### Date Ranges")
        # Date range inputs
        min_date = min(final_merge['departure_date'])
        max_date = max(final_merge['departure_date'])
        from_date = st.date_input("Enter start date:", value=min_date)
        to_date = st.date_input("Enter end date:", value=max_date)
        flag = True
        sentence = "during the chosen date range"
        filter_query = (final_merge['departure_date'] >= str(from_date)) & (final_merge['departure_date'] <= str(to_date))
        
    # Truck ID filter
    elif selected_option == 'truck_id_filter':
        st.write("### Truck ID")
        truck_id = st.selectbox('Select truck ID:', final_merge['truck_id'].unique())
        flag = True
        sentence = "for the specified truck ID"
        filter_query = (final_merge['truck_id'] == truck_id)
    
    # Route ID filter
    elif selected_option == 'route_id_filter':
        st.write("### Route ID")
        route_id = st.selectbox('Select route ID:', final_merge['route_id'].unique())
        flag = True
        sentence = "for the specified route ID"
        filter_query = (final_merge['route_id'] == str(route_id))

    # Load model, scaler, and encoder
    model, scaler, encoder = load_mlflow_assets("ML Models with Hyperparameter Tuning", "XGBoost_Model")

    if st.button('Predict'):
        # Filter data
        filtered_data = final_merge[filter_query]

        if filtered_data.empty:
            st.error("No data found for the selected filter.")
        else:
            st.write(f"Performing prediction {sentence}:")
            # No need to write filtered data again here; this will avoid displaying a second table
            # Reset index for consistency
            filtered_data.reset_index(drop=True, inplace=True)

            # Encoding categorical columns
            encoded_data = encoder.transform(filtered_data[encode_columns])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(encode_columns), index=filtered_data.index)

            # Extract remaining categorical columns that do not need encoding
            other_cat_data = filtered_data[[col for col in cat_cols if col not in encode_columns]]

            # Combine encoded data with remaining categorical data
            full_cat_data = pd.concat([other_cat_data, encoded_df], axis=1)

            # Scaling continuous columns
            scaled_data = scaler.transform(filtered_data[cts_cols])
            scaled_df = pd.DataFrame(scaled_data, columns=cts_cols, index=filtered_data.index)

            # Combine continuous and categorical data for model prediction
            final_data = pd.concat([scaled_df, full_cat_data], axis=1)

            # Make predictions
            predictions = make_prediction(model, final_data)
            results_df = filtered_data.copy()  # Copy all columns from the filtered data
            results_df['delay'] = predictions 
            # filtered_data['delay'] = predictions

            # st.subheader("Prediction Results")
            st.dataframe(results_df)  # Display the results without styling
            # st.write(filtered_data[['truck_id', 'route_id', 'departure_date', 'Prediction']])

if __name__ == "__main__":
    main()

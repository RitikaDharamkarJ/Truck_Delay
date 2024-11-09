import os
import sys
import pandas as pd
import numpy as np
import traceback
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import configparser
from sklearn.metrics import accuracy_score, classification_report
import logging
import joblib
import pickle
import hopsworks
import time
from hsfs.client.exceptions import RestAPIError
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score 

# from src.utils.feature_group import hopswork_login, fetch_df_from_feature_groups

class DataTransformation:
    def __init__(self, project):
        self.config = self.load_config()
        # self.project = hopsworks.login(api_key_value=self.config['HOPSWORK']['hopsworks_api_key'])
        # self.fs = self.project.get_feature_store()
        # self.feature_group_data = self.get_features()
        self.project = project
        self.fs = self.project.get_feature_store()
        self.api_key = self.config['HOPSWORK']['hopsworks_api_key']
        
        
    def load_config(self):
        """Load configuration from the config file."""
        config = configparser.RawConfigParser()
        CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
        config.read(CONFIG_FILE_PATH)
        return config 
    
    # def hopswork_login(config):
    #     """Logs into Hopsworks using the API key from the config."""
    #     api_key = config['HOPSWORK']['hopsworks_api_key']
    #     return hopsworks.login(api_key_value=api_key)
    
    
    def connect_and_download_feature_group(self, api_key, feature_group_name, version):
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
        
        
        
    def get_column_names(self):
        """Return column names."""
        cts_cols=['route_avg_temp', 'route_avg_wind_speed',
              'route_avg_precip', 'route_avg_humidity', 'route_avg_visibility',
              'route_avg_pressure', 'distance', 'average_hours',
              'temp_origin', 'wind_speed_origin', 'precip_origin', 'humidity_origin',
              'visibility_origin', 'pressure_origin',
              'temp_destination','wind_speed_destination','precip_destination',
              'humidity_destination', 'visibility_destination','pressure_destination',
               'avg_no_of_vehicles', 'truck_age','load_capacity_pounds', 'mileage_mpg',
               'age', 'experience','average_speed_mph']

        cat_cols=['route_description',
                        'origin_description', 'destination_description',
                        'accident', 'fuel_type',
                        'gender', 'driving_style', 'ratings','is_midnight']

        target=['delay']  
        
        #Return the column names
        return cts_cols, cat_cols, target
    


    def split_data_based_on_estimated_arrival(self, final_merge):
        # Step 1: Ensure 'estimated_arrival' is timezone-aware (UTC)
        final_merge['estimated_arrival'] = final_merge['estimated_arrival'].dt.tz_convert('UTC')
        
        # Create a UTC-aware comparison timestamp for training
        comparison_date_train = pd.to_datetime('2019-01-30', utc=True)
        
        # Step 2: Filter the DataFrame to create train set
        train_df = final_merge[final_merge['estimated_arrival'] <= comparison_date_train]
        
        # Create UTC-aware comparison timestamps for validation
        start_date_validation = pd.to_datetime('2019-01-30', utc=True)
        end_date_validation = pd.to_datetime('2019-02-07', utc=True)
        
        # Step 3: Filter the DataFrame based on the date range for validation set
        validation_df = final_merge[
            (final_merge['estimated_arrival'] > start_date_validation) & 
            (final_merge['estimated_arrival'] <= end_date_validation)
        ]
        
        # Create a UTC-aware timestamp for test set
        comparison_date_test = pd.to_datetime('2019-02-07', utc=True)
        
        # Step 4: Filter the DataFrame to create test set
        test_df = final_merge[final_merge['estimated_arrival'] > comparison_date_test]
        
        # Resetting the index for alignment
        train_df.reset_index(drop=True, inplace=True)
        validation_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        
        # Return the split DataFrames
        return train_df, validation_df, test_df

    def select_available_columns_train(self, train_df,validation_df,test_df, cts_cols, cat_cols, target_column='delay'):
        
        # Filter columns to include only those present in each DataFrame
        cts_cols = [col for col in cts_cols if col in train_df.columns]
        cat_cols = [col for col in cat_cols if col in train_df.columns]
        
        
        # Select the available columns from the DataFrame
        X_train = train_df[cts_cols + cat_cols]
        y_train=train_df['delay']
        X_valid = validation_df[cts_cols + cat_cols]
        y_valid = validation_df['delay']
        X_test = test_df[cts_cols + cat_cols]
        y_test = test_df['delay']
        
        return {
                'X_train': X_train, 'y_train': y_train,
                'X_valid': X_valid, 'y_valid': y_valid,
                'X_test': X_test, 'y_test': y_test
            }
    
    
    
    def apply_one_hot_encoding(self, X_train, X_valid, X_test, encode_columns):
        
        encode_columns = ['route_description', 'origin_description', 'destination_description', 'fuel_type', 'gender', 'driving_style']
       
        # Step 2: Initialize the OneHotEncoder with updated parameter
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Step 3: Fit the encoder on the training data
        encoder.fit(X_train[encode_columns])
        
        # Generate names for the new one-hot encoded features
        encoded_features = list(encoder.get_feature_names_out(encode_columns))
                
        # Transform the training, validation, and test datasets
        X_train_encoded = pd.DataFrame(encoder.transform(X_train[encode_columns]), columns=encoded_features, index=X_train.index)
        X_valid_encoded = pd.DataFrame(encoder.transform(X_valid[encode_columns]), columns=encoded_features, index=X_valid.index)
        X_test_encoded = pd.DataFrame(encoder.transform(X_test[encode_columns]), columns=encoded_features, index=X_test.index)
        
        
        X_train = pd.concat([X_train.drop(encode_columns, axis=1),
                             pd.DataFrame(X_train_encoded, columns=encoded_features, index=X_train.index)], axis=1)
        X_valid = pd.concat([X_valid.drop(encode_columns, axis=1),
                             pd.DataFrame(X_valid_encoded, columns=encoded_features, index=X_valid.index)], axis=1)
        X_test = pd.concat([X_test.drop(encode_columns, axis=1),
                           pd.DataFrame(X_test_encoded, columns=encoded_features, index=X_test.index)], axis=1)

        
        # # Drop the original categorical columns from the datasets
        # X_train = X_train.drop(encode_columns, axis=1)
        # X_valid = X_valid.drop(encode_columns, axis=1)
        # X_test = X_test.drop(encode_columns, axis=1)
        
        # # Concatenate the encoded columns back to the original dataframes
        # X_train = pd.concat([X_train, X_train_encoded], axis=1)
        # X_valid = pd.concat([X_valid, X_valid_encoded], axis=1)
        # X_test = pd.concat([X_test, X_test_encoded], axis=1)
        print('The final X_train columns post encoding are:',X_train.columns)
    
        # Save encoder as a pickle file
        encoder_file = 'src/models/onehot_encoder.pkl'
        with open(encoder_file, 'wb') as f:
            pickle.dump(encoder, f)
        print(f"Encoder saved as {encoder_file}")

        # Log the encoder pickle file to MLflow
        mlflow.log_artifact(encoder_file)
        print(f"Encoder pickle file logged as an artifact in MLflow.")
        
        return X_train, X_valid, X_test
    
    def apply_scaling(self, X_train, X_valid, X_test, continous_cols):
        """
        Function to apply scaling using StandardScaler to specified columns in the training, validation, and test datasets.

        Parameters:
        X_train (pd.DataFrame): The training dataset.
        X_valid (pd.DataFrame): The validation dataset.
        X_test (pd.DataFrame): The test dataset.
        columns_to_scale (list): List of column names to be scaled. If None, all columns will be scaled.

        Returns:
        X_train_scaled (pd.DataFrame): The scaled training dataset.
        X_valid_scaled (pd.DataFrame): The scaled validation dataset.
        X_test_scaled (pd.DataFrame): The scaled test dataset.
        """
        # Step 1: Initialize the StandardScaler
        scaler = StandardScaler()
        
        # # If no specific columns are provided, scale all columns
        # if columns_to_scale is None:
        #     columns_to_scale = cts_cols
        
        # Fit the scaler on the training data
        logging.debug(f"Fitting scaler on training data columns: {continous_cols}")
        scaler.fit(X_train[continous_cols])

        # Transform the validation and test datasets
        logging.debug("Transforming the validation dataset.")
        X_valid[continous_cols] = scaler.transform(X_valid[continous_cols])

        logging.debug("Transforming the test dataset.")
        X_test[continous_cols] = scaler.transform(X_test[continous_cols])

        
        # # Step 3: Fit the scaler on X_train and transform X_train, X_valid, and X_test
        # X_train[continous_cols] = pd.DataFrame(scaler.fit_transform(X_train[continous_cols]), 
        #                             columns=continous_cols, index=X_train.index)

        # X_valid[continous_cols] = pd.DataFrame(scaler.transform(X_valid[continous_cols]), 
        #                             columns=continous_cols, index=X_valid.index)

        # X_test[continous_cols] = pd.DataFrame(scaler.transform(X_test[continous_cols]), 
        #                             columns=continous_cols, index=X_test.index)
        
        # # Optional: Print the shape to confirm scaling  
        # print(f"X_train shape after scaling: {X_train_scaled.shape}")
        # print(f"X_valid shape after scaling: {X_valid_scaled.shape}")
        # print(f"X_test shape after scaling: {X_test_scaled.shape}")
        
            # Save scaler as a pickle file
        scaler_file = 'src/models/scaler.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved as {scaler_file}")

        # Log the scaler pickle file to MLflow
        mlflow.log_artifact(scaler_file)
        print(f"Scaler pickle file logged as an artifact in MLflow.")
        
        return X_train, X_valid, X_test
    
    
    #training dataset
    def train_and_evaluate_models(self, X_train, y_train, logreg_params, rf_params, xgb_params): #X_train is used in place of X_train_scaled
        """
        Function to train and evaluate multiple models using GridSearchCV and log them with MLflow.

        Parameters:
        X_train_scaled (pd.DataFrame): The scaled training dataset.
        y_train (pd.Series): The target variable for training.
        logreg_params (dict): Hyperparameter grid for Logistic Regression.
        rf_params (dict): Hyperparameter grid for Random Forest.
        xgb_params (dict): Hyperparameter grid for XGBoost.
        """
    # Define X_train, X_valid, y_train, y_valid (use your existing datasets)
        X = X_train  # Assuming scaled features #X_train is used in place of X_train_scaled
        y = y_train
    
        # # Split the data further for GridSearchCV
        # X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        #     X_train, y_train, test_size=0.2, random_state=42 ##X_train is used in place of X_train_scaled
        # )
        
        # Initialize MLflow experiment
        mlflow.set_experiment("ML Models with Hyperparameter Tuning")
        
        # Dictionary to store models and their validation accuracy
        models = {}

        # Define a nested function to handle individual model training and logging
        def train_and_evaluate_model(model, param_grid, model_name):
            
            # End the previous run if it's still active
            if mlflow.active_run():
                mlflow.end_run()
            # Start a new run
            with mlflow.start_run(run_name=model_name):
                # GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)

                # Get the best model from GridSearch
                best_model = grid_search.best_estimator_

                # Predict on training data
                y_pred = best_model.predict(X_train)

                # Evaluate performance
                acc = accuracy_score(y_train, y_pred)
                f1 = f1_score(y_train, y_pred, average='weighted')

                # Log parameters, metrics, and model
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
                mlflow.sklearn.log_model(best_model, model_name)

                # Print the results
                print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                print(f"Accuracy: {acc}, F1 Score: {f1}")
                print(classification_report(y_train, y_pred))
                
               # Return the model, accuracy, and metrics for comparison
            return best_model, acc, {"accuracy": acc, "f1_score": f1}
        
        # Train and evaluate the models, storing their validation accuracies
        models["Logistic_Regression"] = train_and_evaluate_model(LogisticRegression(), logreg_params, "Logistic Regression")
        models["Random_Forest"] = train_and_evaluate_model(RandomForestClassifier(), rf_params, "Random Forest")
        models["XGBoost"] = train_and_evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_params, "XGBoost")

    # Find the model with the highest validation accuracy
        best_model_name = max(models, key=lambda model: models[model][1])
        best_model, best_acc, best_metrics = models[best_model_name]

        # Register the best model in MLflow Model Registry
        with mlflow.start_run(run_name=f"Best_Model_Registration_{best_model_name}"):
            # Log the best model to MLflow
            mlflow.sklearn.log_model(best_model, best_model_name)

            # Register the best model
            best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}"
            model_version = mlflow.register_model(best_model_uri, f"{best_model_name}_Model")

            print(f"Best model '{best_model_name}' with validation accuracy {best_acc:.4f} registered in MLflow as version {model_version}.")

            # Save the best model locally as a pickle file
            pickle_file = f"src/models/{best_model_name}_model.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Model saved locally as {pickle_file}")

            # Log the pickle file as an artifact in the current MLflow run
            mlflow.log_artifact(pickle_file)
            print(f"Pickle file '{pickle_file}' logged as an artifact in MLflow.")

        # Return the best model and its metrics
        return best_model, best_metrics


    #Validation Data
    def train_and_log_models(self, X_train, y_train, X_valid, y_valid, logreg_params, rf_params, xgb_params): #X_train is used in place of X_train_scaled
        """
        Function to train models using GridSearchCV, log them with MLflow, and save the best model locally and in MLflow.
        
        Parameters:
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training labels.
        logreg_params (dict): Hyperparameter grid for Logistic Regression.
        rf_params (dict): Hyperparameter grid for Random Forest.
        xgb_params (dict): Hyperparameter grid for XGBoost.

        Returns:
        best_model_info (dict): Dictionary containing the best model and its metrics.
        """

        # Split the data for validation
        # X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42) #X_train is used in place of X_train_scaled
        print(X_train.columns)
        print(y_train.name)
        
        # Initialize MLflow experiment
        mlflow.set_experiment("ML Models with Hyperparameter Tuning")

        # Function to train, evaluate, and log models
        def train_and_log_model(model, param_grid, model_name):
            with mlflow.start_run(run_name=model_name):
                # GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)

                # Get the best model
                best_model = grid_search.best_estimator_

                # Evaluate on the validation set
                y_val_pred = best_model.predict(X_valid)
                val_acc = accuracy_score(y_valid, y_val_pred)
                val_f1 = f1_score(y_valid, y_val_pred, average='weighted')

                # Log parameters, metrics, and model with MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({"val_accuracy": val_acc, "val_f1_score": val_f1})
                mlflow.sklearn.log_model(best_model, model_name)

                # Save model as a pickle file locally
                pickle_file = f"src/models/{model_name}_model.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"Model saved as {pickle_file}")
                
                 # Save the scaler and encoder files to MLflow artifacts
                scaler_file = 'src/models/scaler.pkl'
                encoder_file = 'src/models/onehot_encoder.pkl'
                if os.path.exists(scaler_file):
                        mlflow.log_artifact(scaler_file)
                        print(f"Scaler file '{scaler_file}' logged as an artifact in MLflow.")
                if os.path.exists(encoder_file):
                        mlflow.log_artifact(encoder_file)
                        print(f"Encoder file '{encoder_file}' logged as an artifact in MLflow.")

                # Log the pickle file as an artifact in the current MLflow run
                mlflow.log_artifact(pickle_file)
                print(f"Pickle file '{pickle_file}' logged as an artifact in MLflow.")

                # Print evaluation results
                print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
                print(classification_report(y_valid, y_val_pred))

                return best_model, val_acc, {"accuracy": val_acc, "f1_score": val_f1}

        # Train models
        best_logistic_model, logistic_acc, logistic_metrics = train_and_log_model(LogisticRegression(), logreg_params, "Logistic_Regression")
        best_rf_model, rf_acc, rf_metrics = train_and_log_model(RandomForestClassifier(), rf_params, "Random_Forest")
        best_xgb_model, xgb_acc, xgb_metrics = train_and_log_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_params, "XGBoost")

        # # Compare validation accuracies and determine the best model
        # models = {
        #     "Logistic_Regression": (best_logistic_model, logistic_acc, logistic_metrics),
        #     "Random_Forest": (best_rf_model, rf_acc, rf_metrics),
        #     "XGBoost": (best_xgb_model, xgb_acc, xgb_metrics)
        # }
        
        # best_model_name = max(models, key=lambda model: models[model][1])
        # best_model, best_acc, best_metrics = models[best_model_name]

        # # Register the best model in MLflow Model Registry
        # with mlflow.start_run(run_name=f"Best_Model_Registration_{best_model_name}"):
        #     # Log the best model to MLflow
        #     mlflow.sklearn.log_model(best_model, best_model_name)

        #     # Register the best model in the registry
        #     best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}"
        #     model_version = mlflow.register_model(best_model_uri, f"{best_model_name}_Model")

        #     print(f"Best model '{best_model_name}' with validation accuracy {best_acc:.4f} registered in MLflow as version {model_version}.")

        # Return the best model and its metrics
        
        return {
            "Logistic_Regression": (best_logistic_model, logistic_metrics),
            "Random_Forest": (best_rf_model, rf_metrics),
            "XGBoost": (best_xgb_model, xgb_metrics)
        }
        # return {
        #     "Best_Model_Name": best_model_name,
        #     "Best_Model": best_model,
        #     "Best_Model_Metrics": best_metrics
        # }


    #Testing Data

    def evaluate_and_log_models(self, X_test, y_test, logreg_params, rf_params, xgb_params, X_train, y_train):
        """
        Function to evaluate models on test data, log them with MLflow, and save the best models to MLflow Model Registry and locally.

        Parameters:
        X_test_split (pd.DataFrame): Scaled test features.
        y_test_split (pd.Series): Test labels.
        logreg_params (dict): Hyperparameter grid for Logistic Regression.
        rf_params (dict): Hyperparameter grid for Random Forest.
        xgb_params (dict): Hyperparameter grid for XGBoost.
        X_train_split (pd.DataFrame): Scaled training features (needed for GridSearchCV to get best parameters).
        y_train_split (pd.Series): Training labels.

        Returns:
        best_model_info (dict): Dictionary containing the best model and its metrics.
        """

        # Initialize MLflow experiment
        mlflow.set_experiment("ML Models with Hyperparameter Tuning")

        # Dictionary to store models and their test accuracy
        models = {}

        # Function to evaluate, log, and save models
        def evaluate_and_log_model(model, param_grid, model_name, X_train,y_train):
            with mlflow.start_run(run_name=model_name):
                # GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)  # Use the best parameters from training
                
                # Get the best model from GridSearchCV
                best_model = grid_search.best_estimator_
                
                # Evaluate on the test dataset
                y_test_pred = best_model.predict(X_test)
                test_acc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                test_precision = precision_score(y_test, y_test_pred, average='weighted')
                test_recall = recall_score(y_test, y_test_pred, average='weighted')
                
                # Log parameters, metrics, and the model with MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({
                    "test_accuracy": test_acc, 
                    "test_f1_score": test_f1,
                    "test_precision": test_precision, 
                    "test_recall": test_recall
                })
                # Log the best model to MLflow
                mlflow.sklearn.log_model(best_model, model_name)
                
                # Save the model as a pickle file locally
                pickle_file = f"src/models/{model_name}_model.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"Model saved as {model_name}_model.pkl")
                
                # Log the pickle file as an artifact in MLflow
                mlflow.log_artifact(pickle_file)
                print(f"Pickle file '{pickle_file}' logged as an artifact in MLflow.")
                
                # Log the scaler and encoder if they exist
                scaler_file = 'src/models/scaler.pkl'
                encoder_file = 'src/models/onehot_encoder.pkl'
                
                if os.path.exists(scaler_file):
                  mlflow.log_artifact(scaler_file)
                  print(f"Scaler file '{scaler_file}' logged as an artifact in MLflow.")
            
                if os.path.exists(encoder_file):
                    mlflow.log_artifact(encoder_file)
                    print(f"Encoder file '{encoder_file}' logged as an artifact in MLflow.")

                # Print evaluation results
                print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                print(f"Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
                print("Test classification report:")
                print(classification_report(y_test, y_test_pred))

                # Return the model, accuracy, and metrics for comparison
                return best_model, test_acc, {"accuracy": test_acc, "f1_score": test_f1, "precision": test_precision, "recall": test_recall}

        # Evaluate models on the test set
        best_logistic_model, logistic_acc, logistic_metrics = evaluate_and_log_model(LogisticRegression(), logreg_params, "Logistic_Regression", X_train, y_train)
        best_rf_model, rf_acc, rf_metrics = evaluate_and_log_model(RandomForestClassifier(), rf_params, "Random_Forest", X_train, y_train)
        best_xgb_model, xgb_acc, xgb_metrics = evaluate_and_log_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_params, "XGBoost", X_train, y_train)

        # # Compare test accuracies and determine the best model
        # models = {
        #     "Logistic_Regression": (best_logistic_model, logistic_acc, logistic_metrics),
        #     "Random_Forest": (best_rf_model, rf_acc, rf_metrics),
        #     "XGBoost": (best_xgb_model, xgb_acc, xgb_metrics)
        # }

        # best_model_name = max(models, key=lambda model: models[model][1])
        # best_model, best_acc, best_metrics = models[best_model_name]

        # # Register the best model in MLflow Model Registry
        # with mlflow.start_run(run_name=f"Best_Model_Registration_{best_model_name}"):
        #     # Log the best model to MLflow
        #     mlflow.sklearn.log_model(best_model, best_model_name)

        #     # Register the best model in the registry
        #     best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}"
        #     model_version = mlflow.register_model(best_model_uri, f"{best_model_name}_Model")

        #     print(f"Best model '{best_model_name}' with test accuracy {best_acc:.4f} registered in MLflow as version {model_version}.")

        # # Save the scaler and encoder files to MLflow artifacts
        # scaler_file = 'src/models/scaler.pkl'
        # encoder_file = 'src/models/onehot_encoder.pkl'
        # if os.path.exists(scaler_file):
        #         mlflow.log_artifact(scaler_file)
        #         print(f"Scaler file '{scaler_file}' logged as an artifact in MLflow.")
        # if os.path.exists(encoder_file):
        #         mlflow.log_artifact(encoder_file)
        #         print(f"Encoder file '{encoder_file}' logged as an artifact in MLflow.")
        # Return the best model and its metrics
        # return {
        #     "Best_Model_Name": best_model_name,
        #     "Best_Model": best_model,
        #     "Best_Model_Metrics": best_metrics
        # }
        return {
            "Logistic_Regression": (best_logistic_model, logistic_metrics),
            "Random_Forest": (best_rf_model, rf_metrics),
            "XGBoost": (best_xgb_model, xgb_metrics)
        }




Truck Delay Prediction — End-to-End ML Pipeline
A three-part series covering data engineering, model development, and production deployment.

Part 1 — Data Ingestion, Exploration & Feature Engineering
1.1 Data Ingestion

Upload raw truck delay datasets to GitHub repository as the data source
Provision a MySQL/PostgreSQL database server (with planned migration to AWS RDS)
Design and create the truck_delays database schema with normalized tables for each data entity
Develop a modular data_ingestion pipeline component to programmatically ingest data from GitHub into the database

1.2 Exploratory Data Analysis (EDA)
Connect to the database, retrieve each table into Pandas DataFrames, and perform the following for each:
Data Quality Checks

Inspect schema, data types, shape, and statistical summaries (info(), describe())
Parse and standardize date columns: df['date'] = pd.to_datetime(df['date'])

Feature-Level Analysis — with written observations and recommendations for each:

Drivers — Histogram distributions for all numeric features; scatter plot of Rating vs. Average Speed; box plot of Driver Ratings segmented by Gender
Trucks — Histogram distributions for numeric features; identification of low-mileage trucks; truck age distribution analysis
Routes — Histogram distributions for numeric route features
Traffic — Histogram distributions; time-of-day categorization into periods (e.g., Early Morning: 03:00–06:00, Morning Rush, Afternoon, Evening Rush, Night)

1.3 Data Cleaning
For each DataFrame:

Identify, analyze, and treat missing/null values using appropriate imputation strategies
Detect and handle outliers using statistical methods (IQR, z-score)

1.4 Feature Store Integration (Hopsworks)

Engineer features and create Feature Groups in the Hopsworks Feature Store
Validate and retrieve feature datasets from the Feature Store for downstream modeling


Part 2 — Model Development, Experimentation & Deployment
2.1 Data Preparation

Retrieve the truck delay feature dataset from Hopsworks Feature Store
Validate dataset integrity and treat any remaining null values
Remove identifier columns and select modeling-relevant features
Verify date ranges and perform chronological train/validation/test split to prevent data leakage

2.2 Feature Encoding & Scaling

Apply OneHotEncoder to categorical features; generate and assign encoded column names
Transform train, validation, and test sets; drop original categorical columns
Fit StandardScaler on training data; apply to X_train, X_valid, and X_test
Persist encoder and scaler as encoder.pkl and scaler.pkl for production use

2.3 Model Development with MLflow
Train, track, and evaluate three classification models using MLflow experiment tracking, GridSearchCV hyperparameter tuning, and cross-validation:

Logistic Regression — baseline model with regularization tuning
Random Forest — ensemble method with n_estimators, max_depth, min_samples_split tuning
XGBoost — gradient boosting with learning_rate, max_depth, subsample tuning

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC. Best model registered to MLflow Model Registry.

Part 3 — Streamlit Application & AWS Deployment
3.1 Application Development (app.py)

Connect to Hopsworks Feature Store to retrieve the final merged inference dataset
Load the registered production model, encoder.pkl, and scaler.pkl from MLflow Model Registry
Build an interactive Streamlit UI for real-time truck delay predictions with input validation and result visualization

3.2 AWS Deployment

Containerize the application using Docker
Deploy Streamlit app on AWS EC2 with production configuration
Configure environment variables, security groups, and port forwarding for public access

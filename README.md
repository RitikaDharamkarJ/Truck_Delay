# 🚛 Truck Delay Prediction — End-to-End ML Pipeline

> A three-part ML engineering series covering data ingestion, feature engineering, model development, and production deployment on AWS.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-AWS_RDS-336791?style=flat-square&logo=postgresql)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-orange?style=flat-square&logo=mlflow)
![AWS](https://img.shields.io/badge/AWS-SageMaker_|_EC2_|_S3-FF9900?style=flat-square&logo=amazonaws)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature_Store-6C3483?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-green?style=flat-square)

---

## 📌 Project Overview

This project builds a production-ready machine learning pipeline to predict truck shipment delays using historical traffic, weather, route, and driver data. The goal is to enable logistics teams to proactively identify high-risk shipments and optimize operational decisions.

The pipeline is structured as a **three-part series**:

| Part | Focus | Status |
|------|-------|--------|
| Part 1 | Data Ingestion, EDA & Feature Engineering | ✅ Complete |
| Part 2 | Model Development, Hyperparameter Tuning & MLflow | ✅ Complete |
| Part 3 | Streamlit App & AWS Deployment | ✅ Complete |

---

## 🏗️ Architecture

```
GitHub (Raw Data)
       ↓
MySQL / PostgreSQL DB  ──→  Hopsworks Feature Store
                                      ↓
                          Train / Validation / Test Split
                                      ↓
                    ┌─────────────────────────────────┐
                    │     MLflow Experiment Tracking   │
                    │  Logistic Regression             │
                    │  Random Forest                   │
                    │  XGBoost ← Best Model            │
                    └─────────────────────────────────┘
                                      ↓
                          MLflow Model Registry
                                      ↓
                    Streamlit App → AWS EC2 Deployment
```

---

## 📂 Repository Structure

```
truck-delay-prediction/
│
├── data/                          # Raw CSV datasets
│   ├── trucks.csv
│   ├── drivers.csv
│   ├── routes.csv
│   ├── traffic.csv
│   └── weather.csv
│
├── notebooks/
│   ├── Part1_Data_Ingestion_EDA.ipynb
│   ├── Part2_Model_Development.ipynb
│   └── Part3_Deployment.ipynb
│
├── src/
│   ├── data_ingestion.py          # DB ingestion pipeline
│   ├── feature_engineering.py     # Feature store pipeline
│   ├── train.py                   # Model training & MLflow
│   └── predict.py                 # Inference pipeline
│
├── app.py                         # Streamlit application
├── encoder.pkl                    # Saved OneHotEncoder
├── scaler.pkl                     # Saved StandardScaler
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔬 Part 1 — Data Ingestion, EDA & Feature Engineering

### 1.1 Data Ingestion
- Uploaded raw truck delay datasets to GitHub as the primary data source
- Provisioned a **MySQL/PostgreSQL** database with planned migration to **AWS RDS**
- Designed the `truck_delays` schema with normalized tables for each data entity
- Developed a modular `data_ingestion` pipeline to programmatically load data from GitHub into the database

### 1.2 Exploratory Data Analysis

For each DataFrame (Trucks, Drivers, Routes, Traffic, Weather):

**Data Quality Checks**
- Schema inspection, data types, shape, and statistical summaries (`info()`, `describe()`)
- Date column standardization: `df['date'] = pd.to_datetime(df['date'])`

**Feature-Level Analysis**

| Dataset | Analysis Performed |
|---------|-------------------|
| **Drivers** | Numeric histograms · Rating vs. Speed scatter plot · Driver Ratings by Gender box plot |
| **Trucks** | Numeric histograms · Low-mileage truck identification · Truck age distribution |
| **Routes** | Numeric histograms · Route distance & duration distributions |
| **Traffic** | Numeric histograms · Time-of-day categorization |

**Traffic Time-of-Day Categorization:**
```python
def categorize_time(hour):
    if 300 <= hour < 600:   return 'Early Morning'
    elif 600 <= hour < 900:  return 'Morning Rush'
    elif 900 <= hour < 1200: return 'Mid Morning'
    elif 1200 <= hour < 1500: return 'Afternoon'
    elif 1500 <= hour < 1800: return 'Evening Rush'
    elif 1800 <= hour < 2100: return 'Evening'
    else:                    return 'Night'
```

### 1.3 Data Cleaning
- **Null Value Treatment** — Identified and imputed missing values using median/mode strategies per feature
- **Outlier Detection & Treatment** — Applied IQR and z-score methods to detect and cap/remove outliers

### 1.4 Hopsworks Feature Store
- Engineered features and created **Feature Groups** in Hopsworks Feature Store
- Validated feature schemas and retrieved datasets for downstream modeling

---

## 🤖 Part 2 — Model Development & Experimentation

### 2.1 Data Preparation
- Retrieved truck delay feature dataset from **Hopsworks Feature Store**
- Removed identifier columns; selected modeling-relevant features
- Performed **chronological train/validation/test split** to prevent data leakage

```python
# Chronological split to prevent data leakage
train_df = df[df['date'] < '2023-01-01']
valid_df = df[(df['date'] >= '2023-01-01') & (df['date'] < '2023-07-01')]
test_df  = df[df['date'] >= '2023-07-01']
```

### 2.2 Feature Encoding & Scaling
```python
# One-Hot Encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_enc = encoder.fit_transform(X_train[cat_cols])

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_cols])

# Persist artifacts
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### 2.3 Model Training with MLflow

Three classification models trained with **GridSearchCV hyperparameter tuning** and tracked via **MLflow**:

| Model | Key Hyperparameters Tuned | 
|-------|--------------------------|
| Logistic Regression | `C`, `solver`, `max_iter` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| **XGBoost** ⭐ | `learning_rate`, `max_depth`, `subsample`, `n_estimators` |

**Evaluation Metrics:** Accuracy · Precision · Recall · F1-Score · ROC-AUC

```python
with mlflow.start_run(run_name="XGBoost_Final"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": roc_auc})
    mlflow.xgboost.log_model(model, "model")
```

> ✅ Best performing model registered to **MLflow Model Registry** for production deployment.

---

## 🚀 Part 3 — Streamlit Application & AWS Deployment

### 3.1 Streamlit Application (`app.py`)
- Connected to **Hopsworks Feature Store** to retrieve the final merged inference dataset
- Loaded registered production model, `encoder.pkl`, and `scaler.pkl` from **MLflow Model Registry**
- Built an interactive UI for real-time truck delay predictions with input validation and result visualization

### 3.2 AWS Deployment
- Containerized application using **Docker**
- Deployed on **AWS EC2** with production configuration
- Configured security groups, environment variables, and port forwarding for public access

```bash
# Build and run Docker container
docker build -t truck-delay-app .
docker run -p 8501:8501 truck-delay-app
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.9+ |
| **Database** | MySQL · PostgreSQL · AWS RDS |
| **Feature Store** | Hopsworks |
| **ML Libraries** | Scikit-learn · XGBoost · Pandas · NumPy |
| **Experiment Tracking** | MLflow |
| **Deployment** | Streamlit · Docker · AWS EC2 |
| **Visualization** | Matplotlib · Seaborn |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/RitikaDharamkarJ/portfolio.git
cd truck-delay-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Add your DB credentials, Hopsworks API key, MLflow tracking URI

# 5. Run data ingestion
python src/data_ingestion.py

# 6. Launch Streamlit app
streamlit run app.py
```

---

## 📊 Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | - | - | - |
| Random Forest | - | - | - |
| **XGBoost** ⭐ | - | - | - |

> 📝 Fill in your actual model metrics here once training is complete.

---

## 👩‍💻 Author

**Ritika Dharamkar**
Data Scientist & ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ritikadharamkar-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/ritikadharamkar)
[![GitHub](https://img.shields.io/badge/GitHub-RitikaDharamkarJ-black?style=flat-square&logo=github)](https://github.com/RitikaDharamkarJ)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-ff4d6d?style=flat-square)](https://ritikadharamkarj.github.io/portfolio)

---

*Part of a larger portfolio of ML & Data Science projects. See full portfolio at [datascienceportfol.io/ritikadharamkar](https://www.datascienceportfol.io/ritikadharamkar)*

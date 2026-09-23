# Telco Customer Churn Analytics Dashboard Engine

## Project Overview
This project applies machine learning classification to telecom customer records to predict churn risk. The engine analyzes contract types, tenure, payment methods, and demographic details to provide predictive business intelligence.

* **Dataset:** [Telco Customer Churn (IBM Sample Data Sets)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Model Type:** Random Forest Classifier
* **Saved Artifact:** `churn_model.pkl`

## Technologies Used
* Python
* Pandas (Data cleaning, manipulation, and encoding)
* Scikit-Learn (Model training, train/test splitting, and evaluation metrics)
* Joblib (Model persistence and serialization)

## Setup and Execution
1. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is placed in the project root.
2. Activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
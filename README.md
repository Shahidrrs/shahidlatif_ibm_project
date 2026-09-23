Markdown
# Executive Decision Dashboard: Telco Customer Churn Analytics

## Brief Overview
This project is an end-to-end business intelligence pipeline that applies machine learning to forecast telecommunications customer churn. Instead of merely outputting statistical probabilities, the project transforms raw data into an interactive Executive Decision Dashboard. It automatically cleans the data, runs predictions using a Scikit-Learn Pipeline, and generates actionable retention strategies (e.g., targeted discounts, support interventions) based on the "Fact ➔ Insight ➔ Opportunity ➔ Action" executive framework.

**Dataset:** [Telco Customer Churn (IBM Sample Data Sets) on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Technologies Used
* **Python 3.x**
* **Pandas:** Data manipulation and processing.
* **Scikit-Learn:** Machine learning (Random Forest Classifier) and Automated Pipelines (ColumnTransformer, SimpleImputer, OrdinalEncoder).
* **Streamlit:** Frontend web application framework for the interactive dashboard.
* **Matplotlib & Seaborn:** Data visualization and trend charting.
* **Joblib:** Model persistence and serialization.

## Setup and Run Instructions

### 1. Environment Setup
Download the dataset from the Kaggle link above and place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file into the root of your project directory. 

Open your terminal, navigate to the project folder, and create a virtual environment:
```bash
python -m venv .venv
Activate the virtual environment:

Windows (PowerShell): .\.venv\Scripts\Activate.ps1

Mac/Linux: source .venv/bin/activate

Install the required dependencies:

Bash
pip install -r requirements.txt
2. Train the Machine Learning Pipeline
Execute the core analytics script to train the data cleaning and predictive pipeline. This step will automatically handle missing values, encode text into numbers, train a Random Forest model, and save the packaged engine as churn_pipeline.pkl.

Bash
python ShahidLatif_PipelineAnalytics.py
3. Launch the Decision Dashboard
Once the .pkl file is generated, launch the interactive Streamlit web application:

Bash
streamlit run ShahidLatif_Dashboard.py
This will open the application in your default web browser (typically at http://localhost:8501). Upload the CSV file via the sidebar to view the executive KPIs, risk trends, and automated action queue.

Key Information & Architecture
Automated Data Pipeline: The training script utilizes sklearn.pipeline.Pipeline to bundle preprocessing and predictive modeling together. This ensures that any raw data uploaded to the dashboard is perfectly formatted before predictions are made, preventing application crashes.

Strategic Triad: The dashboard maps data directly to business value, categorizing outputs into specific Risks (revenue exposure), Opportunities (upsell paths), and Actions (concrete steps management should take to retain customers).

Dataset Guardrails: The Streamlit interface includes validation logic to verify the presence of required columns (tenure, Contract, MonthlyCharges), preventing execution errors if an incompatible dataset is uploaded.

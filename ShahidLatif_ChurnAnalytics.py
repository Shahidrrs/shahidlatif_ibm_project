import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the Dataset
# Ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Data Cleaning & Preprocessing
# Convert TotalCharges to numeric, dropping empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Drop CustomerID as it has no predictive value
df.drop('customerID', axis=1, inplace=True)

# Encode categorical variables (e.g., Yes/No, Gender, Internet Service)
encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# 3. Define Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("Classification Report:\n", classification_report(y_test, predictions))

# 7. Save Model for Future API Integration
joblib.dump(model, 'churn_model.pkl')
print("Model saved as churn_model.pkl")
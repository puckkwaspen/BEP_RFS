import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("BEP_imputed.csv")

print(data.isna().sum())

# Ensure patient_id is treated as a categorical variable
if 'PATIENT_ID' not in data.columns:
    raise ValueError("The dataset must contain a 'PATIENT_ID' column.")

# Step 1: Get unique patient IDs
unique_patients = data['PATIENT_ID'].unique()

# Step 2: Split patient IDs into train and test sets (e.g., 80% train, 20% test)
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

# Step 3: Assign data to train/test based on patient_id
X_train = data[data['PATIENT_ID'].isin(train_patients)]
X_test = data[data['PATIENT_ID'].isin(test_patients)]

print(f"Total Patients: {len(unique_patients)}")
print(f"Train Patients: {len(train_patients)}, Train Samples: {len(X_train)}")
print(f"Test Patients: {len(test_patients)}, Test Samples: {len(X_test)}")

# Step 4: Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)  # Adjust nu for sensitivity
oc_svm.fit(X_train_scaled)

# Step 6: Predict Anomalies
train_preds = oc_svm.predict(X_train_scaled)  # 1 (normal), -1 (anomaly)
test_preds = oc_svm.predict(X_test_scaled)

# Print results
print(f"Training Anomalies: {sum(train_preds == -1)} out of {len(train_preds)}")
print(f"Testing Anomalies: {sum(test_preds == -1)} out of {len(test_preds)}")
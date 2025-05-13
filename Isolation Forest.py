import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

# Load data
data = pd.read_csv("BEP_imputed.csv")
test_data = pd.read_csv("BEP_imputed_TEST.csv")

# Separate test labels (and drop them from feature set)
y_test_true = test_data['RFS'].values  # ground truth labels
X_test = test_data.drop(columns=['RFS'])

# Keep train data as is (assuming no labels in it)
X_train = data

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train_scaled)

# Predict anomalies
y_train_pred = model.predict(X_train_scaled)  # 1 = normal, -1 = anomaly
y_test_pred = model.predict(X_test_scaled)

# Convert IsolationForest predictions to binary: 1 = anomaly, 0 = normal
y_test_pred_binary = (y_test_pred == -1).astype(int)

# Assume RFS = 1 means an actual anomaly
y_test_true_binary = (y_test_true == 1).astype(int)

# Compute metrics
precision = precision_score(y_test_true_binary, y_test_pred_binary)
recall = recall_score(y_test_true_binary, y_test_pred_binary)
f1 = f1_score(y_test_true_binary, y_test_pred_binary)

# Output
print(f"Anomalies identified in training set: {np.sum(y_train_pred == -1)}")
print(f"Anomalies identified in test set:    {np.sum(y_test_pred == -1)}")
print(f"True anomalies in test set (according to ASPEN): {np.sum(y_test_true_binary == 1)}")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

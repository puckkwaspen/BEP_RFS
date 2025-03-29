import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("BEP_imputed.csv")
X = data.values  # Adjust based on your dataset structure

# Optional: Preprocessing (Standardizing the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_scaled)

# Predict anomalies
y_pred = model.predict(X_scaled)
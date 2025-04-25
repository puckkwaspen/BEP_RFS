import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Load data
data = pd.read_csv("BEP_imputed.csv")

X = data.drop(columns=['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE'], errors='ignore')

X_array = X.to_numpy()

# Create the LOF model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)

# Fit and predict (−1 for outliers, 1 for inliers)
y_pred = lof.fit_predict(X_array)

# Get anomaly scores (the lower, the more likely it’s an outlier)
scores = lof.negative_outlier_factor_

unique, counts = np.unique(y_pred, return_counts=True)
outlier_counts = dict(zip(unique, counts))
clean_counts = {int(k): int(v) for k, v in outlier_counts.items()}
print("Counts: ", clean_counts)

# Print outliers
for i, (point, pred, score) in enumerate(zip(X_array, y_pred, scores)):
    if pred == -1:  # Only print if it's an outlier
        print(f"Point {i}: {point}, Prediction: Outlier, Score: {score}")


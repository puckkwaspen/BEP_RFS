import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("BEP_imputed.csv")

# Select the features (in this case all the dynamic ones)
features = ['ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium',
        'Magnesium', 'Weight (kg)', 'BMI', 'Temperature (C)', 'Systolic', 'Diastolic', 'Leucocytes']
X = df[features].values

# Scale the data
# mean = 0 and variance = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# run the Principal Component Analysis
pca = PCA() # n_components=5
X_pca = pca.fit_transform(X_scaled)

# Step 3: Scree plot
explained_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o', label='Explained Variance')
plt.title('Explained Proportion of Variance per Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.grid(True)
plt.legend()
plt.show()

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.title('Cumulative Variance Explained')
plt.show()

print(f"The Principal Components explain the following proportion of the variance: {pca.explained_variance_ratio_}\n")

# # Find number of components for Kaiser criterion
# kaiser_components = np.sum(eigenvalues > 1)
# print(f"Kaiser criterion (>70% of variance) suggests keeping {kaiser_components} components.\n")

# Create a plot to visualize the PCA structure of the first 2 PCs
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.show()

####### Feature Importance Scores #######

# Get PCA components
loadings = pca.components_[:3, :]  # for the first n PCs, adjust if needed

# Compute average absolute loading per feature across the PCs
feature_importance = np.median(np.abs(loadings), axis=0)

# Map to feature names
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance (avg |loading| over PC1–PC8)': feature_importance
}).sort_values(by='Importance (avg |loading| over PC1–PC8)', ascending=False)

print(importance_df)

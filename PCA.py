import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

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
pca = PCA(n_components=8) # n_components=5
X_pca = pca.fit_transform(X_scaled)


############################ VISUALIZATIONS AND FEATURE IMPORTANCE SCORES ############################

### HEATMAP ###
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
plt.figure(figsize=(10, 8))
sns.heatmap(loadings,
            annot=True,
            cmap='coolwarm',
            xticklabels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'],
            vmin=-1,
            vmax=1,
            yticklabels=features)
plt.title('Feature Importance in Principal Components')
plt.show()
#############################

### BIPLOT ###
coeff = pca.components_.T

xs = X_pca[:, 0]
ys = X_pca[:, 1]
plt.figure(figsize=(10, 8))
plt.scatter(xs, ys, color="blue", alpha=0.5)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot")
plt.show()
#############################


### EXPLAINED AND CUMULATIVE VARIANCE ###
explained_var = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_var)

plt.figure(figsize=(10, 6))

# Explained variance (solid dark blue)
plt.plot(
    range(1, len(explained_var) + 1),
    explained_var,
    marker='o',
    color='blue',
    label='Explained Variance'
)

# Cumulative variance (dotted lighter blue)
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o',
    linestyle='--',
    color='blue',
    label='Cumulative Explained Variance'
)

# Add horizontal line at 0.9
plt.axhline(y=0.9, color='orange', linestyle='-', linewidth=1.5, label='90% Threshold')

# Labels and formatting
plt.title('Explained and Cumulative Variance per Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


print(f"The Principal Components explain the following proportion of the variance: {pca.explained_variance_ratio_}\n")
############################################


####### Feature Importance Scores #######

# Get PCA components
loadings = pca.components_[:7, :]  # for the first n PCs, adjust if needed

# Compute average absolute loading per feature across the PCs
feature_importance = np.median(np.abs(loadings), axis=0)

# Map to feature names
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance (avg |loading| over PC1–PC8)': feature_importance
}).sort_values(by='Importance (avg |loading| over PC1–PC8)', ascending=False)

print(importance_df)

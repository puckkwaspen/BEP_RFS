import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Load the dataset
df = pd.read_csv("BEP_imputed.csv")

# Select the features (in this case all the dynamic ones)
features = ['ALT', 'AST', 'Phosphate', 'Potassium', 'Magnesium']
X = df[features].values

# Scale the data
# mean = 0 and variance = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# run the Principal Component Analysis
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)


############################VISUALIZATIONS AND FEATURE IMPORTANCE SCORES ############################

########################### PLOT 1 -> SCREE PLOT ############################
eigenvalues = pca.explained_variance_

plt.figure(figsize=(10, 6))

# Scree plot using eigenvalues
plt.plot(
    range(1, len(eigenvalues) + 1),
    eigenvalues,
    marker='o',
    color='blue',
    label='Eigenvalue'
)

# Labels and formatting
plt.title('Scree Plot of Principal Components', fontsize=20)
plt.xlabel('Principal Component', fontsize=14)
plt.ylabel('Eigenvalue', fontsize=14)
plt.grid(True)
plt.xticks(range(1, len(eigenvalues) + 1), fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

#############################################################


#########################  PLOT 2 --> EXPLAINED AND CUMULATIVE VARIANCE #########################
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

# Labels and formatting
plt.title('Explained and Cumulative Variance per Principal Component', fontsize = 20)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Variance Ratio', fontsize = 16)
plt.ylim(bottom=0, top=1)
plt.grid(True)
plt.legend(fontsize = 16)
plt.xticks(range(1, len(explained_var) + 1), fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()


print(f"The Principal Components explain the following proportion of the variance: {pca.explained_variance_ratio_}\n")

#############################################################

# run the Principal Component Analysis
pca = PCA(n_components=2)    # now with just the two selected components
X_pca = pca.fit_transform(X_scaled)

#########################  PLOT 3 --> FEATURE IMPORTANCE SCORES #########################
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

plt.figure(figsize=(10, 8))
sns.heatmap(loadings,
            annot=True,
            annot_kws={"size": 16},  # Larger font for numbers
            cmap='coolwarm',
            xticklabels=['PC1', 'PC2'],
            yticklabels=features,
            vmin=-1,
            vmax=1,
            cbar=False)  # Remove legend

plt.title('Feature Importance in Principal Components', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=0)
plt.show()

########################################################################

#########################  PLOT 4 --> BIPLOT #########################
coeff = pca.components_.T

# Scale the arrows so they fit nicely (optional factor)
scaling_factor = 3
xs = X_pca[:, 0]
ys = X_pca[:, 1]
plt.figure(figsize=(10, 8))

# Scatter plot of sample scores
plt.scatter(xs, ys, color="grey", alpha=0.5)

# Add arrows for each feature
for i in range(len(features)):
    plt.arrow(0, 0,
              coeff[i, 0] * scaling_factor,
              coeff[i, 1] * scaling_factor,
              color='blue', alpha=0.7, head_width=0.3, length_includes_head=True, linewidth = 3)
    plt.text(coeff[i, 0] * scaling_factor * 1.15,
             coeff[i, 1] * scaling_factor * 1.15,
             features[i], color='blue', fontsize=20, weight = 4)

# Axes labels and title
plt.xlabel("PC1", fontsize=22)
plt.ylabel("PC2", fontsize=22)
plt.title("Biplot", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)
plt.tight_layout()
plt.show()

########################################################################

################# Feature Importance Scores #################

# Get PCA components
loadings = pca.components_[:1, :]  # for the first n PCs, adjust if needed

# Compute average absolute loading per feature across the PCs
feature_importance = np.median(np.abs(loadings), axis=0)

# Map to feature names
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance (avg |loading| over PC1–PC2)': feature_importance
}).sort_values(by='Importance (avg |loading| over PC1–PC2)', ascending=False)

print(importance_df)

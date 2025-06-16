import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
import warnings
import seaborn as sns

# Ignore this warning (safe to do so)
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

# Load data
df = pd.read_csv("Data/BEP_imputed.csv")

####################### PLOT 1 --> CORRELATION MATRIX ###########################################

features = ['Phosphate', 'Potassium', 'Magnesium', 'ALT', 'AST', 'BMI', 'AGE']
X = df[features]
X_scaled = RobustScaler().fit_transform(X)

# Set up figure and axes with constrained layout
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

# Plot correlation matrix
corr_matrix = pd.DataFrame(X_scaled).corr(method='spearman')
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

# Tick labels
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha='right')
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)

# Title with padding to prevent overlap
ax.set_title("Correlation Matrix", pad=10)

# Add correlation values
for i in range(len(features)):
    for j in range(len(features)):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)

plt.show()

###########################################################################################

# Select enzyme + electrolyte features
features = ['Phosphate', 'Potassium', 'Magnesium', 'ALT', 'AST']
X = df[features]
X_scaled = RobustScaler().fit_transform(X)

sns.kdeplot(np.log1p(df['ALT']), fill=True)
plt.title('Log-Transformed Density Plot of AST')
plt.xlabel('log(AST + 1)')
plt.show()

################# PLOT 2 --> CORRELATION MATRIX #################

# Set up figure and axes with constrained layout
fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

# Plot correlation matrix
corr_matrix = np.corrcoef(X_scaled.T)
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

# Tick labels
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha='right')
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)

# Title with padding to prevent overlap
ax.set_title("Correlation matrix of enzyme and electrolyte features", pad=10)

# Add correlation values
for i in range(len(features)):
    for j in range(len(features)):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)

plt.show()

#######################################################################


################# PLOT 3 --> SCREE PLOT #################
# FA setup
fa = FactorAnalyzer(n_factors=5, method='principal')  # 'principal' = PAF
fa.fit(X_scaled)

# Return two arrays, only select the first one (original eigen values)
eigenvalues, _ = fa.get_eigenvalues()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', color='blue')
plt.title('Scree Plot (PAF)', fontsize=16)
plt.xlabel('Factor Number', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.xticks(range(1, len(eigenvalues) + 1))
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.show()

#######################################################################


################# PLOT 4 --> EXPLAINED AND CUMULATIVE VARIANCE #################
# Get variance outputs
variance, prop_var, cum_var = fa.get_factor_variance()
# Plot
plt.figure(figsize=(10, 6))

# Explained variance
plt.plot(range(1, len(prop_var) + 1), prop_var, marker='o', color='blue', label='Explained Variance')

# Cumulative variance
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--', color='blue', label='Cumulative Explained Variance')

# Labels
plt.title('Explained and Cumulative Variance per Factor (PAF)', fontsize=20)
plt.xlabel('Factor Number', fontsize=16)
plt.ylabel('Variance Ratio', fontsize=1)
plt.ylim(bottom=0, top=1)
plt.grid(True)
plt.legend(fontsize = 16)
plt.xticks(range(1, len(prop_var) + 1), fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

# Print variance values
print("Proportional variance per factor:", np.round(prop_var, 3))
print("Cumulative variance per factor:", np.round(cum_var, 3))

#######################################################################


################# PLOT 5 --> FACTOR ANALYSIS LOADINGS #################

# Define number of factors and methods
n_comps = 2
methods = [
    ("Unrotated FA", FactorAnalyzer(n_factors=n_comps, method='principal', rotation=None)),
    ("Varimax FA", FactorAnalyzer(n_factors=n_comps, method='principal', rotation='varimax')),
    ("Promax FA", FactorAnalyzer(n_factors=n_comps, method='principal', rotation='promax')),
]

# Set up figure
fig, axes = plt.subplots(ncols=len(methods), figsize=(8, 5), sharey=True, constrained_layout=True)

# Loop through each FA method
for ax, (title, fa) in zip(axes, methods):
    fa.fit(X_scaled)
    components = fa.loadings_  # Use .loadings_ instead of .components_

    vmax = np.abs(components).max()
    im = ax.imshow(components, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(range(n_comps))
    ax.set_xticklabels([f"Factor {i+1}" for i in range(n_comps)], fontsize=14, rotation=45)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=14)

    # Annotate loadings
    for i in range(len(features)):
        for j in range(n_comps):
            ax.text(j, i, f"{components[i, j]:.2f}", ha='center', va='center', color='black', fontsize=15)

fig.suptitle("Factor Analysis Loadings (PAF)", fontsize=23)
plt.show()

#######################################################################








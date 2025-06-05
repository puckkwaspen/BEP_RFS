import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, fbeta_score, pairwise_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
import shap

# 🔧 Hyperparameters:
# n_neighbors: 20
# contamination: 0.05

# Paths to data
train_path = "Data/BEP_imputed.csv"
test_path = "Data/BEP_imputed_TEST.csv"

# Bloodwork features to plot
blood_features = [
    "Phosphate", "Potassium", "Magnesium", "Glucose", "ALT", "AST", "Weight (kg)",
    "Leucocytes", "Systolic", "Diastolic", "Temperature (C)"
]
exclude_cols = ['DATE', 'SEQUENCE', 'INTAKE_ID', 'PATIENT_ID', 'DAYS_SINCE_ADMISSION', 'AGE', 'Height (m)']
group_col = 'PATIENT_ID'
time_col = 'DAYS_SINCE_ADMISSION'

# Load and preprocess
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def apply_manual_corrections(df):
    corrections = {
        893: {'drop_seq': [1]},
        1048: {'drop_seq': [1]},
        530: {'drop_seq': [14]},
        390: {'drop_seq': [27, 28, 29]},
        1171: {'drop_seq': [16, 17, 18, 19, 20, 21]},
        1576: {'drop_seq': [16, 17, 18, 19, 20, 21, 22, 23, 24]},
        1231: {'keep_seq': list(range(2, 16))}
    }
    for pid, ops in corrections.items():
        if 'drop_seq' in ops:
            df = df[~((df['PATIENT_ID'] == pid) & (df['SEQUENCE'].isin(ops['drop_seq'])))]
        if 'keep_seq' in ops:
            df = df[~((df['PATIENT_ID'] == pid) & (~df['SEQUENCE'].isin(ops['keep_seq'])))]
    return df

test_df = apply_manual_corrections(test_df)

# Prepare input
y_true = (test_df['RFS'].values == 1).astype(int)
X_train = train_df.drop(columns=exclude_cols)
X_test = test_df.drop(columns=exclude_cols + ['RFS'])

# Scale blood features
scaler = RobustScaler()
X_train[blood_features] = scaler.fit_transform(X_train[blood_features])
X_test[blood_features] = scaler.transform(X_test[blood_features])

model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)

model.fit(X_train)  # only fit on training data
y_test_pred = model.predict(X_test)
lof_scores = -model.decision_function(X_test)  # higher = more anomalous

top_n = 21 # amount of anomalies flagged
top_indices = np.argsort(lof_scores)[-top_n:]
X_top_anomalies = X_test.iloc[top_indices]

def analyze_lof_anomaly(x, X_train, k=20):
    dists = pairwise_distances(X_train, x.reshape(1, -1))
    nearest_indices = np.argsort(dists.ravel())[:k]
    neighbors = X_train.iloc[nearest_indices]
    diff = (x - neighbors.mean()).abs()  # Feature-wise deviation from neighbors
    return diff.sort_values(ascending=False)

# Analyze a single top anomaly
idx = top_indices[-1]  # Most anomalous
x_anom = X_test.iloc[idx].values
diffs = analyze_lof_anomaly(x_anom, X_train)

print("Top feature deviations from neighbors for most anomalous point:")
print(diffs.head(10))

model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

# Combine and run LOF
X_all = np.vstack([X_train.values, X_test.values])
y_pred_all = model.fit_predict(X_all)
y_test_pred = y_pred_all[len(X_train):]
y_test_pred_binary = (y_test_pred == -1).astype(int)

# Evaluation
precision = precision_score(y_true, y_test_pred_binary, zero_division=0)
recall = recall_score(y_true, y_test_pred_binary, zero_division=0)
accuracy = accuracy_score(y_true, y_test_pred_binary)
f2 = fbeta_score(y_true, y_test_pred_binary, beta=2, zero_division=0)

print(f"\n🔍 LOF predicted {np.sum(y_test_pred_binary)} anomalies out of {len(y_test_pred_binary)} test instances")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f} | F2 Score: {f2:.4f}")

# FP analysis
test_df['prediction'] = y_test_pred_binary
test_df['true_label'] = y_true

false_positives = test_df[(test_df['prediction'] == 0) & (test_df['true_label'] == 1)]
test_df_scaled = test_df.copy()
test_df_scaled[blood_features] = X_test[blood_features]  # Scaled blood features
test_df_scaled = apply_manual_corrections(test_df_scaled)

# Percent change in phosphate from SEQ 1
percent_changes = []
for idx, row in false_positives.iterrows():
    patient_id = row['PATIENT_ID']
    phosphate_now = row.get('Phosphate', np.nan)
    baseline = test_df[(test_df['PATIENT_ID'] == patient_id) & (test_df['SEQUENCE'] == 1)]
    if not baseline.empty:
        phosphate_base = baseline.iloc[0].get('Phosphate', np.nan)
        if pd.notna(phosphate_now) and pd.notna(phosphate_base) and phosphate_base != 0:
            pct_change = ((phosphate_now - phosphate_base) / phosphate_base) * 100
            percent_changes.append(pct_change)
            print(f"Patient {patient_id} – % change in phosphate: {pct_change:.2f}%")

if percent_changes:
    print(f"\n🔬 Average % change in phosphate for false negatives: {np.mean(percent_changes):.2f}%")
else:
    print("\n⚠️ No valid phosphate comparisons could be made.")

# Time series plots for FP patients
for patient_id in false_positives[group_col].unique():
    patient_data = test_df_scaled[test_df_scaled[group_col] == patient_id].sort_values(by=time_col)
    patient_ilocs = test_df_scaled.index.get_indexer_for(patient_data.index)
    model_anomaly_days = patient_data.iloc[y_test_pred_binary[patient_ilocs] == 1][time_col].values

    plt.figure(figsize=(10, 6))
    for feature in blood_features:
        if feature in patient_data.columns:
            plt.plot(patient_data[time_col], patient_data[feature], label=feature)

    model_set = set(model_anomaly_days)
    rfs_set = set(patient_data[patient_data['RFS'] == 1][time_col].values)
    both_anomalies = model_set & rfs_set
    model_only = model_set - both_anomalies
    rfs_only = rfs_set - both_anomalies

    for day in model_only:
        plt.axvline(x=day, color='blue', linestyle='--', linewidth=1, label='Model Anomaly' if day == list(model_only)[0] else "")
    for day in rfs_only:
        plt.axvline(x=day, color='red', linestyle='--', linewidth=1, label='RFS Label' if day == list(rfs_only)[0] else "")
    for day in both_anomalies:
        plt.axvline(x=day, color='green', linestyle='--', linewidth=1, label='Both Anomaly & RFS' if day == list(both_anomalies)[0] else "")

    plt.title(f"LOF - Patient {patient_id} (scaled)", fontsize=18)
    plt.xlabel("Days Since Admission", fontsize=18)
    plt.ylabel("Scaled Lab Value (Robust)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 1. Compute median of training data for each feature
feature_medians = X_train[blood_features].median()

# 2. Calculate absolute deviation from medians
deviations = np.abs(X_test[blood_features] - feature_medians)

# 3. Select only test instances marked as anomalies
anomalies = deviations[y_test_pred_binary == 1]

# 4. Average deviation per feature
mean_deviation = anomalies.mean().sort_values(ascending=False)

# 5. Plot
plt.figure(figsize=(10, 6))
mean_deviation.plot(kind='barh', color='skyblue')
plt.title("Feature Deviations in LOF-Flagged Anomalies")
plt.xlabel("Mean Absolute Deviation from Training Median")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()





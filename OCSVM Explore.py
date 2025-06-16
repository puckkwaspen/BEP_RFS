import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, fbeta_score
from sklearn.preprocessing import RobustScaler
import shap
import random
import pickle

random.seed(42)

### best model has
# - Hyperparameters:
# • kernel: rbf
# • nu: 0.1
# • gamma: 0.1

# === Load the best model (trained earlier and saved in Models.py) ===
with open("PickleFiles/AllModels/model_delta_One-Class_SVM_kernel-rbf_nu-0.2_gamma-scale.pkl", "rb") as f:
    model = pickle.load(f)

blood_features = [
    "Phosphate_delta_per_day", "Potassium_delta_per_day", "Magnesium_delta_per_day",
    "Glucose_delta_per_day", "ALT_delta_per_day", "AST_delta_per_day",
    "Weight (kg)_delta_per_day", "Leucocytes_delta_per_day", "Systolic_delta_per_day",
    "Diastolic_delta_per_day", "Temperature (C)_delta_per_day"
]

group_col = 'PATIENT_ID'
time_col = 'DAYS_SINCE_ADMISSION'

# === Paths ===
train_path = "Data/BEP_imputed_delta.csv"
test_path = "Data/BEP_imputed_delta_TEST.csv"
raw_test_df = pd.read_csv("Data/BEP_imputed_TEST.csv")

# === Load data ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === Label vector ===
y_true = (test_df['RFS'].values == 1).astype(int)

# === Exclude columns used for identification, grouping, and labels ===
exclude_cols = ['DATE', 'SEQUENCE', 'INTAKE_ID', 'PATIENT_ID', 'DAYS_SINCE_ADMISSION', 'RFS', 'CONTROL']
X_train = train_df.drop(columns=exclude_cols, errors='ignore')
X_test = test_df.drop(columns=exclude_cols, errors='ignore')

# === Match model's feature columns exactly ===
required_cols = list(model.feature_names_in_)

X_train = X_train[required_cols]
X_test = X_test[required_cols]

# === Predict anomalies ===
y_test_pred = model.predict(X_test)
y_test_pred_binary = (y_test_pred == -1).astype(int)

# === Evaluate ===
precision = precision_score(y_true, y_test_pred_binary, zero_division=0)
recall = recall_score(y_true, y_test_pred_binary, zero_division=0)
accuracy = accuracy_score(y_true, y_test_pred_binary)
f2 = fbeta_score(y_true, y_test_pred_binary, beta=2, zero_division=0)

print(f"✅ OCSVM predicted {np.sum(y_test_pred_binary)} anomalies out of {len(y_test_pred_binary)} test instances")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}, F2 Score: {f2:.2f}")




# Only scale a copy for plotting
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = RobustScaler()
X_train_scaled[blood_features] = scaler.fit_transform(X_train_scaled[blood_features])
X_test_scaled[blood_features] = scaler.transform(X_test_scaled[blood_features])

# # Use scaled data for plotting
test_df_scaled = test_df.copy()
test_df_scaled[blood_features] = X_test_scaled[blood_features]

# FP analysis
test_df['prediction'] = y_test_pred_binary
test_df['true_label'] = y_true

false_negatives = test_df[(test_df['prediction'] == 0) & (test_df['true_label'] == 1)]
print(f"There are {len(false_negatives)} false negatives")
test_df_scaled = test_df.copy()
test_df_scaled[blood_features] = X_test_scaled[blood_features]  # Scaled blood features

print("Phosphate change (from original data) for false negatives:\n")
percent_changes = []

for idx, row in false_negatives.iterrows():
    patient_id = row['PATIENT_ID']
    sequence = row['SEQUENCE']

    # Get phosphate value for this FN row from original test data
    current_row = raw_test_df[(raw_test_df['PATIENT_ID'] == patient_id) & (raw_test_df['SEQUENCE'] == sequence)]
    baseline_row = raw_test_df[(raw_test_df['PATIENT_ID'] == patient_id) & (raw_test_df['SEQUENCE'] == 1)]

    if not current_row.empty and not baseline_row.empty:
        phosphate_now = current_row.iloc[0].get('Phosphate', np.nan)
        phosphate_base = baseline_row.iloc[0].get('Phosphate', np.nan)

        if pd.notna(phosphate_now) and pd.notna(phosphate_base) and phosphate_base != 0:
            pct_change = ((phosphate_now - phosphate_base) / phosphate_base) * 100
            percent_changes.append(pct_change)
            print(f"Patient {patient_id} (Seq {sequence}) – % change in phosphate: {pct_change:.2f}%")

if percent_changes:
    print(f"\n🔬 Average % change in phosphate for false negatives: {np.mean(percent_changes):.2f}%\n")
else:
    print("\n⚠️ No valid phosphate comparisons could be made.")


false_positives = test_df[(test_df['prediction'] == 1) & (test_df['true_label'] == 0)]

for idx, row in false_positives.iterrows():
    patient_id = row['PATIENT_ID']
    sequence = row['SEQUENCE']

    # Get phosphate value for this FN row from original test data
    current_row = raw_test_df[(raw_test_df['PATIENT_ID'] == patient_id) & (raw_test_df['SEQUENCE'] == sequence)]
    baseline_row = raw_test_df[(raw_test_df['PATIENT_ID'] == patient_id) & (raw_test_df['SEQUENCE'] == 1)]

    if not current_row.empty and not baseline_row.empty:
        phosphate_now = current_row.iloc[0].get('Phosphate', np.nan)
        phosphate_base = baseline_row.iloc[0].get('Phosphate', np.nan)

        if pd.notna(phosphate_now) and pd.notna(phosphate_base) and phosphate_base != 0:
            pct_change = ((phosphate_now - phosphate_base) / phosphate_base) * 100
            percent_changes.append(pct_change)
            print(f"Patient {patient_id} (Seq {sequence}) – % change in phosphate: {pct_change:.2f}%")


# Step 1: Define rename map for cleaner y-axis labels
rename_map = {
    "Phosphate_delta_per_day": "Phosphate",
    "Potassium_delta_per_day": "Potassium",
    "Magnesium_delta_per_day": "Magnesium",
    "Glucose_delta_per_day": "Glucose",
    "ALT_delta_per_day": "ALT",
    "AST_delta_per_day": "AST",
    "Weight (kg)_delta_per_day": "Weight (kg)",
    "Leucocytes_delta_per_day": "Leucocytes",
    "Systolic_delta_per_day": "Systolic",
    "Diastolic_delta_per_day": "Diastolic",
    "Temperature (C)_delta_per_day": "Temperature (C)"
}

custom_colors = {
    "Phosphate": "royalblue",
    "Potassium": "orange",
    "Magnesium": "green",
    "Glucose": "red",
    "ALT": "purple",
    "AST": "brown",
    "Weight (kg)": "hotpink",
    "Leucocytes": "gray",
    "Systolic": "olive",
    "Diastolic": "teal",
    "Temperature (C)": "black"
}

# Time series plots for FP patients
for patient_id in test_df_scaled[group_col].unique():
    patient_data = test_df_scaled[test_df_scaled[group_col] == patient_id].sort_values(by=time_col)
    patient_ilocs = test_df_scaled.index.get_indexer_for(patient_data.index)
    model_anomaly_days = patient_data.iloc[y_test_pred_binary[patient_ilocs] == 1][time_col].values

    # Check control status for title
    control_status = ""
    if not patient_data.empty and 'CONTROL' in patient_data.columns:
        if patient_data['CONTROL'].iloc[0] == 1:
            control_status = " (Control)"

    plt.figure(figsize=(10, 6))
    for feature in blood_features:
        if feature in patient_data.columns:
            label = rename_map.get(feature, feature)
            color = custom_colors.get(label, None)
            plt.plot(patient_data[time_col], patient_data[feature], label=label, color=color)

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

    plt.title(f"OCSVM - Patient {patient_id}{control_status} (scaled)", fontsize=18)
    plt.xlabel("Days Since Admission", fontsize=18)
    plt.ylabel("Scaled Lab Value (Robust)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Step 2: Summarize background set for SHAP to speed things up
background = shap.sample(X_train, 100, random_state=42)

# Step 3: Initialize KernelExplainer with model decision function
explainer = shap.KernelExplainer(model.decision_function, background)

# Step 4: Compute SHAP values for test set
shap_values = explainer(X_test)

# Step 5: Flip SHAP values so the plot is more intuitive
shap_values.values = -shap_values.values

# Step 6: Filter out unwanted features
exclude_features = ['AGE', 'Height (m)']
keep_indices = [i for i, name in enumerate(shap_values.feature_names) if name not in exclude_features]

# Apply filtering to SHAP components
shap_values.data = shap_values.data.iloc[:, keep_indices]  # FIXED: use .iloc for DataFrame
shap_values.values = shap_values.values[:, keep_indices]   # Still NumPy array, so normal slicing is fine
shap_values.feature_names = [shap_values.feature_names[i] for i in keep_indices]


# Step 7: Rename mapping for cleaner axis labels
shap_values.feature_names = [rename_map.get(name, name) for name in shap_values.feature_names]

# Step 8: Plot
fig = plt.figure(figsize=(20, 20))
shap.plots.beeswarm(shap_values, max_display=len(shap_values.feature_names), show=False)
plt.tight_layout(pad=2)
plt.show()

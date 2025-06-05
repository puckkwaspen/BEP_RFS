import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from itertools import product
import pickle
import joblib
from collections import defaultdict

all_results = {}
all_logs = []
all_predictions = {}  # structure: {dataset: {model_name: y_pred}}

# Config dict: each entry maps to a dataset pair
DATASETS = {
    "raw": {
        "train": "Data/BEP_imputed.csv",
        "test": "Data/BEP_imputed_TEST.csv"
    },
    "delta": {
        "train": "Data/BEP_imputed_delta.csv",
        "test": "Data/BEP_imputed_delta_TEST.csv"
    },
    "pct_change": {
        "train": "Data/BEP_imputed_percentage_change.csv",
        "test": "Data/BEP_imputed_percentage_change_TEST.csv"
    }
}

# Parameter grids for each model (for the hyperparameter tuning)
PARAM_GRIDS = {
    "isolation_forest": {
        "n_estimators": [50, 100],
        "contamination": [0.2, 0.4],
        "max_samples": ['auto', 0.8]
    },
    "one_class_svm": {
        "kernel": ['rbf'],
        "nu": [0.1, 0.2],
        "gamma": [0.01, 0.1]
    },
    "lof": {
        "n_neighbors": [10, 20],
        "contamination": [0.01, 0.05]
    }
}

def load_and_preprocess(train_path, test_path):
    # Columns to exclude from modeling
    exclude_cols = ['DATE', 'SEQUENCE', 'ADMISSION', 'INTAKE_ID', 'PATIENT_ID', 'DAYS_SINCE_ADMISSION']

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Extract binary RFS labels
    y_test_true_binary = (test_df['RFS'].values == 1).astype(int)

    # Drop excluded columns + RFS label for features
    X_train = train_df.drop(columns=exclude_cols, errors='ignore')
    X_test = test_df.drop(columns=exclude_cols + ['RFS'], errors='ignore')

    # Keep only non-constant columns in test set
    non_constant_cols = X_test.loc[:, X_test.std() > 1e-8].columns

    # Apply same column selection to training set
    X_test = X_test[non_constant_cols]
    X_train = X_train[non_constant_cols]

    return X_train, X_test, y_test_true_binary


def evaluate_model(model, X_train, X_test, y_true, model_name="Model", is_lof=False, return_preds=False):
    if is_lof:
        X_all = np.vstack([X_train, X_test])
        y_pred_all = model.fit_predict(X_all)
        y_train_pred = y_pred_all[:len(X_train)]
        y_test_pred = y_pred_all[len(X_train):]
    else:
        model.fit(X_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    y_test_pred_binary = (y_test_pred == -1).astype(int)

    precision = precision_score(y_true, y_test_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_test_pred_binary, zero_division=0)
    accuracy = accuracy_score(y_true, y_test_pred_binary)
    f2 = fbeta_score(y_true, y_test_pred_binary, beta = 2, zero_division=0)

    anomalies_train = np.sum(y_train_pred == -1)
    anomalies_test = np.sum(y_test_pred == -1)
    true_anomalies = np.sum(y_true == 1)

    log = (
        f"\n🔍 {model_name}\n"
        f"Anomalies in training set: {anomalies_train}\n"
        f"Anomalies in test set:    {anomalies_test}\n"
        f"True anomalies in test:   {true_anomalies}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"F2 Score:  {f2:.4f}\n"
    )

    if return_preds:
        return precision, recall, accuracy, f2, log, y_test_pred, anomalies_train, anomalies_test, true_anomalies

    return (precision, recall, accuracy, f2, log, anomalies_train, anomalies_test, true_anomalies)



#############################################################################################################

all_model_metrics = defaultdict(lambda: defaultdict(list))  # config_name -> model_name -> list of metric dicts

for config_name, paths in DATASETS.items():
    X_train_scaled, X_test_scaled, y_test_true_binary = load_and_preprocess(paths['train'], paths['test'])

    best_f2 = 0
    best_model_info = {}

    all_predictions.setdefault(config_name, {})
    # --- Isolation Forest ---
    for n_estimators, contamination, max_samples in product(
            PARAM_GRIDS["isolation_forest"]["n_estimators"],
            PARAM_GRIDS["isolation_forest"]["contamination"],
            PARAM_GRIDS["isolation_forest"]["max_samples"]):

        iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                              max_samples=max_samples, random_state=42)

        iso.fit(X_train_scaled)
        if not hasattr(iso, "n_features_"):
            iso.n_features_ = X_train_scaled.shape[1]
        # Save the corresponding scaled data
        # x_test_filename = f"Pickle/X_test_scaled_{config_name}.pkl"
        # joblib.dump(X_test_scaled, x_test_filename)
        # x_train_filename = f"Pickle/X_train_scaled_{config_name}.pkl"
        # joblib.dump(X_train_scaled, x_train_filename)

        precision, recall, accuracy, f2, log, y_preds, anomalies_train, anomalies_test, true_anomalies = evaluate_model(
            iso, X_train_scaled, X_test_scaled, y_test_true_binary,
            model_name=f"Isolation Forest ({config_name})",
            return_preds=True
        )

        # Format log with config and hyperparameters
        params = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": max_samples
        }
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        full_log = (
            f"📦 Configuration: {config_name.upper()}\n"
            f"{log}"
            f"Hyperparameters:\n{param_str}\n"
        )
        all_logs.append(full_log)

        model_id = f"Isolation Forest (n_estimators={n_estimators}, contamination={contamination}, max_samples={max_samples})"
        all_predictions[config_name][model_id] = y_preds.tolist()

        metrics = {
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }
        all_model_metrics[config_name][model_id.split(' ')[0]].append(metrics)

        if f2 > best_f2:
            best_f2 = f2
            best_model_info = {
                "model": "Isolation Forest",
                "anomalies_train": anomalies_train,
                "anomalies_test": anomalies_test,
                "true_anomalies": true_anomalies,
                "f2": f2,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "params": params
            }

            # model_filename = f"Pickle/model_{config_name}_IF.pkl"
            # joblib.dump(iso, model_filename)

    all_predictions.setdefault(config_name, {})
    # --- One-Class SVM ---
    for kernel, nu, gamma in product(
            PARAM_GRIDS["one_class_svm"]["kernel"],
            PARAM_GRIDS["one_class_svm"]["nu"],
            PARAM_GRIDS["one_class_svm"]["gamma"]):

        svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

        precision, recall, accuracy, f2, log, y_preds, anomalies_train, anomalies_test, true_anomalies = evaluate_model(
            svm, X_train_scaled, X_test_scaled, y_test_true_binary,
            model_name=f"One-Class SVM ({config_name})",
            return_preds = True
        )

        params = {
            "kernel": kernel,
            "nu": nu,
            "gamma": gamma
        }
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        full_log = (
            f"📦 Configuration: {config_name.upper()}\n"
            f"{log}"
            f"Hyperparameters:\n{param_str}\n"
        )
        all_logs.append(full_log)

        model_id = f"One-Class SVM (kernel={kernel}, nu={nu}, gamma={gamma})"
        all_predictions[config_name][model_id] = y_preds.tolist()

        metrics = {
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }
        all_model_metrics[config_name][model_id.split(' ')[0]].append(metrics)

    if f2 > best_f2:
            best_f2 = f2
            best_model_info = {
                "model": "Isolation Forest",
                "anomalies_train": anomalies_train,
                "anomalies_test": anomalies_test,
                "true_anomalies": true_anomalies,
                "f2": f2,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "params": params
            }

    all_predictions.setdefault(config_name, {})
    # --- LOF ---
    for n_neighbors, contamination in product(
            PARAM_GRIDS["lof"]["n_neighbors"],
            PARAM_GRIDS["lof"]["contamination"]):

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

        precision, recall, accuracy, f2, log, y_preds, anomalies_train, anomalies_test, true_anomalies = evaluate_model(
            lof, X_train_scaled, X_test_scaled, y_test_true_binary,
            model_name=f"LOF ({config_name})", is_lof=True,
            return_preds = True
        )

        params = {
            "n_neighbors": n_neighbors,
            "contamination": contamination
        }
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        full_log = (
            f"📦 Configuration: {config_name.upper()}\n"
            f"{log}"
            f"Hyperparameters:\n{param_str}\n"
        )
        all_logs.append(full_log)

        model_id = f"Local Outlier Factor (n_neighbors={n_neighbors}, contamination={contamination})"
        all_predictions[config_name][model_id] = y_preds.tolist()

        metrics = {
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }
        all_model_metrics[config_name][model_id.split(' ')[0]].append(metrics)

        if f2 > best_f2:
            best_f2 = f2
            best_model_info = {
                "model": "Isolation Forest",
                "anomalies_train": anomalies_train,
                "anomalies_test": anomalies_test,
                "true_anomalies": true_anomalies,
                "f2": f2,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "params": params
            }

    all_results[config_name] = best_model_info


# --- Save to files ---
with open("Model Results/best_model_summary.txt", "w", encoding="utf-8") as f:
    for config, result in all_results.items():
        f.write(f"\n📦 Configuration: {config.upper()}\n")
        f.write(f"🏆 Best Model: {result['model']}\n")
        f.write(f"🔍 Anomalies in training set: {result['anomalies_train']}\n")
        f.write(f"🔍 Anomalies in test set:    {result['anomalies_test']}\n")
        f.write(f"🔍 True anomalies in test:   {result['true_anomalies']}\n")
        f.write(f"F2 Score:       {result['f2']:.4f}\n")
        f.write(f"Precision:      {result['precision']:.4f}\n")
        f.write(f"Recall:         {result['recall']:.4f}\n")
        f.write(f"Accuracy:       {result['accuracy']:.4f}\n")
        f.write("🔧 Hyperparameters:\n")
        for k, v in result['params'].items():
            f.write(f"  - {k}: {v}\n")
        f.write("-" * 50 + "\n")

# General model summary (time-aware name)
with open("Model Results/model_summary_time.txt", "w", encoding="utf-8") as f:
    for log in all_logs:
        f.write(log)
        f.write("-" * 60 + "\n")

with open("Model Results/model_summary_time.txt", "a", encoding="utf-8") as f:
    f.write("\n=== Best Metrics per Model Type (by Configuration) ===\n")
    for config, model_results in all_model_metrics.items():
        f.write(f"\n📦 Configuration: {config.upper()}\n")
        best_per_model = {}

        for model_name, metrics_list in model_results.items():
            if not metrics_list:
                continue

            # Find the metric entry with the highest F2 score
            best_metrics = max(metrics_list, key=lambda x: x["f2"])
            best_per_model[model_name] = best_metrics

            f.write(f"🔧 {model_name} (best of {len(metrics_list)} runs):\n")
            f.write(f"  - F2 Score:  {best_metrics['f2']:.4f}\n")
            f.write(f"  - Precision: {best_metrics['precision']:.4f}\n")
            f.write(f"  - Recall:    {best_metrics['recall']:.4f}\n")
            f.write(f"  - Accuracy:  {best_metrics['accuracy']:.4f}\n")

        # Optional: delta or % change vs best model
        best_model_name = all_results[config]["model"]
        best_f2 = all_results[config]["f2"]
        f.write("\n📊 Comparison to Best Model:\n")
        for model_name, best in best_per_model.items():
            delta = best["f2"] - best_f2
            pct_change = 100 * delta / best_f2 if best_f2 else 0
            f.write(f"  - {model_name}: ΔF2 = {delta:.4f}, %Δ = {pct_change:.2f}%\n")
        f.write("-" * 60 + "\n")





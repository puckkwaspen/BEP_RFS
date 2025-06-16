import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from itertools import product
from collections import defaultdict
import warnings
import pickle
import random

random.seed(42)

# ignore warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

all_results = {}
all_logs = []
all_predictions = {}  # structure: {dataset: {model_name: y_pred}}
all_model_objects = defaultdict(list)

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
        "n_estimators": [50, 100, 200],
        "contamination": [0.3, 0.2, 0.1],
        "max_samples": ['auto', 0.8, 0.5]
    },
    "one_class_svm": {
    "kernel": ['rbf', 'sigmoid', 'poly'],
    "nu": [0.01, 0.05, 0.1, 0.2],
    "gamma": ['scale', 0.001, 0.01, 0.1, 1]
    },
    "lof": {
    "n_neighbors": [5, 10, 20, 30, 50],     # smaller n_neighbors = more local sensitivity
    "contamination": [0.05, 0.1, 0.15, 0.2],
    "metric": ['minkowski', 'euclidean', 'manhattan']  # distance metrics
    }
}

def load_and_preprocess(train_path, test_path):
    """
    Loads and preprocesses training and test datasets for modeling.

    This function performs the following steps:
    - Loads CSV files from the provided paths.
    - Extracts binary labels for the RFS (Relapse-Free Survival) outcome from the test set.
    - Drops metadata and label columns that should be excluded from modeling.
    - Removes columns with near-zero variance from the test set and applies the same filtering to the training set.

    Parameters:
    ----------
    train_path : str
        Path to the CSV file containing the training data.
    test_path : str
        Path to the CSV file containing the test data.

    Returns:
    -------
    X_train : pd.DataFrame
        Preprocessed feature matrix for training.
    X_test : pd.DataFrame
        Preprocessed feature matrix for testing, aligned with training features.
    y_test_true_binary : np.ndarray
        Binary RFS labels for the test set (1 if RFS == 1, else 0).
    """
    # Columns to exclude from modeling
    exclude_cols = ['DATE', 'SEQUENCE', 'ADMISSION', 'INTAKE_ID', 'PATIENT_ID', 'DAYS_SINCE_ADMISSION']

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Extract binary RFS labels
    y_test_true_binary = (test_df['RFS'].values == 1).astype(int)

    # Drop excluded columns + RFS label for features
    X_train = train_df.drop(columns=exclude_cols, errors='ignore')
    X_test = test_df.drop(columns=exclude_cols + ['RFS', 'CONTROL'], errors='ignore')

    # Keep only non-constant columns in test set
    non_constant_cols = X_test.loc[:, X_test.std() > 1e-8].columns

    # Apply same column selection to training set
    X_test = X_test[non_constant_cols]
    X_train = X_train[non_constant_cols]

    return X_train, X_test, y_test_true_binary


def evaluate_model(model, X_train, X_test, y_true, model_name="Model", is_lof=False, return_preds=False):
    """
    Fits an anomaly detection model, generates predictions, and evaluates its performance.

    This function supports models such as scikit-learn's anomaly detection estimators (e.g., IsolationForest, OneClassSVM).
    Predictions are mapped to binary anomaly labels (1 = anomaly, 0 = normal) based on whether `predict()` returns -1.

    Metrics reported:
    - Precision, Recall, Accuracy, and F2 Score (with beta=2)
    - Count of predicted anomalies in training and test sets
    - Count of true anomalies (based on `y_true`)

    Parameters:
    ----------
    model : object
        A fitted anomaly detection model with `.fit()` and `.predict()` methods.
    X_train : pd.DataFrame or np.ndarray
        Feature matrix for training the model.
    X_test : pd.DataFrame or np.ndarray
        Feature matrix for testing the model.
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels for the test set (1 = true anomaly, 0 = normal).
    model_name : str, optional (default="Model")
        Name used for logging and print statements.
    is_lof : bool, optional (default=False)
        Reserved for future use (e.g., special handling for Local Outlier Factor).
    return_preds : bool, optional (default=False)
        If True, returns raw predictions and additional diagnostic values.

    Returns:
    -------
    tuple
        If return_preds is False:
            (precision, recall, accuracy, f2, log, anomalies_train, anomalies_test, true_anomalies)
        If return_preds is True:
            (precision, recall, accuracy, f2, log, y_test_pred, anomalies_train, anomalies_test, true_anomalies)
    """
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

# loops through all combinations and evaluates the models, the results are saved to txt files and pickle files
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
        all_model_objects[config_name].append((model_id, iso))
        all_predictions[config_name][model_id] = y_preds.tolist()

        metrics = {
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "params": params
        }
        all_model_metrics[config_name][model_id.split(' ')[0]].append(metrics)

        if f2 > best_f2:
            best_f2 = f2
            best_model_info = {
                "model": "Isolation Forest",
                "model_object": iso,
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
        all_model_objects[config_name].append((model_id, svm))
        all_predictions[config_name][model_id] = y_preds.tolist()

        metrics = {
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "params": params
        }
        all_model_metrics[config_name][model_id.split(' ')[0]].append(metrics)

        if f2 > best_f2:
                best_f2 = f2
                best_model_info = {
                    "model": "One-Class SVM",
                    "model_object": svm,
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

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty = True)

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
        all_model_objects[config_name].append((model_id, lof))
        all_predictions[config_name][model_id] = y_preds.tolist()

        metrics = {
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "params": params
        }
        all_model_metrics[config_name][model_id.split(' ')[0]].append(metrics)

        if f2 > best_f2:
            best_f2 = f2
            best_model_info = {
                "model": "Local Outlier Factor",
                "model_object": lof,
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

def recursively_convert(d):
    if isinstance(d, defaultdict):
        d = {k: recursively_convert(v) for k, v in d.items()}
    return d

all_model_metrics_serializable = recursively_convert(all_model_metrics)

# Save results, logs, and metrics
with open("PickleFiles/all_predictions.pkl", "wb") as f:
    pickle.dump(all_predictions, f)

with open("PickleFiles/all_model_metrics.pkl", "wb") as f:
    pickle.dump(all_model_metrics_serializable, f)

with open("PickleFiles/all_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

with open("PickleFiles/all_logs.pkl", "wb") as f:
    pickle.dump(all_logs, f)

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

# --- Save best model objects as pickle files ---
for config, result in all_results.items():
    model_obj = result.get("model_object", None)
    if model_obj:
        filename = f"PickleFiles/best_model_{config}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model_obj, f)

# General model summary (time-aware name)
with open("Model Results/model_summary.txt", "w", encoding="utf-8") as f:
    for log in all_logs:
        f.write(log)
        f.write("-" * 60 + "\n")

with open("Model Results/model_summary.txt", "a", encoding="utf-8") as f:
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

            if "params" in best_metrics:
                f.write("  - Hyperparameters:\n")
                for k, v in best_metrics["params"].items():
                    f.write(f"      • {k}: {v}\n")

        f.write("-" * 60 + "\n")

model_dir = "PickleFiles/AllModels"

# Clear old pickle files
for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        os.remove(os.path.join(model_dir, file))

for config, models in all_model_objects.items():
    for model_id, model_obj in models:
        # Clean model ID for filename
        safe_model_id = model_id.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "-")
        filename = f"PickleFiles/AllModels/model_{config}_{safe_model_id}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model_obj, f)








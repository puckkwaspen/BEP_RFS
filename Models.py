import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from itertools import product

all_results = {}
all_logs = []

# Config dict: each entry maps to a dataset pair
DATASETS = {
    "raw": {
        "train": "BEP_imputed.csv",
        "test": "BEP_imputed_TEST.csv"
    },
    "delta": {
        "train": "BEP_imputed_delta.csv",
        "test": "BEP_imputed_delta_TEST.csv"
    },
    "pct_change": {
        "train": "BEP_imputed_percentage_change.csv",
        "test": "BEP_imputed_percentage_change_TEST.csv"
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
        "kernel": ['rbf', 'rbf'],
        "nu": [0.1, 0.2],
        "gamma": [0.01, 0.1]
    },
    "lof": {
        "n_neighbors": [10, 20],
        "contamination": [0.01, 0.05]
    }
}

def load_and_preprocess(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_test_true_binary = (test_df['RFS'].values == 1).astype(int)
    X_test = test_df.drop(columns=['RFS'], errors='ignore')
    X_train = train_df

    # Keep only non-constant columns in test set (std > very small value)
    # So e.g. SEX is dropped if only females in test set
    non_constant_cols = X_test.loc[:, X_test.std() > 1e-8].columns
    # Apply same column selection to training set
    X_test = X_test[non_constant_cols]
    X_train = X_train[non_constant_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_test_true_binary


def evaluate_model(model, X_train, X_test, y_true, model_name="Model", is_lof=False):
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
    f1 = f1_score(y_true, y_test_pred_binary, zero_division=0)

    log = (
        f"\n🔍 {model_name}\n"
        f"Anomalies in training set: {np.sum(y_train_pred == -1)}\n"
        f"Anomalies in test set:    {np.sum(y_test_pred == -1)}\n"
        f"True anomalies in test:   {np.sum(y_true == 1)}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
    )

    return precision, recall, accuracy, f1, log



#############################################################################################################


for config_name, paths in DATASETS.items():
    X_train_scaled, X_test_scaled, y_test_true_binary = load_and_preprocess(paths['train'], paths['test'])

    best_f1 = 0
    best_model_info = {}

    # --- Isolation Forest ---
    for n_estimators, contamination, max_samples in product(
            PARAM_GRIDS["isolation_forest"]["n_estimators"],
            PARAM_GRIDS["isolation_forest"]["contamination"],
            PARAM_GRIDS["isolation_forest"]["max_samples"]):

        iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                              max_samples=max_samples, random_state=42)

        precision, recall, accuracy, f1, log = evaluate_model(
            iso, X_train_scaled, X_test_scaled, y_test_true_binary,
            model_name=f"Isolation Forest ({config_name})"
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

        if f1 > best_f1:
            best_f1 = f1
            best_model_info = {
                "model": "Isolation Forest",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "params": params
            }

    # --- One-Class SVM ---
    for kernel, nu, gamma in product(
            PARAM_GRIDS["one_class_svm"]["kernel"],
            PARAM_GRIDS["one_class_svm"]["nu"],
            PARAM_GRIDS["one_class_svm"]["gamma"]):

        svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

        precision, recall, accuracy, f1, log = evaluate_model(
            svm, X_train_scaled, X_test_scaled, y_test_true_binary,
            model_name=f"One-Class SVM ({config_name})"
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

        if f1 > best_f1:
            best_f1 = f1
            best_model_info = {
                "model": "Isolation Forest",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "params": params
            }

    # --- LOF ---
    for n_neighbors, contamination in product(
            PARAM_GRIDS["lof"]["n_neighbors"],
            PARAM_GRIDS["lof"]["contamination"]):

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

        precision, recall, accuracy, f1, log = evaluate_model(
            lof, X_train_scaled, X_test_scaled, y_test_true_binary,
            model_name=f"LOF ({config_name})", is_lof=True
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

        if f1 > best_f1:
            best_f1 = f1
            best_model_info = {
                "model": "Isolation Forest",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "params": params
            }

    all_results[config_name] = best_model_info


# --- Save to files ---
with open("best_model_summary.txt", "w", encoding="utf-8") as f:    # remember to use encoding to allow emojis :)
    for config, result in all_results.items():
        f.write(f"\n📦 Configuration: {config.upper()}\n")
        f.write(f"🏆 Best Model: {result['model']}\n")
        f.write(f"F1 Score:  {result['f1']:.4f}\n")
        f.write(f"Precision: {result['precision']:.4f}\n")
        f.write(f"Recall:    {result['recall']:.4f}\n")
        f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
        f.write("🔧 Hyperparameters:\n")
        for k, v in result['params'].items():
            f.write(f"  - {k}: {v}\n")
        f.write("-" * 50 + "\n")

with open("model_summary.txt", "w", encoding="utf-8") as f:
    for log in all_logs:
        f.write(log)
        f.write("-" * 60 + "\n")


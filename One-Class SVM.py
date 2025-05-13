import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score
from itertools import product

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
        "contamination": [0.01, 0.05],
        "max_samples": ['auto', 0.8]
    },
    "one_class_svm": {
        "kernel": ['rbf', 'linear'],
        "nu": [0.005, 0.1],
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_test_true_binary


def evaluate_model(model, X_train, X_test, y_true, model_name="Model", is_lof=False):
    if is_lof:
        # Combine train + test for LOF
        X_all = np.vstack([X_train, X_test])
        y_pred_all = model.fit_predict(X_all)
        y_train_pred = y_pred_all[:len(X_train)]
        y_test_pred = y_pred_all[len(X_train):]
    else:
        model.fit(X_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # Convert predictions to binary
    y_test_pred_binary = (y_test_pred == -1).astype(int)

    # Compute metrics
    precision = precision_score(y_true, y_test_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_test_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_test_pred_binary, zero_division=0)

    # Output
    print(f"🧮 Showing results for {model_name}:")
    print(f"Anomalies identified in training set: {np.sum(y_train_pred == -1)}")
    print(f"Anomalies identified in test set:    {np.sum(y_test_pred == -1)}")
    print(f"True anomalies in test set (according to ASPEN): {np.sum(y_true == 1)}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall:    {recall:.4f}")
    print(f"🏆 F1 Score:  {f1:.4f}\n")

    return precision, recall, f1

# Run all the configurations and print results
for config_name, paths in DATASETS.items():
    print(f"\n🚀 Running config: {config_name.upper()}\n")
    X_train_scaled, X_test_scaled, y_test_true_binary = load_and_preprocess(paths['train'], paths['test'])

    # --- Isolation Forest Grid Search ---
    best_f1_iso = 0
    for n_estimators, contamination, max_samples in product(
            PARAM_GRIDS["isolation_forest"]["n_estimators"],
            PARAM_GRIDS["isolation_forest"]["contamination"],
            PARAM_GRIDS["isolation_forest"]["max_samples"]):

        iso = IsolationForest(n_estimators=n_estimators,
                              contamination=contamination,
                              max_samples=max_samples,
                              random_state=42)

        _, _, f1 = evaluate_model(iso, X_train_scaled, X_test_scaled, y_test_true_binary,
                                  f"Isolation Forest ({config_name}) [n_estimators={n_estimators}, contamination={contamination}, max_samples={max_samples}]")

        if f1 > best_f1_iso:
            best_f1_iso = f1

    # --- One-Class SVM Grid Search ---
    best_f1_svm = 0
    for kernel, nu, gamma in product(PARAM_GRIDS["one_class_svm"]["kernel"],
                                        PARAM_GRIDS["one_class_svm"]["nu"],
                                        PARAM_GRIDS["one_class_svm"]["gamma"]):

        svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

        _, _, f1 = evaluate_model(svm, X_train_scaled, X_test_scaled, y_test_true_binary,
                                  f"One-Class SVM ({config_name}) [kernel = {kernel}, nu={nu}, gamma={gamma}]")

        if f1 > best_f1_svm:
            best_f1_svm = f1

    # --- LOF Grid Search ---
    best_f1_lof = 0
    for n_neighbors, contamination in product(PARAM_GRIDS["lof"]["n_neighbors"],
                                              PARAM_GRIDS["lof"]["contamination"]):

        lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                                 contamination=contamination)

        _, _, f1 = evaluate_model(lof, X_train_scaled, X_test_scaled, y_test_true_binary,
                                  f"LOF ({config_name}) [n_neighbors={n_neighbors}, contamination={contamination}]",
                                  is_lof=True)

        if f1 > best_f1_lof:
            best_f1_lof = f1

    print(f"✅ Best F1 (Isolation Forest - {config_name}): {best_f1_iso:.4f}")
    print(f"✅ Best F1 (One-Class SVM - {config_name}):   {best_f1_svm:.4f}")
    print(f"✅ Best F1 (LOF - {config_name}):              {best_f1_lof:.4f}")
    print("-----------------------------------------------------------------------------------\n")


#     # Apply your models (reusing `evaluate_model()` from before)
#     from sklearn.ensemble import IsolationForest
#     from sklearn.svm import OneClassSVM
#     from sklearn.neighbors import LocalOutlierFactor
#
#     iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
#     evaluate_model(iso, X_train_scaled, X_test_scaled, y_test_true_binary, f"Isolation Forest ({config_name})")
#
#     svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
#     evaluate_model(svm, X_train_scaled, X_test_scaled, y_test_true_binary, f"One-Class SVM ({config_name})")
#
#     lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
#     evaluate_model(lof, X_train_scaled, X_test_scaled, y_test_true_binary, f"LOF ({config_name})", is_lof=True)
#
#     print("-----------------------------------------------------------------------------------\n")

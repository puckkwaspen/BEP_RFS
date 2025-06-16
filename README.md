# 📝 BEP Anomaly Detection

Anomaly detection in clinical data of patients with **Anorexia Nervosa**, using unsupervised learning techniques.

---

## 📚 Overview

This project investigates the use of unsupervised machine learning to detect anomalies in longitudinal clinical data from patients with Anorexia Nervosa. The goal is to support identification of critical phases such as refeeding or medical destabilization.

---

### 🗂 Key Information

- **Paper**: _[Detecting Anomalies in Clinical Data Using Unsupervised Learning: A Study on Anorexia Nervosa Patients](#)_  
- **Author**: _[Puck Kwaspen](#)_

---

### 📊 Highlights

- ⚔️ **Robust Model Comparison**  
  Comprehensive benchmarking of Isolation Forest, One-Class SVM, and Local Outlier Factor using custom configurations and hyperparameter tuning.

- 🧠 **Clinical Focus**  
  Tailored to the specific challenges of anomaly detection in sparse and heterogeneous clinical datasets.

- 🔄 **Reproducible Pipelines**  
  End-to-end preprocessing, model training, evaluation, and result exploration pipelines.

- 🚀 **Explainable Outputs**  
  Model interpretability via SHAP analysis and intuitive visualizations.

---

## 🚀 Getting Started

Follow these steps to set up the environment and run the project locally.

### 🔧 Requirements

1. **Python Version**  
   Make sure you are using **Python 3.9**.

2. **Install Dependencies**  
   Run the following command in your terminal to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Optional – Virtual Environment Setup**

   This step is only needed if you experience issues installing or using `miceforest` (e.g., due to version conflicts with `numpy` or `pandas`).

   You can use the provided batch scripts to automatically configure your environment:

   - `setup_env.bat`  
     Creates and activates a virtual environment, then installs all dependencies.
   - `repair_miceforest.bat`  
     Fixes known compatibility issues with `miceforest`, `numpy`, and `pandas`.

   > 💡 Double-click the `.bat` files in Windows Explorer to use them.

---

## 📂 File Structure

### 📄 Data Files

The following files are required for the data loading and preprocessing functions.  
Please ensure that the files remain in their respective folders and that names do not change (of both files and folders). If they do, you can either change the names back or edit the paths in the respective code files.
- `annonymizedDatasets/` for training data
- `Puck/` for RFS test data
- `Daisy/` for control test data

**Training Data:**
- `annonymizedDatasets/maskedDAIsy_LabCombinedNew.csv`
- `annonymizedDatasets/maskedDAIsy_Vitals.csv`
- `annonymizedDatasets/maskedDAIsy_AllDatasetsCombinedWoRepIntakes_v1.tsv`

**Testing Data:**  
RFS Group:
- `Puck/anonymized_Labels_refeeding.csv`
- `Puck/anonymized_Labels_refeeding_lab.csv`
- `Puck/anonymized_Labels_refeeding_metingen.csv`

Control Group: 
- `Daisy/Daisy_lab_part1.csv`
- `Daisy/Daisy_lab_part2.csv`
- `Daisy/Daisy_metingen_part1.csv`
- `Daisy/Daisy_metingen_part2.csv`
- `Daisy/Daisy_main.csv`

---

### 📄 Preprocessing Files

- `Preprocessing.py` – Script for preprocessing the **training data**.
- `Preprocessing Test Data 1.py` – Script for preprocessing the **RFS group** of the test data.
- `Preprocessing Test Data 2.py` – Script for preprocessing the **control group** of the test data.

**Development Notebooks (for reference only):**
- `Preparing Data.ipynb` – Development notebook for training data preprocessing.
- `Test Data Preprocessing.ipynb` – Development notebook for test data preprocessing.

---

### 📄 Factor Analysis

- `Factor Analysis.py` – Script for performing factor analysis and generating related visualizations used in the thesis.

---

### 📄 Model Training and Exploration

- `Models.py` – Main script for training and evaluating all models.  
  Outputs are saved in:

  - `ModelResults/` – Contains summary logs and evaluation results as `.txt` files:
    - `best_model_summary.txt`: Best model per configuration with metrics and hyperparameters.
    - `model_summary_time.txt`: Full logs for all configurations and models.

  - `PickleFiles/` – Contains serialized results and objects saved as `.pkl` files, including:
    - Trained models
    - Evaluation metrics
    - Log dictionaries for each model run  
    These allow for full reproducibility and later inspection without re-training.

**Result Exploration:**
- `IFOREST Explore.py` – Script for exploring Isolation Forest results (visuals and output for the thesis).
- `OCSVM Explore.py` – Script for exploring One-Class SVM results (visuals and output for the thesis).
- `LOF Explore.py` – Script for exploring Local Outlier Factor results (visuals and output for the thesis).

---

### 📄 Accessing Pickled Results Without Re-Training

- `pickles.py` – Script that demonstrates how to:
  - Load trained models and associated artifacts from the `PickleFiles/` folder
  - Access saved logs and performance metrics
  - Reuse prediction outputs

---

## 💡 How to Run

### 🧹 Preprocessing
1. Run `Preprocessing.py`, `Preprocessing Test Data 1.py`, and `Preprocessing Test Data 2.py` in this exact order.
2. This generates cleaned CSV files inside the `Data/` folder, which are used for model training.

### 📈 Factor Analysis
1. Run `Factor Analysis.py`.
2. The script will output plots and factor-based analysis results.

### 📈 Factor Analysis
1. Run `Factor Analysis.py`.
2. The script will generate plots and factor-based analysis results used in the thesis.

### 🤖 Model Training
1. Run `Models.py`.
2. Outputs are saved in the following locations:
   - `ModelResults/` folder:
     - `best_model_summary.txt`: Shows the best model per configuration with key metrics and hyperparameters.
     - `model_summary_time.txt`: Detailed logs for all configurations, including per-model summaries.
   - `PickleFiles/` folder:
     - All trained models, performance metrics, and logs are saved as `.pkl` files for reproducibility and later analysis.

### 🔍 Results Exploration
Run the following files to explore model-specific outputs (e.g. SHAP, metrics, and visualizations):
- `IFOREST Explore.py`
- `OCSVM Explore.py`
- `LOF Explore.py`

To manually explore saved models or logs, use the utility script:
- `pickles.py` – A helper script that shows how to load and inspect `.pkl` files without retraining models.

This script load precomputed results directly from the `PickleFiles/` folder, including:
- Trained models
- Prediction results
- Evaluation metrics
- Log dictionaries


---

## 🛠️ Extending the Project
- You can upgrade to **Python 3.10+** to use the latest version of the `miceforest` package.

---

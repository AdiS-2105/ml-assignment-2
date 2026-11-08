"""
QSAR Biodegradation Classification - Model Training and Evaluation Pipeline
Course: Machine Learning (AIMLZG565), BITS Pilani WILP M.Tech (AIML)
Author: Adithya S
Random Seed: 2002

This script implements an end-to-end reproducible machine learning pipeline:
1. Acquires and cleans the UCI QSAR Biodegradation dataset (41 chemical descriptors).
2. Performs an 80:20 stratified train-test split using fixed RANDOM_SEED.
3. Fits a StandardScaler preprocessor exclusively on training data to prevent leakage.
4. Trains 5 diverse classification models:
   - Logistic Regression (Linear baseline with L2 regularization)
   - Decision Tree Classifier (Cost-effective recursive partitioning)
   - K-Nearest Neighbors (Instance-based metric learning)
   - Gaussian Naive Bayes (Probabilistic generative classifier)
   - Support Vector Machine (Max-margin non-linear RBF kernel)
5. Computes all 6 evaluation metrics:
   - Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC (Matthews Correlation Coefficient).
6. Serializes trained artifacts and metrics summary to the model/ directory.
"""

import os
import json
import warnings
import urllib.request
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import joblib

# Suppress benign scikit-learn warnings
warnings.filterwarnings("ignore")

# Fixed reproducibility seed
RANDOM_SEED = 2002

# Official DRAGON Molecular Descriptor Names for UCI QSAR Biodegradation
MOLECULAR_DESCRIPTOR_COLUMNS = [
    'SpMax_L', 'J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-', 'C%',
    'nCp', 'nO', 'F03[C-N]', 'SdssC', 'HyWi_B(m)', 'LOC', 'SM6_L', 'F03[C-O]',
    'Me', 'Mi', 'nN-N', 'nArNO2', 'nCRX3', 'SpPosA_B(p)', 'nCIR', 'B01[C-Br]',
    'B03[C-Cl]', 'N-073', 'SpMax_A', 'Psi_i_1d', 'B04[C-Br]', 'SdO', 'TI2_L',
    'nCrt', 'C-026', 'F02[C-N]', 'nHDon', 'SpMax_B(m)', 'Psi_i_A', 'nN',
    'SM6_B(m)', 'nArCOOR', 'nX'
]

TARGET_COLUMN = 'ReadyBiodegradable'
UCI_DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
ARTIFACT_DIR = 'model'


def fetch_biodegradation_dataset(source_url: str = UCI_DATASET_URL) -> pd.DataFrame:
    """
    Downloads and structures the UCI QSAR Biodegradation dataset.
    Assigns the 41 molecular descriptor column headers and binarizes the target
    (1 = Ready Biodegradable 'RB', 0 = Not Ready Biodegradable 'NRB').
    """
    print(f"[*] Fetching QSAR Biodegradation dataset from {source_url}...")
    try:
        raw_dataframe = pd.read_csv(source_url, sep=';', header=None)
    except Exception as exc:
        print(f"[!] Network fetch failed ({exc}), looking for local backup...")
        if os.path.exists('test_data.csv'):
            print("[*] Reconstructing using local data...")
            raw_dataframe = pd.read_csv('test_data.csv')
            return raw_dataframe
        raise RuntimeError(f"Unable to retrieve dataset: {exc}")

    raw_dataframe.columns = MOLECULAR_DESCRIPTOR_COLUMNS + [TARGET_COLUMN]
    
    # Map experimental class label: 'RB' -> 1, 'NRB' -> 0
    if raw_dataframe[TARGET_COLUMN].dtype != int:
        raw_dataframe[TARGET_COLUMN] = (
            raw_dataframe[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .map({'RB': 1, 'NRB': 0})
            .fillna(0)
            .astype(int)
        )

    print(f"[+] Loaded {len(raw_dataframe)} compounds across {len(MOLECULAR_DESCRIPTOR_COLUMNS)} molecular descriptors.")
    class_distribution = raw_dataframe[TARGET_COLUMN].value_counts().to_dict()
    print(f"    Class distribution: NRB (0) = {class_distribution.get(0, 0)}, RB (1) = {class_distribution.get(1, 0)}")
    return raw_dataframe


def partition_chemical_data(
    dataset: pd.DataFrame,
    test_ratio: float = 0.20,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset into 80% training and 20% held-out test sets
    using stratified sampling to maintain class proportions.
    """
    training_set, testing_set = train_test_split(
        dataset,
        test_size=test_ratio,
        random_state=seed,
        stratify=dataset[TARGET_COLUMN]
    )
    print(f"[+] Partitioned data: Train = {training_set.shape[0]} rows, Test = {testing_set.shape[0]} rows (stratified).")
    return training_set, testing_set


def build_and_fit_preprocessor(train_features: pd.DataFrame) -> StandardScaler:
    """
    Fits a StandardScaler strictly on the training partition.
    Ensures zero information leakage from testing data.
    """
    scaler = StandardScaler()
    scaler.fit(train_features)
    print("[+] Fitted StandardScaler on training feature matrix.")
    return scaler


def instantiate_model_suite(seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Instantiates the five required classification algorithms with reproducible hyperparameters.
    """
    classifiers = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=seed
        ),
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=seed
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='uniform',
            metric='minkowski',
            p=2
        ),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(
            C=1.5,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=seed
        )
    }
    return classifiers


def evaluate_model_performance(
    model: Any,
    feature_matrix: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, Any]:
    """
    Computes all 6 classification evaluation metrics:
    1. Accuracy
    2. AUC Score (ROC-AUC)
    3. Precision
    4. Recall
    5. F1 Score
    6. Matthews Correlation Coefficient (MCC Score)
    """
    predictions = model.predict(feature_matrix)
    
    # Calculate class probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(feature_matrix)[:, 1]
    elif hasattr(model, 'decision_function'):
        dec_vals = model.decision_function(feature_matrix)
        probabilities = (dec_vals - dec_vals.min()) / (dec_vals.max() - dec_vals.min() + 1e-9)
    else:
        probabilities = predictions.astype(float)

    conf_matrix = confusion_matrix(ground_truth, predictions)
    
    metrics = {
        'Accuracy': float(np.round(accuracy_score(ground_truth, predictions), 4)),
        'AUC': float(np.round(roc_auc_score(ground_truth, probabilities), 4)),
        'Precision': float(np.round(precision_score(ground_truth, predictions, zero_division=0), 4)),
        'Recall': float(np.round(recall_score(ground_truth, predictions, zero_division=0), 4)),
        'F1': float(np.round(f1_score(ground_truth, predictions, zero_division=0), 4)),
        'MCC': float(np.round(matthews_corrcoef(ground_truth, predictions), 4)),
        'ConfusionMatrix': conf_matrix.tolist()
    }
    return metrics


def execute_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Orchestrates the entire training, evaluation, and artifact serialization workflow.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 1. Fetch data
    df = fetch_biodegradation_dataset()
    
    # 2. Partition data
    train_df, test_df = partition_chemical_data(df, test_ratio=0.20, seed=RANDOM_SEED)
    
    # 3. Export test dataset to project root
    test_df.to_csv('test_data.csv', index=False)
    print("[+] Exported held-out test data to 'test_data.csv'.")
    
    # 4. Separate features and target
    X_train_raw = train_df[MOLECULAR_DESCRIPTOR_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_test_raw = test_df[MOLECULAR_DESCRIPTOR_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values
    
    # 5. Preprocess (scaling)
    scaler = build_and_fit_preprocessor(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Save preprocessor
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'preprocessor.joblib'))
    print(f"[+] Saved fitted scaler to '{os.path.join(ARTIFACT_DIR, 'preprocessor.joblib')}'.")
    
    # 6. Initialize classifiers
    model_suite = instantiate_model_suite(seed=RANDOM_SEED)
    
    # Mapping for artifact file naming
    filename_map = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'KNN': 'knn.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'svm.joblib'
    }
    
    metrics_summary = {}
    table_rows = []
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING & EVALUATION (Held-out Test Set, n=211)")
    print("=" * 80)
    
    for name, model in model_suite.items():
        print(f"[*] Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Serialize model artifact
        artifact_path = os.path.join(ARTIFACT_DIR, filename_map[name])
        joblib.dump(model, artifact_path)
        print(f"    Saved model artifact to '{artifact_path}'")
        
        # Evaluate on held-out test data
        eval_results = evaluate_model_performance(model, X_test_scaled, y_test)
        metrics_summary[name] = eval_results
        
        table_rows.append({
            'ML Model Name': name,
            'Accuracy': f"{eval_results['Accuracy']:.4f}",
            'AUC': f"{eval_results['AUC']:.4f}",
            'Precision': f"{eval_results['Precision']:.4f}",
            'Recall': f"{eval_results['Recall']:.4f}",
            'F1': f"{eval_results['F1']:.4f}",
            'MCC': f"{eval_results['MCC']:.4f}"
        })
    
    # Save metrics summary as JSON
    json_path = os.path.join(ARTIFACT_DIR, 'metrics_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Saved evaluation metrics summary to '{json_path}'.")
    
    summary_df = pd.DataFrame(table_rows)
    print("\n" + "-" * 80)
    print("OFFICIAL BITS EVALUATION METRICS COMPARISON TABLE")
    print("-" * 80)
    
    # Format markdown table cleanly without requiring third-party tabulate
    headers = list(table_rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(str(row[h]) for h in headers) + " |"
        for row in table_rows
    ]
    markdown_table = "\n".join([header_line, separator_line] + row_lines)
    print(markdown_table)
    print("-" * 80)
    
    return summary_df, metrics_summary


if __name__ == '__main__':
    execute_pipeline()

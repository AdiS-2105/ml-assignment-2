"""
BioDeg-Predict: QSAR Molecular Biodegradation Classification Studio
Course: Machine Learning (AIMLZG565), BITS Pilani WILP M.Tech (AIML)
Student id: 2026AJ05162 | Random Seed: 2002

Interactive Streamlit Web Application:
  - Test Dataset Upload (CSV) with Sample Download
  - Model Selection Dropdown & Hyperparameter Box
  - 6 Key Classification Evaluation Metrics Display
  - Confusion Matrix & Classification Report
  - Multi-Model Benchmarking & Performance Analysis
"""

import os
import json
import inspect
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration (Must be first Streamlit command)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioDeg-Predict | AIMLZG565",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Crisp Light Theme CSS (High Contrast, Zero Low-Contrast Text)
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_LIGHT_CSS = """
<style>
/* ── Global Styles ──────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #0F172A !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1350px !important;
}

/* ── Typography & Headings ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
}

p, span, label {
    color: #1E293B;
}

/* ── Top Header Banner Card ─────────────────────────────────────────────── */
.header-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #2563EB;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.header-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 4px 0;
    line-height: 1.25;
}

.header-subtitle {
    font-size: 0.92rem;
    color: #475569;
    margin: 0 0 12px 0;
    font-weight: 500;
}

.meta-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
    margin-top: 10px;
    font-size: 0.82rem;
}

.meta-item {
    display: inline-flex;
    align-items: center;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 4px 10px;
    border-radius: 6px;
    color: #334155;
    font-weight: 500;
}

.meta-item b {
    color: #0F172A;
    margin-right: 4px;
}

.badge-status {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.badge-winner {
    background: #FEF3C7;
    border: 1px solid #FCD34D;
    color: #92400E;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 6px;
}

/* ── Info & Notification Banners ────────────────────────────────────────── */
.info-banner {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #1E293B;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.success-banner {
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #166534;
    margin-bottom: 20px;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
.metric-card-wrapper {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #2563EB;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    height: 100%;
}

.metric-card-wrapper:hover {
    transform: translateY(-2px);
    border-top-color: #1D4ED8;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
}

.metric-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    margin-bottom: 4px;
}

.metric-number {
    font-size: 1.62rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
}

.metric-desc {
    font-size: 0.70rem;
    font-weight: 600;
    color: #64748B;
    margin-top: 4px;
}

/* ── Sidebar Hyperparameter Box ─────────────────────────────────────────── */
.hp-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 8px;
    margin-bottom: 14px;
    font-size: 0.82rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.hp-title {
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.84rem;
}

.hp-row {
    color: #334155;
    margin-bottom: 4px;
    line-height: 1.4;
}

.hp-row b {
    color: #0F172A;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.footer-container {
    text-align: center;
    color: #64748B;
    font-size: 0.80rem;
    padding: 24px 0 8px 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 36px;
    line-height: 1.6;
}

.footer-container a {
    color: #2563EB;
    text-decoration: none;
    font-weight: 600;
}

.footer-container a:hover {
    text-decoration: underline;
}

/* ── Section Dividers & Cards ───────────────────────────────────────────── */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

/* ── Modern Light Native Streamlit Tabs Styling ──────────────────────────── */
div[data-testid="stTabs"] {
    margin-top: 8px;
    margin-bottom: 24px;
}

div[role="tablist"], [data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    gap: 8px !important;
    border-bottom: 2px solid #E2E8F0 !important;
    padding-bottom: 0px !important;
    background: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    margin-bottom: 20px !important;
}

/* Base Tab Button */
div[data-testid="stTab"], button[data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2.5px solid transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 20px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-bottom: -2px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stTab"] p, button[data-baseweb="tab"] p {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    margin: 0 !important;
}

/* Inactive Tab Hover */
div[data-testid="stTab"]:hover, button[data-baseweb="tab"]:hover {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border-bottom-color: #CBD5E1 !important;
}

div[data-testid="stTab"]:hover p, button[data-baseweb="tab"]:hover p {
    color: #0F172A !important;
}

/* Active Selected Tab Indicator */
div[data-testid="stTab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTab"][data-selected="true"] {
    background-color: #EFF6FF !important;
    color: #2563EB !important;
    border-bottom: 2.5px solid #2563EB !important;
    border-radius: 6px 6px 0 0 !important;
    box-shadow: none !important;
}

div[data-testid="stTab"][aria-selected="true"] p,
div[data-testid="stTab"][aria-selected="true"] span,
div[data-testid="stTab"][data-selected="true"] p,
div[data-testid="stTab"][data-selected="true"] span,
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span {
    color: #2563EB !important;
    font-weight: 700 !important;
}

.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"] {
    background-color: #2563EB !important;
}

/* ── Observations Table (Fully Wrapped, No Horizontal Truncation) ───────── */
.obs-table-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-top: 10px;
    margin-bottom: 18px;
}

.obs-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.90rem;
    text-align: left;
    background-color: #FFFFFF;
}

.obs-table th {
    background-color: #F8FAFC;
    color: #0F172A;
    font-weight: 700;
    padding: 14px 18px;
    border-bottom: 2px solid #E2E8F0;
    letter-spacing: -0.01em;
}

.obs-table td {
    padding: 14px 18px;
    vertical-align: top;
    border-bottom: 1px solid #F1F5F9;
    color: #1E293B;
    line-height: 1.65;
    white-space: normal !important;
    word-wrap: break-word !important;
}

.obs-table tr:last-child td {
    border-bottom: none;
}

.obs-table td.model-name {
    font-weight: 700;
    color: #0F172A;
    width: 24%;
    min-width: 180px;
    border-right: 1px solid #F1F5F9;
}

.obs-table td.obs-content {
    width: 76%;
}

.obs-table tr.winner-row {
    background-color: #EFF6FF !important;
    border-left: 4px solid #2563EB;
}

.obs-table tr.winner-row td {
    color: #1E3A8A;
}

.obs-table tr.winner-row td.model-name {
    color: #1D4ED8;
    font-weight: 800;
}
</style>
"""
st.markdown(CUSTOM_LIGHT_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Constants & Metadata
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 2002
MODEL_DIR = "model"
DEFAULT_TEST_CSV = "test_data.csv"
TARGET_COL = "ReadyBiodegradable"

DESCRIPTOR_COLS = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)",
    "nCIR", "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d",
    "B04[C-Br]", "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon",
    "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX"
]

MODEL_SPECS = {
    "Logistic Regression": {
        "filename": "logistic_regression.joblib",
        "color": "#0284C7",
        "category": "Linear Classifier (L2 Regularized)",
        "params": "Solver: lbfgs | Penalty: L2 | C: 1.0 | max_iter: 1000",
        "description": "L2-regularized linear model providing high interpretability and stable coefficients across collinear descriptors."
    },
    "Decision Tree": {
        "filename": "decision_tree.joblib",
        "color": "#D97706",
        "category": "Non-linear Recursive Partitioning",
        "params": "Criterion: Gini | max_depth: 5 | min_samples_split: 10 | min_samples_leaf: 5",
        "description": "Pre-pruned decision tree that performs axis-aligned feature splits to prevent unconstrained variance."
    },
    "KNN": {
        "filename": "knn.joblib",
        "color": "#7C3AED",
        "category": "Instance-based Metric Space Learner",
        "params": "Neighbors (k): 7 | Metric: Minkowski (p=2 Euclidean) | Weights: Uniform",
        "description": "Non-parametric classifier exploiting the chemical structure-activity similarity principle in standardized feature space."
    },
    "Naive Bayes": {
        "filename": "naive_bayes.joblib",
        "color": "#DC2626",
        "category": "Probabilistic Generative Classifier",
        "params": "Type: GaussianNB | Priors: Empirical | Var Smoothing: 1e-9",
        "description": "Fast probabilistic model under class-conditional independence assumption, delivering extreme recall at lower precision."
    },
    "Support Vector Machine": {
        "filename": "svm.joblib",
        "color": "#2563EB",
        "category": "Maximum-Margin Non-Linear Classifier (🏆 Overall Winner)",
        "params": "Kernel: RBF | Regularization (C): 1.5 | Gamma: scale | Probability: True",
        "description": "Non-linear kernel SVM projecting molecular descriptors into Hilbert space to separate complex biochemical boundaries."
    }
}

MODEL_NAMES = list(MODEL_SPECS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Streamlit Compatibility Helper
# ─────────────────────────────────────────────────────────────────────────────
def render_dataframe(df, **kwargs):
    """Safely renders DataFrame with full container width and clean hidden index across Streamlit versions."""
    sig = inspect.signature(st.dataframe)
    if "width" in sig.parameters:
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    if "hide_index" in sig.parameters and "hide_index" not in kwargs:
        kwargs["hide_index"] = True
    return st.dataframe(df, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Asset Loaders (Cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_preprocessor():
    path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(path):
        st.error(f"Fitted preprocessor not found at `{path}`. Please run `python train_models.py` first.")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_models():
    models = {}
    for name, spec in MODEL_SPECS.items():
        path = os.path.join(MODEL_DIR, spec["filename"])
        if not os.path.exists(path):
            st.error(f"Saved model artifact not found at `{path}`. Please run `python train_models.py` first.")
            st.stop()
        models[name] = joblib.load(path)
    return models

@st.cache_data
def load_default_test_data():
    if os.path.exists(DEFAULT_TEST_CSV):
        return pd.read_csv(DEFAULT_TEST_CSV)
    st.error(f"Held-out test dataset not found at `{DEFAULT_TEST_CSV}`.")
    st.stop()

@st.cache_data
def get_sample_csv_bytes():
    if os.path.exists(DEFAULT_TEST_CSV):
        with open(DEFAULT_TEST_CSV, "rb") as f:
            return f.read()
    return b""

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation Metric Calculation Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Computes all 6 classification evaluation metrics."""
    has_two_classes = len(np.unique(y_true)) > 1
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_prob)) if has_two_classes else 0.0,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

def get_probabilities(model, X_scaled: np.ndarray) -> np.ndarray:
    """Extracts positive class probability P(Y=1) safely."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    if hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_scaled)
        return (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
    return model.predict(X_scaled).astype(float)

def validate_and_prepare_dataset(df: pd.DataFrame):
    """
    Validates uploaded test CSV:
    - Checks target column existence ('ReadyBiodegradable' or aliases)
    - Checks for all 41 DRAGON descriptors
    - Binarizes target ('RB'->1, 'NRB'->0, or integer 0/1)
    """
    target_candidate = None
    for col in [TARGET_COL, "target", "Class", "class"]:
        if col in df.columns:
            target_candidate = col
            break

    if target_candidate is None:
        return None, None, f"Target column `{TARGET_COL}` not found in uploaded CSV. Please include `{TARGET_COL}` with 0/1 or NRB/RB values."

    missing_descriptors = [col for col in DESCRIPTOR_COLS if col not in df.columns]
    if missing_descriptors:
        preview = ", ".join(missing_descriptors[:4])
        ellipsis_str = f" ... and {len(missing_descriptors) - 4} more" if len(missing_descriptors) > 4 else ""
        return None, None, f"Missing {len(missing_descriptors)} required molecular descriptors: {preview}{ellipsis_str}."

    # Parse target
    raw_target = df[target_candidate].astype(str).str.strip()
    if raw_target.isin(["RB", "NRB"]).any():
        y = (raw_target == "RB").astype(int).values
    else:
        y = pd.to_numeric(raw_target, errors="coerce").fillna(0).astype(int).values

    X = df[DESCRIPTOR_COLS].copy()
    for col in DESCRIPTOR_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X, y, None

# ─────────────────────────────────────────────────────────────────────────────
# 7. Matplotlib / Seaborn Plot Helpers (White Theme)
# ─────────────────────────────────────────────────────────────────────────────
def create_white_fig(figsize=(6, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    ax.tick_params(colors="#334155", labelsize=9)
    return fig, ax

def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    fig, ax = create_white_fig(figsize=(5.6, 4.4))
    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)
    
    annot_matrix = [
        [f"TN\n{tn:,}\n({tn/total:.1%})", f"FP\n{fp:,}\n({fp/total:.1%})"],
        [f"FN\n{fn:,}\n({fn/total:.1%})", f"TP\n{tp:,}\n({tp/total:.1%})"]
    ]
    
    sns.heatmap(
        cm,
        annot=np.array(annot_matrix),
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Not Ready (0)", "Ready (1)"],
        yticklabels=["Not Ready (0)", "Ready (1)"],
        linewidths=1.5,
        linecolor="#E2E8F0"
    )
    
    threshold = cm.max() / 2.0
    for text in ax.texts:
        x, y = text.get_position()
        j, i = int(x), int(y)
        val = cm[i, j]
        text.set_color("#FFFFFF" if val > threshold else "#0F172A")
        text.set_weight("bold")
        text.set_fontsize(11)

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("Predicted Label (0: NRB, 1: RB)", fontsize=9, fontweight="bold", color="#334155", labelpad=8)
    ax.set_ylabel("True Label (0: NRB, 1: RB)", fontsize=9, fontweight="bold", color="#334155", labelpad=8)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 8. Load Pre-trained Artifacts
# ─────────────────────────────────────────────────────────────────────────────
preprocessor = load_preprocessor()
trained_models = load_models()
sample_csv_data = get_sample_csv_bytes()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Sidebar Control Panel (Evaluator-First UX)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.05em;">
                AIMLZG565 · Assignment 2
            </div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A;">
                Control Panel
            </div>
            <div style="font-size: 0.80rem; color: #64748B;">
                Student id: <b>2026AJ05162</b>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # ── 1. Test Dataset Upload ───────────────────────────────────────────────
    st.markdown("### 📂 1. Test Dataset Upload")
    uploaded_file = st.file_uploader(
        "Upload Test Data CSV (with target column)",
        type=["csv"],
        help="Upload CSV containing 41 DRAGON descriptors + 'ReadyBiodegradable' column."
    )
    
    st.download_button(
        label="📥 Download Sample test_data.csv",
        data=sample_csv_data,
        file_name="test_data.csv",
        mime="text/csv",
        help="Download the held-out test data (211 compounds with ground truth) to test manual upload.",
        use_container_width=True
    )

    st.divider()

    # ── 2. Model Selection Dropdown ──────────────────────────────────────────
    st.markdown("### 🤖 2. Model Selection")
    selected_model_name = st.selectbox(
        "Select Classification Model",
        MODEL_NAMES,
        index=4,  # Default to Support Vector Machine (🏆 Overall Winner)
        help="Choose one of the 5 required classification architectures to inspect."
    )

    # Selected Model Hyperparameter Summary Box
    spec = MODEL_SPECS[selected_model_name]
    st.markdown(f"""
        <div class="hp-box">
            <div class="hp-title">
                <span>⚙️ Model Hyperparameters</span>
            </div>
            <div class="hp-row"><b>Architecture:</b> {selected_model_name}</div>
            <div class="hp-row"><b>Type:</b> {spec['category']}</div>
            <div class="hp-row"><b>Parameters:</b> <code>{spec['params']}</code></div>
            <div class="hp-row"><b>Random Seed:</b> <code>{RANDOM_SEED}</code></div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Decision Threshold Tuning ───────────────────────────────────────────
    st.markdown("### ⚖️ 3. Classification Threshold")
    threshold = st.slider(
        "Decision Threshold (P(RB) ≥ τ)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability cutoff for predicting compound as Ready Biodegradable (RB). Default: 0.50."
    )
    st.caption("Default classification decision threshold: 0.50")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Data Ingestion & Validation
# ─────────────────────────────────────────────────────────────────────────────
active_df = None
is_custom_upload = False

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        X_parsed, y_parsed, err = validate_and_prepare_dataset(raw_df)
        if err:
            st.error(f"❌ Upload Error: {err}")
            st.info("Falling back to bundled `test_data.csv` so the dashboard remains functional.")
            active_df = load_default_test_data()
            X_parsed, y_parsed, _ = validate_and_prepare_dataset(active_df)
        else:
            active_df = raw_df
            is_custom_upload = True
    except Exception as exc:
        st.error(f"❌ Failed to parse uploaded CSV: {exc}")
        active_df = load_default_test_data()
        X_parsed, y_parsed, _ = validate_and_prepare_dataset(active_df)
else:
    active_df = load_default_test_data()
    X_parsed, y_parsed, _ = validate_and_prepare_dataset(active_df)

# Scale features strictly with pre-fitted preprocessor (zero data leakage)
X_scaled = preprocessor.transform(X_parsed)
y_true = y_parsed
n_total = len(active_df)
n_nrb = int(np.sum(y_true == 0))
n_rb = int(np.sum(y_true == 1))

# ─────────────────────────────────────────────────────────────────────────────
# 11. Top Header Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div class="header-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
            <div>
                <div class="header-title">AIMLZG565: Machine Learning — Assignment 2</div>
                <div class="header-subtitle">Comprehensive In-Silico Biodegradability Classification via Molecular Descriptors</div>
            </div>
            <div>
                <span class="badge-status">● Live Streamlit App</span>
            </div>
        </div>
        <div class="meta-grid">
            <div class="meta-item"><b>Student id:</b> 2026AJ05162</div>
            <div class="meta-item"><b>Dataset:</b> UCI QSAR Biodegradation (ID: 254)</div>
            <div class="meta-item"><b>Features:</b> 41 Molecular Descriptors</div>
            <div class="meta-item"><b>Total Corpus:</b> 1,055 Compounds (844 Train / 211 Test)</div>
            <div class="meta-item"><b>Seed:</b> <code>{RANDOM_SEED}</code></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Dataset Status Info Banner
if not is_custom_upload:
    st.markdown("""
        <div class="info-banner">
            <span>ℹ️</span>
            <div><b>Using bundled test_data.csv by default</b> (211 held-out compounds: 140 NRB / 71 RB). Upload a custom CSV in the sidebar to evaluate other test data.</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="success-banner">
            <b>✅ Custom test dataset active:</b> <code>{uploaded_file.name}</code> ({n_total} compounds, {X_parsed.shape[1]} descriptors: {n_nrb} NRB / {n_rb} RB).
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 12. Main Dashboard Layout (Native Tabs)
# ─────────────────────────────────────────────────────────────────────────────
tab_eval, tab_benchmark = st.tabs([
    "🎯 Model Evaluation & Explorer",
    "📊 Multi-Model Benchmark"
])

# Compute live predictions and probabilities for ALL models across the active test data
all_model_metrics = {}
all_model_probs = {}
all_model_preds = {}
all_model_cms = {}

for name in MODEL_NAMES:
    mdl = trained_models[name]
    probs = get_probabilities(mdl, X_scaled)
    preds = (probs >= threshold).astype(int)
    all_model_probs[name] = probs
    all_model_preds[name] = preds
    all_model_metrics[name] = compute_metrics(y_true, preds, probs)
    all_model_cms[name] = confusion_matrix(y_true, preds)

selected_metrics = all_model_metrics[selected_model_name]
selected_probs = all_model_probs[selected_model_name]
selected_preds = all_model_preds[selected_model_name]
selected_cm = all_model_cms[selected_model_name]

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Selected Model Evaluation & Explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab_eval:
    is_svm = (selected_model_name == "Support Vector Machine")
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <h2 style="margin: 0; font-size: 1.25rem; color: #0F172A;">
                Selected Architecture: {selected_model_name}
            </h2>
            {"<span class='badge-winner'>🏆 Overall Winner</span>" if is_svm else ""}
        </div>
    """, unsafe_allow_html=True)
    st.caption(f"{MODEL_SPECS[selected_model_name]['description']} Evaluating dynamically on {n_total} test compounds.")

    # ── Model Evaluation Metrics ─────────────────────────────────────────────
    st.markdown("#### Performance Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
            <div class="metric-card-wrapper">
                <div class="metric-title">Accuracy</div>
                <div class="metric-number">{selected_metrics['Accuracy']:.4f}</div>
                <div class="metric-desc">{selected_metrics['Accuracy']*100:.2f}% Correct</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="metric-card-wrapper">
                <div class="metric-title">AUC Score</div>
                <div class="metric-number">{selected_metrics['AUC']:.4f}</div>
                <div class="metric-desc">ROC-AUC Discrim.</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="metric-card-wrapper">
                <div class="metric-title">Precision</div>
                <div class="metric-number">{selected_metrics['Precision']:.4f}</div>
                <div class="metric-desc">PPV (RB Class)</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="metric-card-wrapper">
                <div class="metric-title">Recall</div>
                <div class="metric-number">{selected_metrics['Recall']:.4f}</div>
                <div class="metric-desc">Sensitivity / TPR</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class="metric-card-wrapper">
                <div class="metric-title">F1 Score</div>
                <div class="metric-number">{selected_metrics['F1']:.4f}</div>
                <div class="metric-desc">Harmonic Mean</div>
            </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
            <div class="metric-card-wrapper">
                <div class="metric-title">MCC Score</div>
                <div class="metric-number">{selected_metrics['MCC']:.4f}</div>
                <div class="metric-desc">Matthews Correlation</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ── Confusion Matrix & Classification Report ─────────────────────────────
    st.markdown("#### Confusion Matrix & Classification Report")
    col_cm, col_report = st.columns([1, 1.2])

    with col_cm:
        st.markdown("**Confusion Matrix Heatmap**")
        fig_cm = plot_confusion_matrix(selected_cm, selected_model_name)
        st.pyplot(fig_cm)
        tn, fp, fn, tp = selected_cm.ravel()
        st.caption(f"**Breakdown:** TN: **{tn}** · FP: **{fp}** · FN: **{fn}** · TP: **{tp}** (Total: **{n_total}**)")

    with col_report:
        st.markdown("**Per-Class Classification Report**")
        report_dict = classification_report(
            y_true,
            selected_preds,
            target_names=["Not Ready (0)", "Ready (1)"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).T.round(4)
        report_df["support"] = report_df["support"].astype(int)
        
        # Style classification report dataframe
        styled_report = report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:d}"
        })
        render_dataframe(styled_report)

        # Single-model ROC curve directly below classification report
        st.markdown("**Receiver Operating Characteristic (ROC)**")
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, selected_probs)
            fig_roc, ax_roc = create_white_fig(figsize=(6, 2.6))
            ax_roc.plot(fpr, tpr, color="#2563EB", lw=2.2, label=f"{selected_model_name} (AUC = {selected_metrics['AUC']:.4f})")
            ax_roc.plot([0, 1], [0, 1], "--", color="#94A3B8", lw=1.2, label="Random Chance")
            ax_roc.set_xlabel("False Positive Rate", fontsize=8.5, fontweight="bold", color="#334155")
            ax_roc.set_ylabel("True Positive Rate", fontsize=8.5, fontweight="bold", color="#334155")
            ax_roc.set_title(f"ROC Curve — {selected_model_name}", fontsize=10, fontweight="bold", color="#0F172A")
            ax_roc.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor="#E2E8F0")
            ax_roc.grid(color="#F1F5F9", linestyle="-", linewidth=1)
            fig_roc.tight_layout()
            st.pyplot(fig_roc)
        else:
            st.info("Single class in test data; ROC curve requires at least two distinct classes.")

    st.markdown("---")

    # ── 🧪 Compound Predictions Explorer ─────────────────────────────────────
    st.markdown(f"### 🧪 Compound Predictions Explorer — {selected_model_name}")
    st.caption("Inspect individual compound predictions, class probabilities, and diagnostic error classifications.")

    results_df = active_df.copy()
    results_df["Ground_Truth"] = y_true
    results_df["Predicted_Class"] = selected_preds
    results_df["P(RB)"] = np.round(selected_probs, 4)
    results_df["Status"] = np.where(
        y_true == selected_preds, "Correct ✅",
        np.where((y_true == 0) & (selected_preds == 1), "False Positive ⚠️", "False Negative ❌")
    )

    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        status_filter = st.radio(
            "Filter Prediction Outcome",
            ["All Compounds", "Classification Errors Only", "False Positives (Type I)", "False Negatives (Type II)", "Correct Predictions"],
            horizontal=True
        )

    if status_filter == "Classification Errors Only":
        filtered_view = results_df[results_df["Status"] != "Correct ✅"]
    elif status_filter == "False Positives (Type I)":
        filtered_view = results_df[results_df["Status"] == "False Positive ⚠️"]
    elif status_filter == "False Negatives (Type II)":
        filtered_view = results_df[results_df["Status"] == "False Negative ❌"]
    elif status_filter == "Correct Predictions":
        filtered_view = results_df[results_df["Status"] == "Correct ✅"]
    else:
        filtered_view = results_df

    st.markdown(f"**Showing {len(filtered_view)} compounds:**")
    display_cols = ["Ground_Truth", "Predicted_Class", "P(RB)", "Status"] + DESCRIPTOR_COLS[:8]
    render_dataframe(filtered_view[display_cols].head(100))

    csv_export_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Full Predictions CSV ({selected_model_name})",
        data=csv_export_bytes,
        file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 Quick Preview: All 5 Models Benchmark Leaderboard & Winner Table"):
        preview_rows = []
        for name in MODEL_NAMES:
            m = all_model_metrics[name]
            preview_rows.append({
                "ML Model Name": name,
                "Accuracy": m["Accuracy"],
                "AUC": m["AUC"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "MCC": m["MCC"]
            })
        preview_df = pd.DataFrame(preview_rows)
        render_dataframe(
            preview_df.style
            .hide(axis="index")
            .format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})
            .highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#DBEAFE")
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Multi-Model Benchmark Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("### Multi-Model Benchmark Comparison")
    st.caption("Quantitative benchmarking of all 5 required machine learning classifiers evaluated on active test data.")

    # ── Benchmark Metrics Table ──────────────────────────────────────────────
    table_records = []
    for name in MODEL_NAMES:
        m = all_model_metrics[name]
        table_records.append({
            "ML Model Name": name,
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"]
        })
    
    benchmark_df = pd.DataFrame(table_records)

    st.markdown("#### Model Comparison Table")
    render_dataframe(
        benchmark_df.style
        .hide(axis="index")
        .format({col: "{:.4f}" for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})
        .highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#DBEAFE"),
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── Side-by-Side Visual Benchmarks ───────────────────────────────────────
    col_bar, col_rocs = st.columns(2)

    with col_bar:
        st.markdown("#### Multi-Metric Performance Bar Chart")
        fig_bar, ax_bar = create_white_fig(figsize=(7.5, 4.8))
        
        melted_df = pd.melt(
            benchmark_df,
            id_vars=["ML Model Name"],
            value_vars=["Accuracy", "AUC", "F1", "MCC"],
            var_name="Metric",
            value_name="Score"
        )
        
        palette_dict = {name: MODEL_SPECS[name]["color"] for name in MODEL_NAMES}
        
        sns.barplot(
            data=melted_df,
            x="Metric",
            y="Score",
            hue="ML Model Name",
            palette=palette_dict,
            ax=ax_bar,
            edgecolor="#FFFFFF",
            linewidth=0.8
        )
        ax_bar.set_ylim(0.35, 1.0)
        ax_bar.set_title("Classifier Comparison across Key Metrics", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
        ax_bar.set_ylabel("Metric Score", fontsize=9, fontweight="bold", color="#334155")
        ax_bar.set_xlabel("Evaluation Metric", fontsize=9, fontweight="bold", color="#334155")
        ax_bar.legend(title="", fontsize=7.5, frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower right")
        ax_bar.grid(axis="y", color="#F1F5F9", linestyle="-", linewidth=1)
        fig_bar.tight_layout()
        st.pyplot(fig_bar)

    with col_rocs:
        st.markdown("#### Multi-Model ROC Curves")
        fig_roc_all, ax_roc_all = create_white_fig(figsize=(7.5, 4.8))
        
        if len(np.unique(y_true)) > 1:
            for name in MODEL_NAMES:
                fpr, tpr, _ = roc_curve(y_true, all_model_probs[name])
                auc_val = auc(fpr, tpr)
                ax_roc_all.plot(
                    fpr,
                    tpr,
                    lw=2.0,
                    color=MODEL_SPECS[name]["color"],
                    label=f"{name} (AUC = {auc_val:.4f})"
                )
            ax_roc_all.plot([0, 1], [0, 1], "--", color="#94A3B8", lw=1.2, label="Random Chance")
            ax_roc_all.set_title("Receiver Operating Characteristic Comparison", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
            ax_roc_all.set_xlabel("False Positive Rate", fontsize=9, fontweight="bold", color="#334155")
            ax_roc_all.set_ylabel("True Positive Rate", fontsize=9, fontweight="bold", color="#334155")
            ax_roc_all.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor="white", edgecolor="#E2E8F0")
            ax_roc_all.grid(color="#F1F5F9", linestyle="-", linewidth=1)
            fig_roc_all.tight_layout()
            st.pyplot(fig_roc_all)
        else:
            st.info("Single class in test data; ROC curves require at least two distinct classes.")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Model Performance Observations & Overall Winner Table ───────────────
    st.markdown("#### Model Performance Observations & Overall Winner")
    
    observations_rows = [
        ("Logistic Regression",
         "Strong Linear Baseline: Achieves 88.63% Accuracy and 0.9358 AUC. The L2 ridge penalty (C=1.0) effectively controls coefficient inflation on collinear molecular descriptors. Positive weights for oxygen atom count (<code>nO</code>) and ester linkages (<code>nArCOOR</code>) align with biochemical enzymatic cleavage pathways facilitating biodegradation."),
        ("Decision Tree",
         "Moderate Performance with Pruning: Delivers 80.57% Accuracy, 0.7007 F1-score, and 0.5580 MCC. Pre-pruning to <code>max_depth=5</code> successfully prevents unconstrained overfitting, but axis-aligned orthogonal decision boundaries struggle to model diagonal chemical response surfaces."),
        ("KNN",
         "Effective Metric-Space Learning: Achieves 87.68% Accuracy and 0.9326 AUC. Leverages the chemical structure-activity similarity principle: congeneric molecules with proximate topological descriptors exhibit similar bioactivity. StandardScaler was essential to prevent Euclidean distance distortion."),
        ("Naive Bayes",
         "Extreme Sensitivity / High Recall: Achieves highest Recall across all models (95.77%, detecting 68 of 71 biodegradable compounds), but lowest Precision (49.28%) due to 70 false positives. Correlated molecular descriptors violate class-conditional independence, inflating joint posteriors."),
        ("Support Vector Machine",
         "Superior Non-Linear Margin Maximization (Overall Winner 🏆): Leads across Accuracy (89.57%), AUC (0.9534), Precision (86.57%), F1 (0.8406), and MCC (0.7639). The RBF kernel maps the 41-D descriptor space into an infinite-dimensional Hilbert space, smoothly untangling complex non-linear biochemical boundaries while the maximum-margin hyperplane enforces strong structural risk minimization."),
        ("Overall, Winner for your dataset?",
         "Support Vector Machine (RBF Kernel) 🏆: Definitive overall winner for the QSAR Biodegradation dataset. With moderate sample size (n=844 training compounds) and high feature dimensionality (d=41), kernel SVM strikes the optimal bias-variance tradeoff via structural risk minimization. It produces the fewest false positives (only 9/140) and achieves the highest Matthews Correlation Coefficient (0.7639) — the most reliable metric for class-imbalanced binary classification.")
    ]
    
    obs_table_rows_html = []
    for model_name, obs_text in observations_rows:
        is_winner = "Overall" in model_name
        
        # Format key titles in bold for clean visual hierarchy
        if ":" in obs_text:
            title_part, rest_part = obs_text.split(":", 1)
            formatted_obs = f"<b>{title_part}:</b>{rest_part}"
        else:
            formatted_obs = obs_text

        if is_winner:
            tr_attrs = 'class="winner-row" style="background-color: #EFF6FF !important; border-left: 4px solid #2563EB;"'
            name_style = 'style="padding: 14px 18px; font-weight: 800; color: #1D4ED8; vertical-align: top; width: 24%; min-width: 170px; border-right: 1px solid #DBEAFE; line-height: 1.6;"'
            content_style = 'style="padding: 14px 18px; color: #1E3A8A; vertical-align: top; width: 76%; line-height: 1.65; white-space: normal !important; word-wrap: break-word !important;"'
        else:
            tr_attrs = 'style="border-bottom: 1px solid #F1F5F9;"'
            name_style = 'style="padding: 14px 18px; font-weight: 700; color: #0F172A; vertical-align: top; width: 24%; min-width: 170px; border-right: 1px solid #F1F5F9; line-height: 1.6;"'
            content_style = 'style="padding: 14px 18px; color: #1E293B; vertical-align: top; width: 76%; line-height: 1.65; white-space: normal !important; word-wrap: break-word !important;"'
            
        row_html = f'<tr {tr_attrs}><td class="model-name" {name_style}>{model_name}</td><td class="obs-content" {content_style}>{formatted_obs}</td></tr>'
        obs_table_rows_html.append(row_html)

    rows_joined = "".join(obs_table_rows_html)
    obs_html = (
        '<div class="obs-table-container" style="background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-top: 10px; margin-bottom: 18px;">'
        '<table class="obs-table" style="width: 100%; border-collapse: collapse; font-family: inherit; font-size: 0.90rem; background-color: #FFFFFF; text-align: left;">'
        '<thead><tr>'
        '<th style="width: 24%; min-width: 170px; background-color: #F8FAFC; color: #0F172A; font-weight: 700; padding: 14px 18px; border-bottom: 2px solid #E2E8F0; text-align: left;">ML Model Name</th>'
        '<th style="width: 76%; background-color: #F8FAFC; color: #0F172A; font-weight: 700; padding: 14px 18px; border-bottom: 2px solid #E2E8F0; text-align: left;">Observation about model performance</th>'
        '</tr></thead>'
        f'<tbody>{rows_joined}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(obs_html, unsafe_allow_html=True)

    # Clean CSV download for evaluator convenience
    obs_export_df = pd.DataFrame(observations_rows, columns=["ML Model Name", "Observation about model performance"])
    obs_export_df["Observation about model performance"] = obs_export_df["Observation about model performance"].str.replace(r"<[^>]*>", "", regex=True)
    obs_csv_bytes = obs_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Observations Table (CSV)",
        data=obs_csv_bytes,
        file_name="model_performance_observations.csv",
        mime="text/csv",
        help="Download the complete observations table in clean CSV format."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 13. Application Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="footer-container">
        <b>BITS Pilani WILP M.Tech (AIML)</b> · Course: AIMLZG565 (Machine Learning) — Assignment 2<br>
        Developed by <b>Adithya Sathish Kumar</b> · 
        <a href="https://github.com/AdiS-2105/ml-assignment-2" target="_blank">GitHub Repository</a> · 
        <a href="https://adis-2105-ml-assignment-2.streamlit.app/" target="_blank">Streamlit Cloud App</a> · 
        Seed: <code>2002</code>
    </div>
""", unsafe_allow_html=True)

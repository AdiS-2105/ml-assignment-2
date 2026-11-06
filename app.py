"""
BioDeg-Predict: In-Silico Molecular Biodegradation Classification Studio
Course: Machine Learning (AIMLZG565), BITS Pilani WILP M.Tech (AIML)
Student: Adithya S | Random Seed: 2002

This interactive Streamlit web application fulfills all criteria for BITS ML Assignment 2:
- Dynamic test dataset uploader (with one-click default test_data.csv loader)
- Model selection dropdown supporting individual inspection and 5-model comparison
- Live evaluation of all 6 rubric metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap and detailed classification report
- Zero retraining: utilizes pre-trained artifacts serialized in model/
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
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
    auc
)
import joblib

# -----------------------------------------------------------------------------
# Page Configuration & Visual Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BioDeg-Predict | ML Assignment 2",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .metric-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-winner {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
    }
    .student-badge {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .status-box {
        background-color: #0d2818;
        border: 1px solid #16a34a;
        color: #86efac;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants & Descriptors
# -----------------------------------------------------------------------------
RANDOM_SEED = 2002
MODEL_DIR = "model"
DEFAULT_TEST_PATH = "test_data.csv"

MOLECULAR_DESCRIPTOR_COLUMNS = [
    'SpMax_L', 'J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-', 'C%',
    'nCp', 'nO', 'F03[C-N]', 'SdssC', 'HyWi_B(m)', 'LOC', 'SM6_L', 'F03[C-O]',
    'Me', 'Mi', 'nN-N', 'nArNO2', 'nCRX3', 'SpPosA_B(p)', 'nCIR', 'B01[C-Br]',
    'B03[C-Cl]', 'N-073', 'SpMax_A', 'Psi_i_1d', 'B04[C-Br]', 'SdO', 'TI2_L',
    'nCrt', 'C-026', 'F02[C-N]', 'nHDon', 'SpMax_B(m)', 'Psi_i_A', 'nN',
    'SM6_B(m)', 'nArCOOR', 'nX'
]
TARGET_COLUMN = 'ReadyBiodegradable'

MODEL_FILE_REGISTRY = {
    'Support Vector Machine': ('svm.joblib', 'SVM (RBF Kernel) - Overall Winner 🏆'),
    'Logistic Regression': ('logistic_regression.joblib', 'Logistic Regression (L2 Regularized)'),
    'KNN': ('knn.joblib', 'K-Nearest Neighbors (k=7)'),
    'Decision Tree': ('decision_tree.joblib', 'Decision Tree (Pruned Depth=5)'),
    'Naive Bayes': ('naive_bayes.joblib', 'Gaussian Naive Bayes (Generative)')
}


# -----------------------------------------------------------------------------
# Cached Asset Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_molecular_preprocessor():
    path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(path):
        st.error(f"Preprocessor artifact not found at {path}. Please run train_models.py first.")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def load_classification_models():
    models = {}
    for key, (filename, _) in MODEL_FILE_REGISTRY.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(model_path):
            models[key] = joblib.load(model_path)
        else:
            st.error(f"Missing model artifact: {model_path}. Run train_models.py to build it.")
            st.stop()
    return models


@st.cache_data
def load_default_test_dataset():
    if os.path.exists(DEFAULT_TEST_PATH):
        return pd.read_csv(DEFAULT_TEST_PATH)
    return None


@st.cache_data
def load_precomputed_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# -----------------------------------------------------------------------------
# Metric Calculation Utilities
# -----------------------------------------------------------------------------
def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'AUC': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_true, y_pred))
    }


def parse_and_validate_dataframe(uploaded_df: pd.DataFrame):
    df = uploaded_df.copy()
    
    # Check for target column
    found_target = None
    for candidate in [TARGET_COLUMN, 'target', 'Class', 'class', 'biodeg', 'Ready_Biodegradable']:
        if candidate in df.columns:
            found_target = candidate
            break
            
    if found_target is None:
        st.error(f"Target column not found! Expected '{TARGET_COLUMN}' or 'Class' in the CSV.")
        return None, None, None

    # Standardize target values to 0 and 1
    if df[found_target].dtype == object or df[found_target].dtype == str:
        # Check if values are strings like 'RB' / 'NRB'
        cleaned_series = df[found_target].astype(str).str.strip()
        if cleaned_series.isin(['RB', 'NRB']).any():
            y = (cleaned_series == 'RB').astype(int).values
        else:
            y = pd.to_numeric(cleaned_series, errors='coerce').fillna(0).astype(int).values
    else:
        y = df[found_target].astype(int).values

    # Check for missing molecular descriptors
    missing_cols = [c for c in MOLECULAR_DESCRIPTOR_COLUMNS if c not in df.columns]
    if missing_cols:
        st.error(f"Missing {len(missing_cols)} required molecular descriptor columns: {missing_cols[:5]}...")
        return None, None, None

    X = df[MOLECULAR_DESCRIPTOR_COLUMNS].copy()
    return X, y, found_target


# -----------------------------------------------------------------------------
# Sidebar: Controls & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="student-badge">Adithya S | AIMLZG565</div>', unsafe_allow_html=True)
    st.title("🧪 Studio Controls")
    st.caption("BITS Pilani WILP M.Tech (AIML) - ML Assignment 2")
    st.markdown("---")

    # Step a: Dataset Upload Option (CSV)
    st.subheader("📂 1. Test Dataset Upload")
    uploaded_file = st.file_uploader(
        "Upload Held-out Test Data (CSV)",
        type=["csv"],
        help="Upload held-out test data containing 41 molecular descriptors and ReadyBiodegradable target."
    )

    use_sample = st.checkbox("Use default test_data.csv (211 instances)", value=True if uploaded_file is None else False)

    st.markdown("---")

    # Step b: Model Selection Dropdown
    st.subheader("🤖 2. Model Selection")
    model_choices = ["All 5 Models Comparison (Leaderboard)"] + list(MODEL_FILE_REGISTRY.keys())
    selected_option = st.selectbox(
        "Select Model Architecture:",
        options=model_choices,
        index=1,  # Default to Support Vector Machine (Overall Winner)
        format_func=lambda x: MODEL_FILE_REGISTRY[x][1] if x in MODEL_FILE_REGISTRY else f"📊 {x}"
    )

    st.markdown("---")
    st.subheader("⚙️ 3. Inference Parameters")
    classification_threshold = st.slider(
        "Decision Threshold (Positive Class: RB)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Adjust probability threshold for classifying a compound as Ready Biodegradable."
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">
            <b>Evaluation Reproducibility:</b><br>
            • Seed: <code>2002</code><br>
            • Metric Standards: Accuracy, AUC, Precision, Recall, F1, MCC<br>
            • Preprocessing: <code>StandardScaler</code> (Fitted strictly on 80% train split)
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# Load Preprocessing Artifacts & Models
# -----------------------------------------------------------------------------
preprocessor = load_molecular_preprocessor()
trained_models = load_classification_models()
precomputed_metrics = load_precomputed_metrics()


# -----------------------------------------------------------------------------
# Main Application Canvas
# -----------------------------------------------------------------------------
st.title("🧪 BioDeg-Predict: In-Silico Molecular Biodegradation Studio")
st.markdown(
    "Predicting ready biodegradability of chemical compounds from **41 DRAGON molecular descriptors** "
    "using calibrated machine learning models. Built for **BITS Pilani WILP M.Tech (AIML) Assignment 2**."
)

# Ingest test dataset from file uploader or default test_data.csv
active_df = None
data_source_label = ""

if uploaded_file is not None:
    try:
        active_df = pd.read_csv(uploaded_file)
        data_source_label = f"Uploaded File: `{uploaded_file.name}`"
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_sample:
    active_df = load_default_test_dataset()
    data_source_label = "Default Held-Out Dataset (`test_data.csv` - 211 compounds, 80:20 Stratified Split)"

if active_df is None:
    st.info("👆 Please upload a test CSV file or check 'Use default test_data.csv' in the sidebar to begin.")
    st.stop()

# Parse features and target
X_raw, y_true, target_col_name = parse_and_validate_dataframe(active_df)
if X_raw is None:
    st.stop()

# Scale features strictly with preprocessor (zero leakage)
X_scaled = preprocessor.transform(X_raw)

st.markdown(
    f"""
    <div class="status-box">
        ✅ <b>Dataset Loaded:</b> {data_source_label} | <b>Instances:</b> {len(active_df)} compounds | <b>Descriptors:</b> {X_raw.shape[1]} features | <b>Class Balance:</b> {np.sum(y_true == 0)} NRB / {np.sum(y_true == 1)} RB
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# View 1: Comparative Leaderboard (When "All 5 Models Comparison" is selected)
# -----------------------------------------------------------------------------
if selected_option == "All 5 Models Comparison (Leaderboard)":
    st.header("📊 Multi-Model Comparative Evaluation Leaderboard")
    st.caption("Rigorous evaluation across all 5 required BITS ML architectures on the active test set.")

    # Compute metrics for all 5 models dynamically on the loaded dataset
    comparison_records = []
    roc_data = {}
    cm_data = {}

    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            dv = model.decision_function(X_scaled)
            probs = (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
        else:
            probs = model.predict(X_scaled).astype(float)
            
        preds = (probs >= classification_threshold).astype(int)
        metrics = calculate_all_metrics(y_true, preds, probs)
        metrics['ML Model Name'] = name
        comparison_records.append(metrics)
        roc_data[name] = (y_true, probs)
        cm_data[name] = confusion_matrix(y_true, preds)

    comp_df = pd.DataFrame(comparison_records)[['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    comp_df_sorted = comp_df.sort_values(by='MCC', ascending=False).reset_index(drop=True)

    # Display styled comparison table
    st.subheader("Official Comparison Table (BITS Rubric Step 5d)")
    st.dataframe(
        comp_df_sorted.style.format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'MCC': '{:.4f}'
        }).highlight_max(
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            color='#166534'
        ),
        use_container_width=True
    )

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Multi-Metric Bar Chart")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        melted = pd.melt(comp_df, id_vars=['ML Model Name'], value_vars=['Accuracy', 'AUC', 'F1', 'MCC'])
        sns.barplot(data=melted, x='variable', y='value', hue='ML Model Name', ax=ax_bar, palette='viridis')
        ax_bar.set_ylim(0.4, 1.0)
        ax_bar.set_ylabel("Metric Score", fontweight='bold')
        ax_bar.set_xlabel("Evaluation Metric", fontweight='bold')
        ax_bar.set_title("Performance Metrics Across Classifiers", fontweight='bold')
        ax_bar.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig_bar)

    with col_chart2:
        st.subheader("📉 Combined ROC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
        for m_name, (y_t, y_p) in roc_data.items():
            fpr, tpr, _ = roc_curve(y_t, y_p)
            score = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=2, label=f"{m_name} (AUC = {score:.4f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance')
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
        ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=8.5)
        st.pyplot(fig_roc)

    # Observations table
    st.markdown("---")
    st.subheader("📝 Observations on Model Performance (Rubric Step 5d - 3 Marks)")
    obs_data = [
        {
            "ML Model Name": "Logistic Regression",
            "Key Observation": "Competitive linear baseline (Acc: 88.63%, AUC: 0.9358). L2 regularization prevents overfitting on correlated chemical descriptors. Highly interpretable."
        },
        {
            "ML Model Name": "Decision Tree",
            "Key Observation": "Performs recursive axis-aligned splits (Acc: 80.57%, MCC: 0.5580). Pruned to depth 5 to control variance; orthogonal boundaries are suboptimal for smooth chemical response surfaces."
        },
        {
            "ML Model Name": "KNN",
            "Key Observation": "Effective non-parametric neighborhood classifier (Acc: 87.68%, AUC: 0.9326). Reliably identifies structurally similar analog molecules in scaled descriptor space."
        },
        {
            "ML Model Name": "Naive Bayes",
            "Key Observation": "Exceptional recall of 95.77% but lower precision (49.28%) due to violated conditional independence across correlated molecular descriptors. Functions as a sensitive first-pass screening filter."
        },
        {
            "ML Model Name": "Support Vector Machine",
            "Key Observation": "Highest overall performer across Accuracy (89.57%), AUC (0.9534), F1 (0.8406), and MCC (0.7639). RBF kernel maps 41D chemical space into non-linear separability with maximum margin."
        },
        {
            "ML Model Name": "🏆 Overall Winner",
            "Key Observation": "Support Vector Machine (RBF Kernel) is the definitive winner. It provides optimal structural risk minimization and superior generalization on biochemical activity manifolds."
        }
    ]
    st.table(pd.DataFrame(obs_data))


# -----------------------------------------------------------------------------
# View 2: Individual Model Deep-Dive
# -----------------------------------------------------------------------------
else:
    active_model_name = selected_option
    active_model = trained_models[active_model_name]
    display_title = MODEL_FILE_REGISTRY[active_model_name][1]

    st.header(f"🔍 Model Inspection: {display_title}")

    # Compute predictions and probabilities
    if hasattr(active_model, "predict_proba"):
        y_prob = active_model.predict_proba(X_scaled)[:, 1]
    elif hasattr(active_model, "decision_function"):
        dv = active_model.decision_function(X_scaled)
        y_prob = (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
    else:
        y_prob = active_model.predict(X_scaled).astype(float)

    y_pred = (y_prob >= classification_threshold).astype(int)

    # Compute the 6 required evaluation metrics
    metrics = calculate_all_metrics(y_true, y_pred, y_prob)

    # 1. Metric Display Cards (BITS Rubric Step 5c)
    st.subheader("1. Evaluation Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">{metrics['Accuracy']:.2%}</div>
                <div class="metric-badge badge-winner">{metrics['Accuracy']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">ROC-AUC</div>
                <div class="metric-value">{metrics['AUC']:.4f}</div>
                <div class="metric-badge badge-winner">Discrimination</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{metrics['Precision']:.2%}</div>
                <div class="metric-badge badge-winner">PPV</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{metrics['Recall']:.2%}</div>
                <div class="metric-badge badge-winner">Sensitivity</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value">{metrics['F1']:.4f}</div>
                <div class="metric-badge badge-winner">Harmonic Mean</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c6:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">MCC Score</div>
                <div class="metric-value">{metrics['MCC']:.4f}</div>
                <div class="metric-badge badge-winner">Correlation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Confusion Matrix & Classification Report (BITS Rubric Step 5d)
    st.markdown("---")
    st.subheader("2. Diagnostic Analysis (Confusion Matrix & Classification Report)")

    col_cm, col_report = st.columns([1, 1.1])

    with col_cm:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Not Ready (0)", "Ready (1)"],
            yticklabels=["Not Ready (0)", "Ready (1)"]
        )
        ax_cm.set_title(f"Confusion Matrix ({active_model_name})", fontweight="bold", fontsize=11)
        ax_cm.set_xlabel("Predicted Class", fontweight="bold")
        ax_cm.set_ylabel("True Ground Truth", fontweight="bold")
        st.pyplot(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"**Breakdown**: True Negatives = {tn} | False Positives = {fp} | "
            f"False Negatives = {fn} | True Positives = {tp}"
        )

    with col_report:
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=["Not Ready Biodegradable (0)", "Ready Biodegradable (1)"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.markdown("**Per-Class Classification Report**")
        st.dataframe(
            report_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )

        # ROC Curve for selected model
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig_single_roc, ax_sroc = plt.subplots(figsize=(6, 2.5))
        ax_sroc.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {metrics['AUC']:.4f})")
        ax_sroc.plot([0, 1], [0, 1], "k--", lw=1)
        ax_sroc.set_xlabel("FPR", fontsize=8)
        ax_sroc.set_ylabel("TPR", fontsize=8)
        ax_sroc.legend(loc="lower right", fontsize=8)
        st.pyplot(fig_single_roc)

    # 3. Predictions Inspector & Export
    st.markdown("---")
    st.subheader("3. Compound Prediction Explorer & Audit")

    results_table = active_df.copy()
    results_table['Ground_Truth'] = y_true
    results_table['Predicted_Class'] = y_pred
    results_table['Confidence_P(RB)'] = np.round(y_prob, 4)
    results_table['Prediction_Status'] = np.where(
        y_true == y_pred,
        "Correct ✅",
        np.where((y_true == 0) & (y_pred == 1), "False Positive ⚠️", "False Negative ❌")
    )

    filter_option = st.radio(
        "Filter Predictions Preview:",
        options=["All Compounds", "Errors Only (False Positives & Negatives)", "Correct Predictions Only"],
        horizontal=True
    )

    if filter_option == "Errors Only (False Positives & Negatives)":
        filtered_view = results_table[results_table['Prediction_Status'] != "Correct ✅"]
    elif filter_option == "Correct Predictions Only":
        filtered_view = results_table[results_table['Prediction_Status'] == "Correct ✅"]
    else:
        filtered_view = results_table

    display_cols = ['Ground_Truth', 'Predicted_Class', 'Confidence_P(RB)', 'Prediction_Status'] + MOLECULAR_DESCRIPTOR_COLUMNS[:6]
    st.dataframe(filtered_view[display_cols].head(50), use_container_width=True)

    csv_data = results_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Predictions CSV",
        data=csv_data,
        file_name=f"biodeg_predictions_{active_model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# Footer & Academic Integrity Declaration
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px 0;">
        <b>BITS Pilani WILP M.Tech (AIML)</b> | Course: Machine Learning (AIMLZG565) | Assignment 2<br>
        Developed by <b>Adithya S</b> (<a href="https://github.com/AdiS-2105" target="_blank" style="color: #3b82f6; text-decoration: none;">@AdiS-2105</a>) | Reproducible Evaluation Seed: <code>2002</code> | Artifact Pipeline: <code>v1.0.0</code>
    </div>
    """,
    unsafe_allow_html=True
)

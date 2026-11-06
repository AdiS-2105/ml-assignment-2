"""
BioDeg-Predict: QSAR Molecular Biodegradation Classification Studio
Course: Machine Learning (AIMLZG565), BITS Pilani WILP M.Tech (AIML)
Student: Adithya S | Random Seed: 2002

Streamlit web application for BITS ML Assignment 2 (15/15 rubric):
  Step 5a — Dataset upload option (CSV)           [1 Mark]
  Step 5b — Model selection dropdown               [1 Mark]
  Step 5c — Display evaluation metrics             [1 Mark]
  Step 5d — Confusion matrix & observations        [1 Mark + 3 Marks]
"""

import os
import json
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
# Page Config — MUST be the very first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioDeg-Predict | AIMLZG565",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Clean White Theme (Light Mode)
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_THEME_CSS = """
<style>
/* ── Global ────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
}

/* ── Metric Cards ──────────────────────────────────────────────────────── */
.mc {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 12px 14px 12px;
    text-align: center;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.mc:hover {
    border-color: #2563EB;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.10);
}
.mc-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 2px;
}
.mc-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
}
.mc-sub {
    font-size: 0.68rem;
    font-weight: 600;
    color: #94A3B8;
    margin-top: 2px;
}

/* ── Status Banner ─────────────────────────────────────────────────────── */
.status-ok {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #22C55E;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #166534;
    margin-bottom: 20px;
}

/* ── Sidebar Badge ─────────────────────────────────────────────────────── */
.sb {
    background: #2563EB;
    color: #FFFFFF;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 10px;
    letter-spacing: 0.02em;
}

/* ── Section Divider ───────────────────────────────────────────────────── */
.sdiv {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 28px 0 20px 0;
}

/* ── Winner Highlight ──────────────────────────────────────────────────── */
.winner-tag {
    display: inline-block;
    background: #DBEAFE;
    color: #1E40AF;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    margin-left: 6px;
}

/* ── Footer ────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.78rem;
    padding: 16px 0 8px 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 32px;
}
.app-footer a { color: #2563EB; text-decoration: none; font-weight: 600; }
.app-footer a:hover { text-decoration: underline; }

/* ── Tables – remove dark row stripes ──────────────────────────────────── */
[data-testid="stDataFrame"] th {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
}
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 2002
MODEL_DIR = "model"
DEFAULT_TEST_CSV = "test_data.csv"

DESCRIPTOR_COLS = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)",
    "nCIR", "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d",
    "B04[C-Br]", "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon",
    "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX",
]
TARGET_COL = "ReadyBiodegradable"

MODEL_REGISTRY = {
    "Support Vector Machine":  ("svm.joblib",                 "SVM (RBF Kernel)"),
    "Logistic Regression":     ("logistic_regression.joblib",  "Logistic Regression (L2)"),
    "KNN":                     ("knn.joblib",                  "K-Nearest Neighbors (k=7)"),
    "Decision Tree":           ("decision_tree.joblib",        "Decision Tree (depth=5)"),
    "Naive Bayes":             ("naive_bayes.joblib",          "Gaussian Naive Bayes"),
}

# Consistent colour palette across all matplotlib charts
CHART_PALETTE = {
    "Support Vector Machine": "#2563EB",
    "Logistic Regression":    "#0EA5E9",
    "KNN":                    "#8B5CF6",
    "Decision Tree":          "#F59E0B",
    "Naive Bayes":            "#EF4444",
}

# ─────────────────────────────────────────────────────────────────────────────
# Asset Loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_preprocessor():
    p = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(p):
        st.error(f"Preprocessor not found at `{p}`. Run `python train_models.py` first.")
        st.stop()
    return joblib.load(p)


@st.cache_resource
def _load_models():
    models = {}
    for key, (fname, _) in MODEL_REGISTRY.items():
        fp = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fp):
            st.error(f"Missing artifact: `{fp}`. Run `python train_models.py`.")
            st.stop()
        models[key] = joblib.load(fp)
    return models


@st.cache_data
def _load_default_test():
    if os.path.exists(DEFAULT_TEST_CSV):
        return pd.read_csv(DEFAULT_TEST_CSV)
    return None


@st.cache_data
def _load_precomputed():
    mp = os.path.join(MODEL_DIR, "metrics_summary.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "AUC":       float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC":       float(matthews_corrcoef(y_true, y_pred)),
    }


def _get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        dv = model.decision_function(X)
        return (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
    return model.predict(X).astype(float)


def _validate_df(df):
    """Return (X, y, target_name) or (None, None, None) on error."""
    target_name = None
    for c in [TARGET_COL, "target", "Class", "class"]:
        if c in df.columns:
            target_name = c
            break
    if target_name is None:
        st.error(f"Target column not found. Expected `{TARGET_COL}` or `Class`.")
        return None, None, None

    raw = df[target_name].astype(str).str.strip()
    if raw.isin(["RB", "NRB"]).any():
        y = (raw == "RB").astype(int).values
    else:
        y = pd.to_numeric(raw, errors="coerce").fillna(0).astype(int).values

    missing = [c for c in DESCRIPTOR_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing {len(missing)} descriptor columns: {missing[:5]} …")
        return None, None, None

    return df[DESCRIPTOR_COLS].copy(), y, target_name


def _metric_card(label: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="mc">
            <div class="mc-label">{label}</div>
            <div class="mc-value">{value}</div>
            <div class="mc-sub">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _light_fig(figsize=(7, 5)):
    """Create a matplotlib figure with white background matching the theme."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    ax.tick_params(colors="#334155", labelsize=9)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb">Adithya S · AIMLZG565</div>', unsafe_allow_html=True)
    st.title("🧬 Controls")
    st.caption("BITS Pilani WILP M.Tech (AIML)")
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    # ── Step 5a: Dataset Upload (CSV) ──────────────────────────────────────
    st.subheader("📂  Upload Test Data")
    uploaded_file = st.file_uploader(
        "CSV with 41 descriptors + target",
        type=["csv"],
        help="Upload a CSV containing the 41 DRAGON molecular descriptors and the `ReadyBiodegradable` target column.",
    )
    use_default = st.checkbox(
        "Use built-in test_data.csv (211 compounds)",
        value=(uploaded_file is None),
    )

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    # ── Step 5b: Model Selection Dropdown ──────────────────────────────────
    st.subheader("🤖  Select Model")
    model_options = ["Compare All 5 Models"] + list(MODEL_REGISTRY.keys())
    chosen_model = st.selectbox(
        "Classification Architecture",
        options=model_options,
        index=1,   # default = SVM (winner)
        format_func=lambda x: (
            f"📊  {x}" if x == "Compare All 5 Models"
            else f"{MODEL_REGISTRY[x][1]}{' 🏆' if x == 'Support Vector Machine' else ''}"
        ),
    )

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    st.subheader("⚙️  Parameters")
    threshold = st.slider(
        "Decision Threshold",
        0.10, 0.90, 0.50, 0.05,
        help="Probability threshold for classifying a compound as Ready Biodegradable (RB).",
    )

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.caption(
        "**Reproducibility** · Seed `2002`\n\n"
        "**Preprocessing** · StandardScaler fitted on 80 % train split only\n\n"
        "**Metrics** · Accuracy · AUC · Precision · Recall · F1 · MCC"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────
preprocessor = _load_preprocessor()
trained_models = _load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧬 BioDeg-Predict")
st.markdown(
    "Predicting **ready biodegradability** of organic compounds from "
    "**41 DRAGON molecular descriptors** using five calibrated ML classifiers.  \n"
    "_BITS Pilani WILP M.Tech (AIML) · Machine Learning (AIMLZG565) · Assignment 2_"
)

# ─────────────────────────────────────────────────────────────────────────────
# Ingest Data
# ─────────────────────────────────────────────────────────────────────────────
active_df = None
source_label = ""

if uploaded_file is not None:
    try:
        active_df = pd.read_csv(uploaded_file)
        source_label = f"Uploaded: **{uploaded_file.name}**"
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
elif use_default:
    active_df = _load_default_test()
    source_label = "Built-in held-out test set — `test_data.csv` (211 compounds, 80:20 stratified)"

if active_df is None:
    st.info("Upload a test CSV or enable the built-in dataset in the sidebar to begin.")
    st.stop()

X_raw, y_true, _ = _validate_df(active_df)
if X_raw is None:
    st.stop()

X_scaled = preprocessor.transform(X_raw)
n_nrb = int(np.sum(y_true == 0))
n_rb  = int(np.sum(y_true == 1))

st.markdown(
    f'<div class="status-ok">'
    f"✅ &nbsp;<b>Dataset loaded</b> — {source_label} &nbsp;·&nbsp; "
    f"<b>{len(active_df)}</b> compounds &nbsp;·&nbsp; "
    f"<b>{X_raw.shape[1]}</b> descriptors &nbsp;·&nbsp; "
    f"{n_nrb} NRB / {n_rb} RB"
    f"</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW A — All-Model Comparison Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
if chosen_model == "Compare All 5 Models":
    st.header("📊  Model Comparison Leaderboard")
    st.caption("All five architectures evaluated on the active test set.")

    records, roc_data, cm_data = [], {}, {}
    for name, mdl in trained_models.items():
        prob = _get_proba(mdl, X_scaled)
        pred = (prob >= threshold).astype(int)
        m = _compute_metrics(y_true, pred, prob)
        m["Model"] = name
        records.append(m)
        roc_data[name] = (y_true, prob)
        cm_data[name] = confusion_matrix(y_true, pred)

    comp_df = (
        pd.DataFrame(records)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        .sort_values("MCC", ascending=False)
        .reset_index(drop=True)
    )

    # ── Metrics table ──────────────────────────────────────────────────────
    st.subheader("Evaluation Metrics Comparison Table")
    st.dataframe(
        comp_df.style
        .format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})
        .highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#DBEAFE"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Charts ─────────────────────────────────────────────────────────────
    col_bar, col_roc = st.columns(2)

    with col_bar:
        st.subheader("Multi-Metric Bar Chart")
        fig_b, ax_b = _light_fig((8, 5))
        melted = pd.melt(comp_df, id_vars=["Model"], value_vars=["Accuracy", "AUC", "F1", "MCC"])
        sns.barplot(
            data=melted, x="variable", y="value", hue="Model",
            ax=ax_b, palette=CHART_PALETTE, edgecolor="#FFFFFF", linewidth=0.5,
        )
        ax_b.set_ylim(0.35, 1.0)
        ax_b.set_ylabel("Score", fontweight="bold", color="#334155")
        ax_b.set_xlabel("Metric", fontweight="bold", color="#334155")
        ax_b.set_title("Classifier Performance Comparison", fontweight="bold", color="#0F172A", fontsize=12)
        ax_b.legend(fontsize=7.5, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="lower right")
        ax_b.grid(axis="y", color="#F1F5F9", linewidth=0.8)
        st.pyplot(fig_b)

    with col_roc:
        st.subheader("ROC Curves")
        fig_r, ax_r = _light_fig((8, 5))
        for name, (yt, yp) in roc_data.items():
            fpr, tpr, _ = roc_curve(yt, yp)
            ax_r.plot(fpr, tpr, lw=2.2, color=CHART_PALETTE[name],
                      label=f"{name} (AUC={auc(fpr, tpr):.4f})")
        ax_r.plot([0, 1], [0, 1], "--", lw=1, color="#CBD5E1", label="Random Chance")
        ax_r.set_xlabel("False Positive Rate", fontweight="bold", color="#334155")
        ax_r.set_ylabel("True Positive Rate", fontweight="bold", color="#334155")
        ax_r.set_title("Receiver Operating Characteristic", fontweight="bold", color="#0F172A", fontsize=12)
        ax_r.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")
        ax_r.grid(color="#F1F5F9", linewidth=0.8)
        st.pyplot(fig_r)

    # ── Confusion matrices (compact strip) ─────────────────────────────────
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(5)
    for idx, (name, cm) in enumerate(cm_data.items()):
        with cm_cols[idx]:
            fig_c, ax_c = _light_fig((3.2, 2.8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_c,
                annot_kws={"size": 13, "weight": "bold", "color": "#0F172A"},
                xticklabels=["NRB", "RB"], yticklabels=["NRB", "RB"],
                linewidths=1, linecolor="#FFFFFF",
            )
            short = name.split()[0] if name != "Support Vector Machine" else "SVM"
            ax_c.set_title(short, fontsize=10, fontweight="bold", color="#0F172A")
            ax_c.set_xlabel("Pred", fontsize=8, color="#64748B")
            ax_c.set_ylabel("True", fontsize=8, color="#64748B")
            st.pyplot(fig_c)

    # ── Observations on model performance (rubric 3 marks) ─────────────────
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.subheader("📝  Observations on Model Performance")

    observations = [
        ("Logistic Regression",
         "Strong linear baseline (Acc 88.63 %, AUC 0.9358). L2 regularisation prevents overfitting "
         "on collinear molecular descriptors. Highly interpretable; positive coefficients for `nO` "
         "and `nArCOOR` align with enzymatic cleavage sites facilitating microbial degradation."),
        ("Decision Tree",
         "Moderate performer (Acc 80.57 %, MCC 0.5580). Pruned to depth 5 to control variance; "
         "axis-aligned orthogonal splits are sub-optimal for capturing smooth non-linear chemical "
         "response surfaces that span diagonal interaction boundaries."),
        ("KNN",
         "Reliable instance-based learner (Acc 87.68 %, AUC 0.9326). Leverages the chemical "
         "structure–activity similarity principle: molecules with similar topological descriptors "
         "share biodegradability. Feature standardisation was critical to prevent scale distortion "
         "of Euclidean distances."),
        ("Naive Bayes",
         "Highest recall across all models (95.77 %) but lowest precision (49.28 %) and accuracy "
         "(65.40 %). The class-conditional independence assumption is severely violated because "
         "molecular descriptors are inherently correlated, inflating joint posterior probabilities "
         "and generating 70 false positives. Useful as a conservative first-pass safety screen."),
        ("Support Vector Machine",
         "Overall winner 🏆 — Accuracy 89.57 %, AUC 0.9534, Precision 86.57 %, F1 0.8406, "
         "MCC 0.7639. The RBF kernel projects the 41-D descriptor space into an infinite-"
         "dimensional Hilbert space, smoothly separating complex non-linear biochemical boundaries "
         "while the maximum-margin hyperplane enforces strong structural risk minimisation."),
        ("Overall Winner",
         "**Support Vector Machine (RBF Kernel)** is the definitive winner. With moderate sample "
         "size (n = 844) and high dimensionality (d = 41), kernel SVM achieves the best bias–"
         "variance trade-off, produces the fewest false positives (9/140), and attains the highest "
         "Matthews Correlation Coefficient (0.7639) — the most reliable metric for imbalanced "
         "binary classification."),
    ]
    obs_df = pd.DataFrame(observations, columns=["Model", "Observation"])
    st.dataframe(obs_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW B — Individual Model Deep-Dive
# ═══════════════════════════════════════════════════════════════════════════════
else:
    model_name = chosen_model
    model_obj = trained_models[model_name]
    disp_name = MODEL_REGISTRY[model_name][1]
    is_winner = model_name == "Support Vector Machine"

    st.header(f"🔍  {disp_name}" + (" 🏆" if is_winner else ""))

    y_prob = _get_proba(model_obj, X_scaled)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = _compute_metrics(y_true, y_pred, y_prob)

    # ── Step 5c: Evaluation Metrics Display ────────────────────────────────
    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: _metric_card("Accuracy",  f"{metrics['Accuracy']:.2%}",  f"{metrics['Accuracy']:.4f}")
    with c2: _metric_card("ROC-AUC",   f"{metrics['AUC']:.4f}",      "Discrimination")
    with c3: _metric_card("Precision", f"{metrics['Precision']:.2%}", "PPV")
    with c4: _metric_card("Recall",    f"{metrics['Recall']:.2%}",    "Sensitivity")
    with c5: _metric_card("F1 Score",  f"{metrics['F1']:.4f}",       "Harmonic Mean")
    with c6: _metric_card("MCC",       f"{metrics['MCC']:.4f}",      "Correlation")

    # ── Step 5d: Confusion Matrix & Classification Report ──────────────────
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix & Classification Report")

    col_cm, col_rpt = st.columns([1, 1.15])

    with col_cm:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = _light_fig((5.5, 4.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
            annot_kws={"size": 15, "weight": "bold", "color": "#0F172A"},
            xticklabels=["Not Ready (0)", "Ready (1)"],
            yticklabels=["Not Ready (0)", "Ready (1)"],
            linewidths=1.5, linecolor="#FFFFFF",
        )
        ax_cm.set_title(f"Confusion Matrix — {model_name}", fontweight="bold",
                        fontsize=11, color="#0F172A")
        ax_cm.set_xlabel("Predicted Class", fontweight="bold", color="#334155")
        ax_cm.set_ylabel("True Class", fontweight="bold", color="#334155")
        st.pyplot(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN = {tn} · FP = {fp} · FN = {fn} · TP = {tp}")

    with col_rpt:
        rpt = classification_report(
            y_true, y_pred,
            target_names=["Not Ready Biodegradable (0)", "Ready Biodegradable (1)"],
            output_dict=True, zero_division=0,
        )
        rpt_df = pd.DataFrame(rpt).T
        st.markdown("**Per-Class Classification Report**")
        st.dataframe(
            rpt_df.style.format({
                "precision": "{:.4f}", "recall": "{:.4f}",
                "f1-score": "{:.4f}", "support": "{:.0f}",
            }),
            use_container_width=True,
        )

        # Compact ROC curve for this model
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig_sr, ax_sr = _light_fig((6, 2.8))
        ax_sr.plot(fpr, tpr, color="#2563EB", lw=2.2, label=f"AUC = {metrics['AUC']:.4f}")
        ax_sr.plot([0, 1], [0, 1], "--", lw=1, color="#CBD5E1")
        ax_sr.set_xlabel("FPR", fontsize=8, color="#64748B")
        ax_sr.set_ylabel("TPR", fontsize=8, color="#64748B")
        ax_sr.legend(loc="lower right", fontsize=8, frameon=True,
                     facecolor="#FFFFFF", edgecolor="#E2E8F0")
        ax_sr.grid(color="#F1F5F9", linewidth=0.6)
        st.pyplot(fig_sr)

    # ── Predictions Explorer ───────────────────────────────────────────────
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.subheader("Compound Prediction Explorer")

    results = active_df.copy()
    results["Ground_Truth"]     = y_true
    results["Predicted_Class"]  = y_pred
    results["P(RB)"]            = np.round(y_prob, 4)
    results["Status"]           = np.where(
        y_true == y_pred, "Correct ✅",
        np.where((y_true == 0) & (y_pred == 1), "False Positive ⚠️", "False Negative ❌"),
    )

    filt = st.radio(
        "Filter", ["All", "Errors Only", "Correct Only"], horizontal=True,
    )
    if filt == "Errors Only":
        view = results[results["Status"] != "Correct ✅"]
    elif filt == "Correct Only":
        view = results[results["Status"] == "Correct ✅"]
    else:
        view = results

    show_cols = ["Ground_Truth", "Predicted_Class", "P(RB)", "Status"] + DESCRIPTOR_COLS[:6]
    st.dataframe(view[show_cols].head(50), use_container_width=True, hide_index=True)

    st.download_button(
        "📥  Download Full Predictions CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    "<b>BITS Pilani WILP M.Tech (AIML)</b> · Machine Learning (AIMLZG565) · Assignment 2<br>"
    'Developed by <b>Adithya S</b>'
    ' (<a href="https://github.com/AdiS-2105">@AdiS-2105</a>)'
    " · Seed <code>2002</code>"
    "</div>",
    unsafe_allow_html=True,
)

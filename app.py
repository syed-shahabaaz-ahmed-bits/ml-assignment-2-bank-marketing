"""Streamlit demo: live evaluation of Bank Marketing classifiers on uploaded test CSV."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.preprocess import (
    MODEL_FILENAMES,
    compute_classification_metrics,
    prepare_xy,
    validate_uploaded_frame,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_BLURBS = {
    "Logistic Regression": "Linear baseline with class_weight=balanced (strong recall).",
    "Decision Tree": "Single tree; easy to overfit on this imbalanced data.",
    "K-Nearest Neighbors": "Distance-based; often conservative on the minority class.",
    "Naive Bayes": "Fast probabilistic baseline after one-hot encoding.",
    "Random Forest": "Ensemble model; usually best overall AUC / F1 / MCC here.",
}

st.set_page_config(
    page_title="Bank Marketing Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1120px;
      }
      [data-testid="stSidebar"] { background: #f7f9fb; }
      [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
      h1 {
        font-size: 1.85rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem !important;
        color: #14212b;
      }
      h2, h3 {
        color: #1f2d3a !important;
        margin-top: 0.35rem !important;
      }
      div[data-testid="stMetric"] {
        background: #f4f7f9;
        border: 1px solid #e3ebf0;
        border-radius: 10px;
        padding: 0.85rem 0.9rem 0.7rem 0.9rem;
      }
      div[data-testid="stMetricValue"] { font-size: 1.28rem; color: #0f1c24; }
      div[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #5b6b75; }
      .subtitle {
        color: #5c6b73;
        font-size: 0.98rem;
        line-height: 1.5;
        margin: 0 0 1.35rem 0;
      }
      .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.25rem 0 1.4rem 0;
      }
      .meta-chip {
        background: #eef4f7;
        color: #243542;
        border: 1px solid #d7e3ea;
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.82rem;
      }
      .section-gap { height: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


with st.sidebar:
    st.markdown("### Controls")
    st.caption("Upload labeled test data and pick a trained model.")
    st.write("")
    uploaded = st.file_uploader("Test CSV (features + `y`)", type=["csv"])
    st.write("")
    model_name = st.selectbox("Model", list(MODEL_FILENAMES.keys()))
    st.caption(MODEL_BLURBS.get(model_name, ""))
    st.write("")
    st.markdown("---")
    st.caption("Recommended file: `data/test_data.csv` from this repository.")

st.title("Bank Term Deposit Subscription Predictor")
st.markdown(
    '<p class="subtitle">Binary classification on the UCI Bank Marketing dataset. '
    "Models were trained without <code>duration</code> (call-length leakage). "
    "All six metrics below are computed live on your uploaded CSV.</p>",
    unsafe_allow_html=True,
)

if uploaded is None:
    st.info(
        "Upload a labeled test CSV from the sidebar to view live metrics, "
        "a compact confusion matrix, and the classification report."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

ref_path = ROOT / "data" / "test_data.csv"
if not ref_path.exists():
    st.error(
        f"Reference schema file not found: {ref_path}. "
        "Run `PYTHONPATH=. python model/train.py`."
    )
    st.stop()

ref_cols = [c for c in pd.read_csv(ref_path, nrows=0).columns if c != "y"]
missing = validate_uploaded_frame(df, required_feature_cols=ref_cols, target_col="y")

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

model_path = MODEL_DIR / MODEL_FILENAMES[model_name]
pipe = load_model(model_path)
if pipe is None:
    st.error(
        f"Model file not found: {model_path.name}. "
        "Run `PYTHONPATH=. python model/train.py`."
    )
    st.stop()

try:
    X, y_true = prepare_xy(df)
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

n_rows = len(y_true)
pos_rate = float(np.mean(y_true)) * 100
st.markdown(
    f"""
    <div class="meta-row">
      <span class="meta-chip">Model: {model_name}</span>
      <span class="meta-chip">Rows evaluated: {n_rows:,}</span>
      <span class="meta-chip">Positive rate (yes): {pos_rate:.1f}%</span>
      <span class="meta-chip">Features: {X.shape[1]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Live evaluation metrics")
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
metric_cols = st.columns(6, gap="medium")
labels = [
    ("Accuracy", "accuracy"),
    ("AUC", "auc"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("MCC", "mcc"),
]
for col, (label, key) in zip(metric_cols, labels):
    val = metrics[key]
    col.metric(label, "N/A" if val is None else f"{val:.4f}")
if metrics["auc"] is None:
    st.warning("AUC is N/A because the uploaded file contains only one class.")

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("Confusion matrix")
    st.caption("Actual rows vs predicted columns (`no` / `yes`).")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.15, 2.55), dpi=120)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        square=True,
        linewidths=0.6,
        linecolor="#d5dee5",
        xticklabels=["no", "yes"],
        yticklabels=["no", "yes"],
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.45)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    st.caption("Per-class precision, recall, F1, and support.")
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        )
    ).T
    display_cols = [
        c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns
    ]
    st.dataframe(
        report_df[display_cols].style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }
        ),
        use_container_width=True,
        height=280,
    )

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

# Theme-safe CSS: no forced dark text colors (breaks Streamlit dark mode).
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1120px;
      }
      div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 0.85rem 0.9rem 0.7rem 0.9rem;
      }
      div[data-testid="stMetricValue"] { font-size: 1.28rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.82rem; opacity: 0.85; }
      .section-gap { height: 0.75rem; }
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
    st.header("Controls")
    st.caption("Upload labeled test data and choose a trained model.")
    uploaded = st.file_uploader(
        "Test CSV (must include target column y)",
        type=["csv"],
    )
    model_name = st.selectbox("Model", list(MODEL_FILENAMES.keys()))
    st.caption(MODEL_BLURBS.get(model_name, ""))
    st.divider()
    st.caption("Recommended upload: data/test_data.csv from this repository.")

st.title("Bank Term Deposit Subscription Predictor")
st.write(
    "Binary classification on the UCI Bank Marketing dataset. "
    "Models were trained without the duration feature (call-length leakage). "
    "All six metrics below are computed live on your uploaded CSV."
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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", model_name.split()[0] if " " in model_name else model_name)
m2.metric("Rows evaluated", f"{n_rows:,}")
m3.metric("Positive rate (yes)", f"{pos_rate:.1f}%")
m4.metric("Features", f"{X.shape[1]}")
# Show full model name under the truncated metric if needed
st.caption(f"Selected model: {model_name}")

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.subheader("Live evaluation metrics")
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
st.divider()

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("Confusion matrix")
    st.caption("Actual rows vs predicted columns (no / yes).")
    cm = confusion_matrix(y_true, y_pred)
    # Light figure so annotations stay readable in both app themes
    fig, ax = plt.subplots(figsize=(3.15, 2.55), dpi=120, facecolor="white")
    ax.set_facecolor("white")
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
        annot_kws={"size": 11, "color": "#111111"},
    )
    ax.set_xlabel("Predicted", fontsize=9, color="#111111")
    ax.set_ylabel("Actual", fontsize=9, color="#111111")
    ax.tick_params(labelsize=9, colors="#111111")
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
    # Plain dataframe (no Styler) for reliable contrast in dark/light themes
    pretty = report_df[display_cols].copy()
    for col_name in ["precision", "recall", "f1-score"]:
        if col_name in pretty.columns:
            pretty[col_name] = pretty[col_name].map(lambda x: f"{x:.3f}")
    if "support" in pretty.columns:
        pretty["support"] = pretty["support"].map(lambda x: f"{x:.0f}")
    st.dataframe(pretty, use_container_width=True, height=280)

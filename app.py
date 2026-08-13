"""Streamlit demo: live evaluation of Bank Marketing classifiers on uploaded test CSV."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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

st.set_page_config(
    page_title="Bank Marketing Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
      h1 { font-size: 1.75rem !important; margin-bottom: 0.25rem !important; }
      .subtitle { color: #5c6b73; font-size: 0.95rem; margin-bottom: 1rem; }
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
    uploaded = st.file_uploader("Test CSV (features + `y`)", type=["csv"])
    model_name = st.selectbox("Model", list(MODEL_FILENAMES.keys()))
    st.caption("Tip: use `data/test_data.csv` from the repo.")

st.title("Bank Term Deposit Subscription Predictor")
st.markdown(
    '<p class="subtitle">UCI Bank Marketing binary classifier. '
    "Models were trained without <code>duration</code> to avoid leakage. "
    "Metrics are computed live on your upload.</p>",
    unsafe_allow_html=True,
)

if uploaded is None:
    st.info("Upload a labeled test CSV from the sidebar to see metrics, confusion matrix, and report.")
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

st.subheader("Live evaluation metrics")
metric_cols = st.columns(6)
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

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=120)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="#d0d7de",
        xticklabels=["no", "yes"],
        yticklabels=["no", "yes"],
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        )
    ).T
    display_cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns]
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
        height=260,
    )

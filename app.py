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

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")

st.title("Bank Term Deposit Subscription Predictor")
st.markdown(
    """
Predict whether a client will subscribe to a **term deposit** after a telemarketing campaign
(UCI Bank Marketing). Models were trained **without** the `duration` feature to avoid leakage.
Upload a labeled test CSV to see live metrics.
"""
)


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


uploaded = st.file_uploader("Upload test CSV (features + column `y`)", type=["csv"])
model_name = st.selectbox("Select model", list(MODEL_FILENAMES.keys()))

if uploaded is None:
    st.info("Upload `data/test_data.csv` from this repo to reproduce assignment results.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

ref_path = ROOT / "data" / "test_data.csv"
if not ref_path.exists():
    st.error(f"Reference schema file not found: {ref_path}. Run `PYTHONPATH=. python model/train.py`.")
    st.stop()

ref_cols = [c for c in pd.read_csv(ref_path, nrows=0).columns if c != "y"]
missing = validate_uploaded_frame(df, required_feature_cols=ref_cols, target_col="y")

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

model_path = MODEL_DIR / MODEL_FILENAMES[model_name]
pipe = load_model(model_path)
if pipe is None:
    st.error(f"Model file not found: {model_path.name}. Run `PYTHONPATH=. python model/train.py`.")
    st.stop()

try:
    X, y_true = prepare_xy(df)
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.subheader("Evaluation metrics (live)")
cols = st.columns(6)
labels = [
    ("Accuracy", "accuracy"),
    ("AUC", "auc"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("MCC", "mcc"),
]
for col, (label, key) in zip(cols, labels):
    val = metrics[key]
    col.metric(label, "N/A" if val is None else f"{val:.4f}")
if metrics["auc"] is None:
    st.warning("AUC is N/A because the uploaded file contains only one class.")

st.subheader("Confusion matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["no", "yes"], yticklabels=["no", "yes"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification report")
st.text(classification_report(y_true, y_pred, target_names=["no", "yes"], zero_division=0))

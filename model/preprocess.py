"""Shared loading, preprocessing, and metric helpers for Bank Marketing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET_COL = "y"
DROP_COLS = ["duration"]

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def load_raw_bank(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    if TARGET_COL not in data.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")
    y = data[TARGET_COL].map({"yes": 1, "no": 0, 1: 1, 0: 0, "1": 1, "0": 0})
    if y.isna().any():
        raise ValueError("Target contains values other than yes/no or 0/1")
    X = data.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in data.columns])
    return X, y.astype(int)


def get_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def make_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def compute_classification_metrics(
    y_true, y_pred, y_proba=None
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": None,
    }
    if y_proba is not None:
        y_true_arr = np.asarray(y_true)
        if len(np.unique(y_true_arr)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def validate_uploaded_frame(
    df: pd.DataFrame,
    required_feature_cols: Iterable[str],
    target_col: str = TARGET_COL,
) -> list[str]:
    required = list(required_feature_cols) + [target_col]
    return [c for c in required if c not in df.columns]

from pathlib import Path

import numpy as np
import pandas as pd

from model.preprocess import (
    compute_classification_metrics,
    load_raw_bank,
    prepare_xy,
    validate_uploaded_frame,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "bank-full.csv"


def test_load_raw_bank_has_expected_columns():
    df = load_raw_bank(DATA)
    assert "y" in df.columns
    assert "duration" in df.columns
    assert len(df) >= 500


def test_prepare_xy_drops_duration_and_encodes_target():
    df = load_raw_bank(DATA)
    X, y = prepare_xy(df)
    assert "duration" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert X.shape[1] >= 12


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4])
    m = compute_classification_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "auc", "precision", "recall", "f1", "mcc"]:
        assert key in m


def test_validate_uploaded_frame_reports_missing():
    df = pd.DataFrame({"age": [1], "y": ["yes"]})
    missing = validate_uploaded_frame(df, required_feature_cols=["age", "job"], target_col="y")
    assert "job" in missing

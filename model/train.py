"""Train all Bank Marketing classifiers and persist pipelines + test CSV."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from model.preprocess import (
    MODEL_FILENAMES,
    build_models,
    build_preprocessor,
    compute_classification_metrics,
    get_feature_columns,
    load_raw_bank,
    make_pipeline,
    prepare_xy,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-full.csv"
TEST_OUT = ROOT / "data" / "test_data.csv"
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    df = load_raw_bank(DATA_PATH)
    X, y = prepare_xy(df)
    numeric_cols, categorical_cols = get_feature_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Export test data with original-style target labels for Streamlit demos
    test_export = X_test.copy()
    test_export["y"] = y_test.map({1: "yes", 0: "no"}).values
    test_export.to_csv(TEST_OUT, index=False)

    rows = []
    for name, estimator in build_models().items():
        pipe = make_pipeline(clone(preprocessor), estimator)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        out_path = MODEL_DIR / MODEL_FILENAMES[name]
        joblib.dump(pipe, out_path)
        rows.append({"model": name, **metrics, "path": str(out_path.name)})
        print(f"Saved {out_path}")

    results = pd.DataFrame(rows)
    print("\n=== Test metrics ===")
    print(results.to_string(index=False))
    results.to_csv(MODEL_DIR / "last_test_metrics.csv", index=False)


if __name__ == "__main__":
    main()

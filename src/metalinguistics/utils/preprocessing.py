"""metalinguistics.utils.preprocessing
------------------------------------------------
Utility helpers for cleaning feature DataFrames before feeding them to
scikit-learn models.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["clean_features"]


def _coerce_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map textual booleans and 0/1 strings to integers 0/1 in-place."""
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].str.contains(r"^(True|False|0|1)$", regex=True).any():
                df[col] = (
                    df[col]
                    .replace({"True": 1, "False": 0, "0": 0, "1": 1, True: 1, False: 0})
                    .astype("int32", errors="ignore")
                )
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature matrix suitable for scikit-learn.

    Steps
    -----
    1. Coerce any textual boolean columns to numeric 0/1.
    2. One-hot encode remaining object (categorical) columns.
    3. Ensure every column is numeric; coerce invalid strings to NaN then fill with 0.
    """
    df = df.copy()

    # 1. Boolean coercion
    df = _coerce_boolean_columns(df)

    # 2. One-hot encode categorical leftovers
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        df = pd.get_dummies(df, columns=object_cols, dummy_na=False)

    # 3. Force numeric matrix
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df 
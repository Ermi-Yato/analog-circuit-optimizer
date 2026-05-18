"""
Dataset preprocessor.

Handles validation, scaling, and statistical summary of generated datasets.
Used by the training pipeline (Phase 4) and the dataset view (GUI).
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate(df: pd.DataFrame) -> list[str]:
    """
    Check a dataset DataFrame for common data quality issues.

    Returns a list of warning strings. An empty list means the data is clean.
    Does NOT raise — the caller decides whether to abort or proceed.
    """
    issues = []

    if df.empty:
        issues.append("Dataset is empty.")
        return issues

    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values found in columns: {nan_cols}")

    inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        issues.append(f"Infinite values found in columns: {inf_cols}")

    zero_var = [c for c in df.columns if df[c].std() == 0]
    if zero_var:
        issues.append(f"Zero-variance columns (constant values): {zero_var}")

    dupe_count = df.duplicated().sum()
    if dupe_count > 0:
        issues.append(f"{dupe_count} duplicate rows found.")

    return issues


def stats(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for each column:
      {col: {"min", "max", "mean", "std", "count"}}
    """
    result = {}
    for col in df.columns:
        s = df[col]
        result[col] = {
            "min":   float(s.min()),
            "max":   float(s.max()),
            "mean":  float(s.mean()),
            "std":   float(s.std()),
            "count": int(s.count()),
        }
    return result


def fit_transform(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features with StandardScaler and return arrays ready for training.

    Args:
        df:           Full dataset DataFrame.
        feature_cols: Column names used as ML input features (parameters).
        target_cols:  Column names used as ML output targets (metrics).

    Returns:
        X_scaled:  Feature matrix, shape (n_samples, n_features), scaled.
        y:         Target matrix, shape (n_samples, n_targets), unscaled.
        scaler:    Fitted StandardScaler (save alongside model for inference).
    """
    X = df[feature_cols].values
    y = df[target_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def load_csv(circuit_id: str, data_dir: str | None = None) -> pd.DataFrame:
    """
    Load the dataset CSV for a given circuit ID.

    Raises FileNotFoundError if the file doesn't exist.
    """
    if data_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "..", "..", "data")

    path = os.path.join(data_dir, f"{circuit_id}_dataset.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found for circuit '{circuit_id}'. "
            f"Expected: {path}\n"
            f"Run dataset generation first."
        )
    return pd.read_csv(path)

"""
Surrogate model training pipeline.

Handles the full train cycle:
  1. Load dataset CSV via preprocessor
  2. Validate data quality
  3. Train/test split
  4. Scale features (StandardScaler)
  5. Train RandomForest surrogate model
  6. Evaluate (R², MAE per metric)
  7. Save model + scaler to trained_models/<circuit_id>/
  8. Update circuit JSON "model" block with scores + timestamp

Usage:
    from core.models.trainer import train
    metrics = train("common_emitter_amplifier")
"""
import os
import json
import datetime

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import registry.circuit_registry as reg
from core.dataset.preprocessor import load_csv, validate, fit_transform
from core.models.random_forest import RandomForestModel


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "trained_models")


def train(
    circuit_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full training pipeline for a circuit's surrogate model.

    Args:
        circuit_id:   Registry circuit ID.
        test_size:    Fraction of data held out for evaluation (default 0.2).
        random_state: Seed for train/test split reproducibility.
        verbose:      Print progress and evaluation results.

    Returns:
        metrics_dict: {
            "r2":  {metric_name: float, ...},
            "mae": {metric_name: float, ...},
            "n_train": int,
            "n_test":  int,
        }

    Raises:
        FileNotFoundError: Dataset CSV not found — run dataset generation first.
        ValueError:        Data quality issues found by preprocessor.validate().
    """
    circuit = reg.get(circuit_id)
    param_names  = [p["name"] for p in circuit["parameters"]]
    metric_names = [m["name"] for m in circuit["metrics"]]

    if verbose:
        print(f"\nTraining surrogate model: {circuit['name']}")
        print(f"  Features : {param_names}")
        print(f"  Targets  : {metric_names}")

    # 1. Load dataset
    df = load_csv(circuit_id)

    # 2. Validate data quality
    issues = validate(df)
    if issues:
        raise ValueError(
            f"Data quality issues in dataset for '{circuit_id}':\n"
            + "\n".join(f"  - {i}" for i in issues)
        )

    if verbose:
        print(f"  Dataset  : {len(df)} rows (clean)")

    # 3. Scale features, extract arrays
    X_scaled, y, scaler = fit_transform(df, param_names, metric_names)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print(f"  Split    : {len(X_train)} train / {len(X_test)} test")

    # 5. Train
    model = RandomForestModel()
    model.fit(X_train, y_train)

    # 6. Evaluate
    metrics = evaluate(model, X_test, y_test, metric_names)

    if verbose:
        print(f"\n  Results:")
        for name in metric_names:
            print(
                f"    {name:30s}  R²={metrics['r2'][name]:.4f}  "
                f"MAE={metrics['mae'][name]:.4f}"
            )

    # 7. Save model and scaler
    out_dir = os.path.join(_MODELS_DIR, circuit_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path  = os.path.join(out_dir, "circuit_model.pkl")
    scaler_path = os.path.join(out_dir, "feature_scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    if verbose:
        print(f"\n  Saved model  : {model_path}")
        print(f"  Saved scaler : {scaler_path}\n")

    # 8. Update circuit JSON "model" block
    _update_model_block(circuit_id, model_path, scaler_path, metrics, len(df))

    return metrics


def evaluate(
    model: RandomForestModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric_names: list[str],
) -> dict:
    """
    Evaluate a trained model on held-out test data.

    Returns:
        {
            "r2":     {metric_name: float},
            "mae":    {metric_name: float},
            "n_train": 0,   # not set here — caller fills if needed
            "n_test":  int,
        }
    """
    y_pred = model.predict(X_test)

    # y_test may be 1-D for single-target circuits — normalise to 2-D
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    r2_scores  = {}
    mae_scores = {}
    for i, name in enumerate(metric_names):
        r2_scores[name]  = float(r2_score(y_test[:, i], y_pred[:, i]))
        mae_scores[name] = float(mean_absolute_error(y_test[:, i], y_pred[:, i]))

    return {
        "r2":     r2_scores,
        "mae":    mae_scores,
        "n_train": 0,
        "n_test":  len(X_test),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_model_block(
    circuit_id: str,
    model_path: str,
    scaler_path: str,
    metrics: dict,
    n_samples: int,
) -> None:
    """
    Write the "model" block back into the circuit's JSON file.

    Paths stored as forward-slash relative paths from project root.
    """
    circuits_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "registry", "circuits"
    )
    json_path = os.path.join(circuits_dir, f"{circuit_id}.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Store paths relative to project root with forward slashes
    rel_model  = os.path.relpath(model_path,  _PROJECT_ROOT).replace("\\", "/")
    rel_scaler = os.path.relpath(scaler_path, _PROJECT_ROOT).replace("\\", "/")

    data["model"] = {
        "surrogate_path": rel_model,
        "scaler_path":    rel_scaler,
        "trained_on":     datetime.date.today().isoformat(),
        "samples":        n_samples,
        "r2_scores":      metrics["r2"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Reload registry cache so model_exists() reflects the new state
    reg.load_all()

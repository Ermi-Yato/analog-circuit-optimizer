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
from core.dataset.preprocessor import load_csv, validate, fit_transform, add_derived_features
from core.models.random_forest import RandomForestModel
from core.models.mlp import MLPModel


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "trained_models")


def train(
    circuit_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    model_type: str = "random_forest",
) -> dict:
    """
    Full training pipeline for a circuit's surrogate model.

    Args:
        circuit_id:   Registry circuit ID.
        test_size:    Fraction of data held out for evaluation (default 0.2).
        random_state: Seed for train/test split reproducibility.
        verbose:      Print progress and evaluation results.
        model_type:   "random_forest" | "mlp" | "auto"
                      "auto" trains both and keeps whichever has higher mean R².

    Returns:
        metrics_dict: {
            "r2":         {metric_name: float, ...},
            "mae":        {metric_name: float, ...},
            "n_train":    int,
            "n_test":     int,
            "model_type": str,   # actual type used ("random_forest" or "mlp")
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
        print(f"  Features   : {param_names}")
        print(f"  Targets    : {metric_names}")
        print(f"  Model type : {model_type}")

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
        print(f"  Dataset    : {len(df)} rows (clean)")

    # 2b. Add physics-informed derived features
    df, derived_features = add_derived_features(df, circuit_id)
    if derived_features:
        param_names = param_names + derived_features
        if verbose:
            print(f"  Derived    : {len(derived_features)} physics-based features")

    # 3a. Log-transform log-scale parameters before StandardScaler.
    #     For circuits like Sallen-Key where fc = 1/(2pi*sqrt(R1*R2*C1*C2)),
    #     the relationship is linear in log-space, so log10 of log-scale
    #     parameters dramatically improves model accuracy.
    log_params = [p["name"] for p in circuit["parameters"] if p.get("scale") == "log"]
    if log_params:
        df = df.copy()
        for col in log_params:
            if col in df.columns:
                df[col] = np.log10(df[col].clip(lower=1e-300))
        if verbose:
            print(f"  Log10 (params)  : {log_params}")

    # 3b. Log-transform log-scale metrics.
    #     Metrics spanning many decades (e.g. Cutoff_Freq_Hz: 1Hz–500kHz) must be
    #     trained in log-space so the model is uniformly accurate at all magnitudes.
    #     Without this, MAE=±1784Hz is fine at 100kHz but 178% error at 1kHz.
    log_metrics = [m["name"] for m in circuit["metrics"] if m.get("scale") == "log"]
    if log_metrics:
        df = df.copy() if not log_params else df  # already copied above if log_params
        for col in log_metrics:
            if col in df.columns:
                df[col] = np.log10(df[col].clip(lower=1e-300))
        if verbose:
            print(f"  Log10 (metrics) : {log_metrics}")

    # 3c. Scale features, extract arrays
    X_scaled, y, scaler = fit_transform(df, param_names, metric_names)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print(f"  Split      : {len(X_train)} train / {len(X_test)} test")

    # 5. Train — single model or auto-select best

    model, metrics, actual_type = _train_and_select(
        model_type, X_train, X_test, y_train, y_test, metric_names, verbose
    )

    if verbose:
        print(f"\n  Results ({actual_type}):")
        for name in metric_names:
            print(
                f"    {name:30s}  R²={metrics['r2'][name]:.4f}  "
                f"MAE={metrics['mae'][name]:.4f}"
            )

    # 6. Save model and scaler
    out_dir = os.path.join(_MODELS_DIR, circuit_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path  = os.path.join(out_dir, "circuit_model.pkl")
    scaler_path = os.path.join(out_dir, "feature_scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    if verbose:
        print(f"\n  Saved model  : {model_path}")
        print(f"  Saved scaler : {scaler_path}\n")

    metrics["n_train"]    = len(X_train)
    metrics["model_type"] = actual_type

    # 7. Update circuit JSON "model" block
    _update_model_block(circuit_id, model_path, scaler_path, metrics, len(df), actual_type)

    return metrics


def _train_and_select(
    model_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    metric_names: list[str],
    verbose: bool,
):
    """
    Train one or both model types and return (model, metrics, actual_type).

    For "auto": trains RF and MLP, keeps the one with higher mean R².
    """
    def _fit_eval(ModelClass, label):
        m = ModelClass()
        m.fit(X_train, y_train)
        ev = evaluate(m, X_test, y_test, metric_names)
        mean_r2 = float(np.mean(list(ev["r2"].values())))
        if verbose:
            print(f"  [{label}] mean R² = {mean_r2:.4f}")
        return m, ev, mean_r2

    if model_type == "mlp":
        model, metrics, _ = _fit_eval(MLPModel, "MLP")
        return model, metrics, "mlp"

    if model_type == "random_forest":
        model, metrics, _ = _fit_eval(RandomForestModel, "RandomForest")
        return model, metrics, "random_forest"

    # "auto" — train both, pick best mean R²
    if verbose:
        print("  Auto-select: training RandomForest and MLP...")
    rf_model,  rf_metrics,  rf_r2  = _fit_eval(RandomForestModel, "RandomForest")
    mlp_model, mlp_metrics, mlp_r2 = _fit_eval(MLPModel, "MLP")

    if mlp_r2 > rf_r2:
        if verbose:
            print(f"  -> Chose MLP  (mean R2 {mlp_r2:.4f} > {rf_r2:.4f})")
        return mlp_model, mlp_metrics, "mlp"
    else:
        if verbose:
            print(f"  -> Chose RF   (mean R2 {rf_r2:.4f} >= {mlp_r2:.4f})")
        return rf_model, rf_metrics, "random_forest"


def evaluate(
    model,
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
    model_type: str = "random_forest",
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

    log_metric_names = [m["name"] for m in data.get("metrics", []) if m.get("scale") == "log"]
    data["model"] = {
        "surrogate_path":  rel_model,
        "scaler_path":     rel_scaler,
        "trained_on":      datetime.date.today().isoformat(),
        "samples":         n_samples,
        "r2_scores":       metrics["r2"],
        "model_type":      model_type,
        "log_metrics":     log_metric_names,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Reload registry cache so model_exists() reflects the new state
    reg.load_all()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def load_model(circuit_id: str):
    """
    Load the trained surrogate model for a circuit.

    Respects the "model_type" field written by train() so that MLP and
    RandomForest models are loaded with the correct class.

    Returns:
        model instance (BaseModel subclass), already loaded from disk.

    Raises:
        FileNotFoundError: if model PKL does not exist.
        KeyError:          if circuit has no trained model block.
    """
    circuit     = reg.get(circuit_id)
    model_block = circuit.get("model")
    if not model_block:
        raise KeyError(f"Circuit '{circuit_id}' has no trained model. Train it first.")

    model_path = os.path.join(_PROJECT_ROOT, model_block["surrogate_path"])
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_type = model_block.get("model_type", "random_forest")
    if model_type == "mlp":
        model = MLPModel()
    else:
        model = RandomForestModel()

    model.load(model_path)
    return model

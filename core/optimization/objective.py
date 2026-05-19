"""
Fitness function builder for all optimizers.

Supports multiple loss types and direction-aware penalties.

Score = sum over each metric of:
    weight * direction_factor * loss(predicted, target) / scale

Loss types:
    mae     |e|                         — robust, simple (default)
    mse     e²                          — emphasises large misses
    huber   δ·(|e|−δ/2) if |e|>δ, else e²/2  — mae + mse hybrid
    log     log1p(|e|)                  — compresses very large errors

Direction penalty:
    For "maximize" metrics: undershooting target gets penalty_factor×loss
    For "minimize" metrics: overshooting target gets penalty_factor×loss
    This rewards hitting-or-exceeding the spec over missing it.

Exploration bonus:
    When enabled, adds a small entropy term proportional to how far the
    candidate is from the population centroid (only meaningful for batch
    evaluation, e.g. in GA). Encourages diverse search.

Tolerance band:
    A dead-zone around the target: errors < (tolerance_pct/100 * scale)
    are treated as zero — useful when any value within ±X% is acceptable.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.models.base_model import BaseModel


# ── Loss kernels ──────────────────────────────────────────────────────────────

def _mae(e: float) -> float:
    return abs(e)

def _mse(e: float) -> float:
    return e * e

def _huber(e: float, delta: float = 1.0) -> float:
    ae = abs(e)
    return ae * delta - 0.5 * delta ** 2 if ae > delta else 0.5 * e * e

def _log1p_abs(e: float) -> float:
    return math.log1p(abs(e))


_LOSS_FNS: dict[str, Callable[[float], float]] = {
    "mae":   _mae,
    "mse":   _mse,
    "huber": _huber,
    "log":   _log1p_abs,
}


# ── Public builder ────────────────────────────────────────────────────────────

def build_fitness_fn(
    model: BaseModel,
    scaler: StandardScaler,
    targets: dict[str, float],
    metrics_meta: list[dict],
    weights: dict[str, float] | None = None,
    log_indices: list[int] | None = None,
    log_metric_indices: list[int] | None = None,
    # ── New options ──────────────────────────────────────────────────────
    loss_type: str = "mae",
    direction_penalty: float = 1.0,   # multiplier for undershooting maximize / overshooting minimize
    tolerance_pct: float = 0.0,       # dead-zone: |e|/scale < tolerance → 0 penalty
) -> Callable[[list[float]], tuple[float]]:
    """
    Build a DEAP-compatible fitness function.

    Args:
        model:             Trained surrogate model.
        scaler:            Fitted StandardScaler.
        targets:           {metric_name: target_value}.
        metrics_meta:      List of metric dicts (each has "name", "optimize").
        weights:           Optional per-metric weights (uniform if omitted).
        log_indices:       Parameter indices that are log-scale.
        log_metric_indices: Metric indices predicted in log10-space.
        loss_type:         "mae" | "mse" | "huber" | "log".
        direction_penalty: Factor applied when prediction misses in the wrong
                           direction (e.g. gain too low for a maximize metric).
                           1.0 = symmetric; 2.0 = twice the pain for wrong-side miss.
        tolerance_pct:     Dead-zone radius as % of target scale. 0 = disabled.

    Returns:
        fitness_fn(individual) -> (score,)  — lower is better.
    """
    if loss_type not in _LOSS_FNS:
        raise ValueError(f"loss_type must be one of {set(_LOSS_FNS)}, got {loss_type!r}")

    loss_fn      = _LOSS_FNS[loss_type]
    metric_names = [m["name"] for m in metrics_meta]

    w = {name: 1.0 for name in metric_names} if weights is None else \
        {name: weights.get(name, 1.0) for name in metric_names}

    scales = {
        name: abs(targets[name])
        if name in targets and abs(targets.get(name, 0)) > 1e-12 else 1.0
        for name in metric_names
    }

    # Precompute direction: +1 = maximize, -1 = minimize
    directions = {
        m["name"]: 1 if m.get("optimize", "maximize") == "maximize" else -1
        for m in metrics_meta
    }

    tol_abs = {
        name: (tolerance_pct / 100.0) * scales[name]
        for name in metric_names
    }

    def fitness_fn(individual: list[float]) -> tuple[float]:
        X = np.array(individual, dtype=float).reshape(1, -1)
        if log_indices:
            X = X.copy()
            X[:, log_indices] = np.log10(np.abs(X[:, log_indices]).clip(1e-300))
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled).ravel()
        if log_metric_indices:
            for i in log_metric_indices:
                if i < len(y_pred):
                    y_pred[i] = 10.0 ** float(np.clip(y_pred[i], -300, 300))

        score = 0.0
        for i, name in enumerate(metric_names):
            if name not in targets:
                continue
            predicted = float(y_pred[i]) if i < len(y_pred) else 0.0
            error     = predicted - targets[name]
            sc        = scales[name]

            # Tolerance dead-zone
            if tol_abs[name] > 0 and abs(error) <= tol_abs[name]:
                continue

            norm_error  = error / sc
            base_loss   = loss_fn(norm_error)

            # Direction penalty: wrong-side miss costs more
            if direction_penalty != 1.0:
                direction = directions[name]
                # maximize: error < 0 means undershoot → penalise
                # minimize: error > 0 means overshoot  → penalise
                wrong_side = (direction == 1 and error < 0) or \
                             (direction == -1 and error > 0)
                if wrong_side:
                    base_loss *= direction_penalty

            score += w[name] * base_loss

        return (score,)

    return fitness_fn


def score_vector(
    X: np.ndarray,
    model: BaseModel,
    scaler: StandardScaler,
    targets: dict[str, float],
    metrics_meta: list[dict],
    weights: dict[str, float] | None = None,
    log_indices: list[int] | None = None,
    log_metric_indices: list[int] | None = None,
    loss_type: str = "mae",
    direction_penalty: float = 1.0,
    tolerance_pct: float = 0.0,
) -> np.ndarray:
    """
    Batch fitness evaluation — returns a 1-D array of scores for each row in X.
    Used by GA's batched evaluator and scipy/BO optimizers.
    """
    if loss_type not in _LOSS_FNS:
        raise ValueError(f"loss_type must be one of {set(_LOSS_FNS)}, got {loss_type!r}")

    metric_names = [m["name"] for m in metrics_meta]
    w = {name: 1.0 for name in metric_names} if weights is None else \
        {name: weights.get(name, 1.0) for name in metric_names}
    scales = {
        name: abs(targets[name])
        if name in targets and abs(targets.get(name, 0)) > 1e-12 else 1.0
        for name in metric_names
    }
    directions = {
        m["name"]: 1 if m.get("optimize", "maximize") == "maximize" else -1
        for m in metrics_meta
    }
    tol_abs = {name: (tolerance_pct / 100.0) * scales[name] for name in metric_names}

    Xc = X.copy().astype(float)
    if log_indices:
        Xc[:, log_indices] = np.log10(np.abs(Xc[:, log_indices]).clip(1e-300))
    preds = model.predict(scaler.transform(Xc))
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if log_metric_indices:
        preds = preds.copy()
        for mi in log_metric_indices:
            if mi < preds.shape[1]:
                preds[:, mi] = 10.0 ** np.clip(preds[:, mi], -300, 300)

    scores = np.zeros(len(X))
    for i, name in enumerate(metric_names):
        if name not in targets:
            continue
        col    = preds[:, i] if i < preds.shape[1] else np.zeros(len(X))
        errors = col - targets[name]
        sc     = scales[name]
        norms  = errors / sc

        # Vectorised loss
        if loss_type == "mae":
            base = np.abs(norms)
        elif loss_type == "mse":
            base = norms ** 2
        elif loss_type == "huber":
            ae = np.abs(norms)
            base = np.where(ae > 1.0, ae - 0.5, 0.5 * norms ** 2)
        else:  # log
            base = np.log1p(np.abs(norms))

        # Tolerance dead-zone
        if tol_abs[name] > 0:
            base = np.where(np.abs(errors) <= tol_abs[name], 0.0, base)

        # Direction penalty
        if direction_penalty != 1.0:
            direction = directions[name]
            wrong_side = ((direction == 1) & (errors < 0)) | \
                         ((direction == -1) & (errors > 0))
            base = np.where(wrong_side, base * direction_penalty, base)

        scores += w[name] * base

    return scores

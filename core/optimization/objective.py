"""
Fitness function builder for the GA optimizer.

The fitness function maps a candidate parameter vector to a scalar score
that DEAP tries to minimise. Lower = better.

Score = sum over each metric of:
    weight * |predicted - target| / target_scale

where target_scale normalises each metric so that a 10% miss on gain
and a 10% miss on bandwidth contribute equally regardless of units.

Metrics marked "maximize" contribute the same penalty as "minimize" —
both use absolute normalised error. The direction only matters when
interpreting "closer to target is better".
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.models.base_model import BaseModel


def build_fitness_fn(
    model: BaseModel,
    scaler: StandardScaler,
    targets: dict[str, float],
    metrics_meta: list[dict],
    weights: dict[str, float] | None = None,
) -> Callable[[list[float]], tuple[float]]:
    """
    Build a DEAP-compatible fitness function.

    Args:
        model:        Trained surrogate model (implements BaseModel).
        scaler:       Fitted StandardScaler for feature normalisation.
        targets:      {metric_name: target_value} — the desired specs.
        metrics_meta: List of metric dicts from circuit JSON
                      (each has "name", "optimize").
        weights:      Optional {metric_name: weight} — uniform if omitted.

    Returns:
        fitness_fn(individual) -> (score,)   ← DEAP expects a tuple

        Lower score = better candidate.
    """
    metric_names  = [m["name"] for m in metrics_meta]
    n_metrics     = len(metric_names)

    # Default weights: equal
    if weights is None:
        w = {name: 1.0 for name in metric_names}
    else:
        w = {name: weights.get(name, 1.0) for name in metric_names}

    # Pre-compute normalisation scales: use target value if non-zero, else 1.0
    scales = {
        name: abs(targets[name]) if name in targets and abs(targets.get(name, 0)) > 1e-12 else 1.0
        for name in metric_names
    }

    def fitness_fn(individual: list[float]) -> tuple[float]:
        X = np.array(individual, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled).ravel()  # shape (n_metrics,)

        score = 0.0
        for i, name in enumerate(metric_names):
            if name not in targets:
                continue
            predicted = float(y_pred[i]) if i < len(y_pred) else 0.0
            error     = abs(predicted - targets[name]) / scales[name]
            score    += w[name] * error

        return (score,)

    return fitness_fn

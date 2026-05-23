"""
Scipy-based optimizers for analog circuit parameter search.

Supports:
    differential_evolution  — population-based, excellent for multimodal landscapes
    dual_annealing          — global SA + local polishing, fast on smooth objectives

Both produce the same result_dict format as genetic_algorithm.optimize() so they
are drop-in replacements in the worker and UI.

result_dict keys:
    "best_params"    : {param_name: value, ...}
    "best_score"     : float
    "best_predicted" : {metric_name: value, ...}
    "history"        : [(iteration, best_score), ...]
    "population"     : [(params_dict, score, predicted_dict), ...]  — best 50
"""
from __future__ import annotations

import os

import joblib
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

import registry.circuit_registry as reg
from core.models.random_forest import RandomForestModel
from core.models.mlp import MLPModel
from core.optimization.objective import score_vector

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_ALGORITHMS = {"differential_evolution", "dual_annealing"}


def optimize(
    circuit_id: str,
    targets: dict[str, float],
    algorithm: str = "differential_evolution",
    # Differential Evolution settings
    de_popsize: int = 15,          # individuals per parameter dimension
    de_mutation: float = 0.7,      # mutation factor F [0, 2]
    de_recombination: float = 0.7, # crossover probability
    de_maxiter: int = 300,
    # Dual Annealing settings
    da_maxiter: int = 1000,
    da_initial_temp: float = 5230.0,
    da_restart_temp: float = 2e-5,
    da_visit: float = 2.62,
    da_accept: float = -5.0,
    # Fitness settings
    weights: dict[str, float] | None = None,
    loss_type: str = "mae",
    direction_penalty: float = 1.0,
    tolerance_pct: float = 0.0,
    progress_callback=None,
    seed: int | None = None,
) -> dict:
    """
    Run a scipy-based optimizer for a circuit.

    Args:
        circuit_id:        Registry circuit ID.
        targets:           {metric_name: target_value}.
        algorithm:         "differential_evolution" or "dual_annealing".
        progress_callback: Called as (iteration: int, best_score: float).
        seed:              Random seed for reproducibility.

    Returns:
        result_dict with same schema as genetic_algorithm.optimize().
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {_ALGORITHMS}.")

    circuit     = reg.get(circuit_id)
    param_defs  = circuit["parameters"]
    metric_defs = circuit["metrics"]
    param_names = [p["name"] for p in param_defs]

    # Load model + scaler
    model_block = circuit.get("model")
    if not model_block:
        raise FileNotFoundError(f"No trained model for '{circuit_id}'. Train the model first.")

    model_path  = os.path.join(_PROJECT_ROOT, model_block["surrogate_path"])
    scaler_path = os.path.join(_PROJECT_ROOT, model_block["scaler_path"])
    for p in (model_path, scaler_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Model file not found: {p}")

    model_type = model_block.get("model_type", "random_forest")
    model = MLPModel() if model_type == "mlp" else RandomForestModel()
    model.load(model_path)
    scaler = joblib.load(scaler_path)

    log_indices = [i for i, p in enumerate(param_defs) if p.get("scale") == "log"]
    log_metric_names = model_block.get("log_metrics", [])
    log_metric_indices = [i for i, m in enumerate(metric_defs) if m["name"] in log_metric_names]

    bounds = [(float(p["min"]), float(p["max"])) for p in param_defs]

    history: list[tuple[int, float]] = []
    iteration_counter = [0]
    best_so_far = [float("inf")]

    def _predict_linear(X: np.ndarray) -> np.ndarray:
        Xc = X.copy()
        if log_indices:
            Xc[:, log_indices] = np.log10(np.abs(Xc[:, log_indices]).clip(1e-300))
        # Compute physics-informed derived features if applicable
        if circuit_id == "common_emitter_amplifier":
            from core.dataset.preprocessor import compute_ce_derived_features
            Xc = compute_ce_derived_features(Xc)
        preds = model.predict(scaler.transform(Xc))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if log_metric_indices:
            preds = preds.copy()
            for mi in log_metric_indices:
                if mi < preds.shape[1]:
                    preds[:, mi] = 10.0 ** np.clip(preds[:, mi], -300, 300)
        return preds

    def objective(x: np.ndarray) -> float:
        scores = score_vector(
            x.reshape(1, -1), model, scaler, targets, metric_defs,
            weights=weights,
            log_indices=log_indices or None,
            log_metric_indices=log_metric_indices or None,
            loss_type=loss_type,
            direction_penalty=direction_penalty,
            tolerance_pct=tolerance_pct,
            circuit_id=circuit_id,
        )
        return float(scores[0])

    def _tick(x: np.ndarray, score: float):
        iteration_counter[0] += 1
        if score < best_so_far[0]:
            best_so_far[0] = score
        history.append((iteration_counter[0], best_so_far[0]))
        if progress_callback is not None:
            progress_callback(iteration_counter[0], best_so_far[0])

    def _cb_de(xk, convergence=None):
        """DE callback: (xk, convergence)"""
        _tick(xk, objective(xk))

    def _cb_da(x, f, context):
        """DA callback: (x, f, context) — f is already the objective value."""
        _tick(x, float(f))

    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(int(rng.integers(0, 2**31)))

    # ── Run the chosen algorithm ──────────────────────────────────────────────

    if algorithm == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            mutation=de_mutation,
            recombination=de_recombination,
            callback=_cb_de,
            tol=1e-7,
            polish=True,
            vectorized=False,
        )
        best_x = result.x
        best_score = float(result.fun)

    else:  # dual_annealing
        result = dual_annealing(
            objective,
            bounds,
            maxiter=da_maxiter,
            initial_temp=da_initial_temp,
            restart_temp_ratio=da_restart_temp,
            visit=da_visit,
            accept=da_accept,
            callback=_cb_da,
        )
        best_x = result.x
        best_score = float(result.fun)

    # ── Build output ─────────────────────────────────────────────────────────

    best_params = {name: float(best_x[i]) for i, name in enumerate(param_names)}

    X_best = best_x.reshape(1, -1)
    y_best = _predict_linear(X_best).ravel()
    best_predicted = {
        m["name"]: float(y_best[i]) if i < len(y_best) else float("nan")
        for i, m in enumerate(metric_defs)
    }

    # Approximate population: sample neighbourhood of best + random spread
    # (scipy doesn't expose a final population — we sample 50 nearby points)
    rng2 = np.random.default_rng(seed)
    sigmas = np.array([
        0.05 * (float(p["max"]) - float(p["min"])) for p in param_defs
    ])
    pop_X = np.clip(
        best_x + rng2.normal(0, sigmas, size=(50, len(param_defs))),
        [b[0] for b in bounds],
        [b[1] for b in bounds],
    )
    pop_X[0] = best_x  # Ensure best is always first

    pop_preds = _predict_linear(pop_X)
    population_out = []
    for i in range(len(pop_X)):
        score_i = objective(pop_X[i])
        params_i = {name: float(pop_X[i, j]) for j, name in enumerate(param_names)}
        pred_i   = {m["name"]: float(pop_preds[i, j]) if j < pop_preds.shape[1] else float("nan")
                    for j, m in enumerate(metric_defs)}
        population_out.append((params_i, score_i, pred_i))

    population_out.sort(key=lambda t: t[1])

    return {
        "best_params":    best_params,
        "best_score":     best_score,
        "best_predicted": best_predicted,
        "history":        history,
        "population":     population_out,
    }

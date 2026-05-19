"""
Bayesian Optimization using Optuna (TPE sampler).

Uses Tree-structured Parzen Estimator (TPE) to build a probabilistic model of
the objective surface, balancing exploration vs exploitation via an acquisition
function. Typically needs far fewer evaluations than GA/DE to converge on
well-behaved surrogate models.

Public API:
    optimize(circuit_id, targets, ...) -> result_dict

result_dict schema is identical to genetic_algorithm.optimize().
"""
from __future__ import annotations

import os
import logging

import joblib
import numpy as np

import registry.circuit_registry as reg
from core.models.random_forest import RandomForestModel
from core.models.mlp import MLPModel
from core.optimization.objective import score_vector

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Silence optuna's verbose INFO logs — only show warnings+
logging.getLogger("optuna").setLevel(logging.WARNING)


def optimize(
    circuit_id: str,
    targets: dict[str, float],
    n_trials: int = 300,
    sampler: str = "tpe",          # "tpe" | "cmaes" | "random"
    n_startup_trials: int = 20,    # random exploration before TPE/CMA-ES kicks in
    weights: dict[str, float] | None = None,
    loss_type: str = "mae",
    direction_penalty: float = 1.0,
    tolerance_pct: float = 0.0,
    progress_callback=None,
    seed: int | None = None,
) -> dict:
    """
    Run Bayesian optimization for a circuit.

    Args:
        circuit_id:        Registry circuit ID.
        targets:           {metric_name: target_value}.
        n_trials:          Total evaluations of the surrogate model.
        sampler:           Optuna sampler — "tpe", "cmaes", or "random".
        n_startup_trials:  Trials using random sampling before TPE/CMA-ES starts.
        progress_callback: Called as (trial: int, best_score: float).
        seed:              Random seed.

    Returns:
        result_dict — same schema as genetic_algorithm.optimize().
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna is required for Bayesian optimization.\n"
            "Install it with:  pip install optuna"
        )

    circuit      = reg.get(circuit_id)
    param_defs   = circuit["parameters"]
    metric_defs  = circuit["metrics"]
    param_names  = [p["name"] for p in param_defs]

    model_block = circuit.get("model")
    if not model_block:
        raise FileNotFoundError(f"No trained model for '{circuit_id}'. Train first.")

    model_path  = os.path.join(_PROJECT_ROOT, model_block["surrogate_path"])
    scaler_path = os.path.join(_PROJECT_ROOT, model_block["scaler_path"])
    for p in (model_path, scaler_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Model file not found: {p}")

    model_type = model_block.get("model_type", "random_forest")
    model = MLPModel() if model_type == "mlp" else RandomForestModel()
    model.load(model_path)
    scaler = joblib.load(scaler_path)

    log_indices        = [i for i, p in enumerate(param_defs) if p.get("scale") == "log"]
    log_metric_names   = model_block.get("log_metrics", [])
    log_metric_indices = [i for i, m in enumerate(metric_defs) if m["name"] in log_metric_names]

    # ── Optuna sampler ────────────────────────────────────────────────────────
    if sampler == "tpe":
        _sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            seed=seed,
        )
    elif sampler == "cmaes":
        _sampler = optuna.samplers.CmaEsSampler(
            n_startup_trials=n_startup_trials,
            seed=seed,
        )
    else:
        _sampler = optuna.samplers.RandomSampler(seed=seed)

    study = optuna.create_study(direction="minimize", sampler=_sampler)

    history: list[tuple[int, float]] = []
    best_so_far = [float("inf")]

    def objective(trial: "optuna.Trial") -> float:
        x = []
        for p in param_defs:
            lo, hi = float(p["min"]), float(p["max"])
            if p.get("scale") == "log":
                val = trial.suggest_float(p["name"], lo, hi, log=True)
            else:
                val = trial.suggest_float(p["name"], lo, hi)
            x.append(val)

        X = np.array(x, dtype=float).reshape(1, -1)
        scores = score_vector(
            X, model, scaler, targets, metric_defs,
            weights=weights,
            log_indices=log_indices or None,
            log_metric_indices=log_metric_indices or None,
            loss_type=loss_type,
            direction_penalty=direction_penalty,
            tolerance_pct=tolerance_pct,
        )
        return float(scores[0])

    def _optuna_callback(study: "optuna.Study", trial: "optuna.FrozenTrial"):
        best = study.best_value
        if best < best_so_far[0]:
            best_so_far[0] = best
        t = trial.number + 1
        history.append((t, best_so_far[0]))
        if progress_callback is not None:
            progress_callback(t, best_so_far[0])

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_optuna_callback],
        show_progress_bar=False,
    )

    # ── Build output ──────────────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = {name: best_trial.params[name] for name in param_names}
    best_x      = np.array([best_params[name] for name in param_names], dtype=float)

    def _predict_linear(X: np.ndarray) -> np.ndarray:
        Xc = X.copy()
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
        return preds

    y_best = _predict_linear(best_x.reshape(1, -1)).ravel()
    best_predicted = {
        m["name"]: float(y_best[i]) if i < len(y_best) else float("nan")
        for i, m in enumerate(metric_defs)
    }

    # Population: top-50 trials by score
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    top_trials = sorted_trials[:50]

    pop_X = np.array(
        [[float(t.params.get(name, 0.0)) for name in param_names] for t in top_trials],
        dtype=float,
    )
    pop_preds = _predict_linear(pop_X) if len(pop_X) > 0 else np.zeros((0, len(metric_defs)))

    population_out = []
    for i, t in enumerate(top_trials):
        params_i = {name: float(t.params.get(name, 0.0)) for name in param_names}
        score_i  = float(t.value) if t.value is not None else float("inf")
        pred_i   = {
            m["name"]: float(pop_preds[i, j]) if j < pop_preds.shape[1] else float("nan")
            for j, m in enumerate(metric_defs)
        }
        population_out.append((params_i, score_i, pred_i))

    return {
        "best_params":    best_params,
        "best_score":     float(study.best_value),
        "best_predicted": best_predicted,
        "history":        history,
        "population":     population_out,
    }

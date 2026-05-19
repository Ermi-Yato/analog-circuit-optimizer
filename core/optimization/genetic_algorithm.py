"""
Genetic Algorithm optimizer.

Uses DEAP to find component values that minimise the fitness function
built from the surrogate model + user-supplied targets.

Public API:
    optimize(circuit_id, targets, ...) -> result_dict

result_dict keys:
    "best_params"   : {param_name: value, ...}  — best candidate found
    "best_score"    : float                     — fitness score (lower = better)
    "best_predicted": {metric_name: value, ...} — surrogate predictions for best
    "history"       : [(generation, best_score), ...]
    "population"    : list of (params_dict, score, predicted_metrics_dict) for full final population
"""
from __future__ import annotations

import os
import random
import joblib
import copy

import numpy as np
from deap import base, creator, tools

import registry.circuit_registry as reg
from core.models.random_forest import RandomForestModel
from core.models.mlp import MLPModel
from core.optimization.objective import build_fitness_fn


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def optimize(
    circuit_id: str,
    targets: dict[str, float],
    n_generations: int = 100,
    pop_size: int = 200,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    tournament_size: int = 3,
    elite_size: int = 4,  # Kept intact every generation
    weights: dict[str, float] | None = None,
    progress_callback=None,
    seed: int | None = None,
) -> dict:
    """
    Run the GA optimizer for a circuit.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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

    # Indices of log-scale parameters — need log10 before scaler.transform
    log_indices = [i for i, p in enumerate(param_defs) if p.get("scale") == "log"]

    # Indices of log-scale metrics — model predicts log10(value); inverse at inference
    log_metric_names   = model_block.get("log_metrics", [])
    log_metric_indices = [i for i, m in enumerate(metric_defs) if m["name"] in log_metric_names]

    # Pre-compute fitness weight and scale tables for batch evaluation
    metric_names = [m["name"] for m in metric_defs]
    _weights = {n: 1.0 for n in metric_names} if weights is None else {n: weights.get(n, 1.0) for n in metric_names}
    _scales  = {n: abs(targets[n]) if n in targets and abs(targets.get(n, 0)) > 1e-12 else 1.0
                for n in metric_names}

    fitness_fn = build_fitness_fn(model, scaler, targets, metric_defs, weights,
                                  log_indices=log_indices or None,
                                  log_metric_indices=log_metric_indices or None)

    # DEAP toolbox setup
    toolbox = base.Toolbox()

    import math
    for p in param_defs:
        lo, hi = float(p["min"]), float(p["max"])
        if p.get("scale") == "log":
            # Sample uniformly in log space so all decades are equally represented
            ll, lh = math.log10(lo), math.log10(hi)
            toolbox.register(f"attr_{p['name']}",
                             lambda ll=ll, lh=lh: 10.0 ** random.uniform(ll, lh))
        else:
            toolbox.register(f"attr_{p['name']}", random.uniform, lo, hi)

    attr_fns = [getattr(toolbox, f"attr_{p['name']}") for p in param_defs]

    def make_individual():
        return creator.Individual([fn() for fn in attr_fns])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate",        tools.cxBlend, alpha=0.5)
    toolbox.register("select",      tools.selTournament, tournsize=tournament_size)

    # Bounds enforcement decorator
    bounds = [(float(p["min"]), float(p["max"])) for p in param_defs]
    toolbox.decorate("mate", _clip_bounds(bounds))

    def evaluate_population_batched(individuals):
        """
        True batch evaluation — single model forward pass for all individuals.
        Avoids N separate scaler.transform + predict calls (critical for MLP).
        """
        if not individuals:
            return

        X = np.array([list(ind) for ind in individuals], dtype=float)
        if log_indices:
            X = X.copy()
            X[:, log_indices] = np.log10(np.abs(X[:, log_indices]).clip(1e-300))
        preds = model.predict(scaler.transform(X))   # (n_inds, n_metrics)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        # Inverse-transform log-metric columns (model predicted log10, we need linear)
        if log_metric_indices:
            preds = preds.copy()
            for mi in log_metric_indices:
                if mi < preds.shape[1]:
                    preds[:, mi] = 10.0 ** np.clip(preds[:, mi], -300, 300)

        for row, ind in enumerate(individuals):
            score = 0.0
            for i, name in enumerate(metric_names):
                if name not in targets:
                    continue
                predicted = float(preds[row, i]) if i < preds.shape[1] else 0.0
                score += _weights[name] * abs(predicted - targets[name]) / _scales[name]
            ind.fitness.values = (score,)

    # Base ranges for simulated annealing mutation
    # Log-scale params: sigma is 10% of the log10 range, converted back to linear
    # to keep mutations proportional across decades (e.g. 1 decade sigma for C).
    initial_sigmas = []
    for p in param_defs:
        lo, hi = float(p["min"]), float(p["max"])
        if p.get("scale") == "log":
            # Sigma in log space = 0.1 * (log10_hi - log10_lo) decades
            # Sigma in linear space = 0.1 * (hi - lo) in log-space → use geometric midpoint
            log_range = math.log10(hi) - math.log10(lo)
            mid_log   = 0.5 * (math.log10(lo) + math.log10(hi))
            initial_sigmas.append(0.1 * (10.0 ** (mid_log + log_range / 2) - 10.0 ** (mid_log - log_range / 2)))
        else:
            initial_sigmas.append(0.1 * (hi - lo))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    history = []

    # Initial batch evaluation
    evaluate_population_batched(pop)
    hof.update(pop)

    for gen in range(1, n_generations + 1):
        # 1. Elitism: Extract the absolute best individuals to survive untouched
        elites = tools.selBest(pop, elite_size)
        elite_clones = [toolbox.clone(el) for el in elites]

        # 2. Selection for the rest of the population
        offspring = toolbox.select(pop, pop_size - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # 3. Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 4. Mutation with Linear Decay (Simulated Annealing)
        # Sigma shrinks linearly from 100% of initial down to 5% near final generation
        decay_factor = max(0.05, 1.0 - (gen / n_generations))
        current_sigmas = [sig * decay_factor for sig in initial_sigmas]

        for mutant in offspring:
            if random.random() < mutation_prob:
                # Custom local mutation call utilizing dynamic sigmas
                tools.mutGaussian(mutant, mu=0, sigma=current_sigmas, indpb=0.3)
                # Clip bounds manually on mutation since it bypasses decorated toolbox keys
                for i, (lo, hi) in enumerate(bounds):
                    mutant[i] = max(lo, min(hi, mutant[i]))
                del mutant.fitness.values

        # 5. Efficiently evaluate only altered individuals in batch
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        evaluate_population_batched(invalid)

        # 6. Reconstruct population combining unchanged Elites + New Offspring
        pop[:] = elite_clones + offspring
        hof.update(pop)

        best_score = hof[0].fitness.values[0]
        history.append((gen, best_score))

        if progress_callback is not None:
            progress_callback(gen, best_score)

    # Prepare detailed results
    best_ind = hof[0]
    best_params = {name: float(best_ind[i]) for i, name in enumerate(param_names)}

    def _predict_linear(X: np.ndarray) -> np.ndarray:
        """Apply feature log-transform, run model, inverse-transform log metrics."""
        if log_indices:
            X = X.copy()
            X[:, log_indices] = np.log10(np.abs(X[:, log_indices]).clip(1e-300))
        preds = model.predict(scaler.transform(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if log_metric_indices:
            preds = preds.copy()
            for mi in log_metric_indices:
                if mi < preds.shape[1]:
                    preds[:, mi] = 10.0 ** np.clip(preds[:, mi], -300, 300)
        return preds

    X_best = np.array(list(best_params.values()), dtype=float).reshape(1, -1)
    y_pred = _predict_linear(X_best).ravel()
    best_predicted = {
        m["name"]: float(y_pred[i]) if i < len(y_pred) else float("nan")
        for i, m in enumerate(metric_defs)
    }

    # Dedup & sort final population output list
    merged = {id(ind): ind for ind in pop}
    for ind in hof:
        merged[id(ind)] = ind
    sorted_inds = sorted(merged.values(), key=lambda ind: ind.fitness.values[0])

    all_X = np.array([[float(v) for v in ind] for ind in sorted_inds], dtype=float)
    all_preds = _predict_linear(all_X)

    population_out = [
        (
            {name: float(ind[i]) for i, name in enumerate(param_names)},
            float(ind.fitness.values[0]),
            {m["name"]: float(all_preds[row, i]) if i < all_preds.shape[1] else float("nan")
             for i, m in enumerate(metric_defs)},
        )
        for row, ind in enumerate(sorted_inds)
    ]

    return {
        "best_params": best_params,
        "best_score": float(best_ind.fitness.values[0]),
        "best_predicted": best_predicted,
        "history": history,
        "population": population_out,
    }


def _clip_bounds(bounds: list[tuple[float, float]]):
    """DEAP decorator that clips each gene to [min, max] after mating operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for ind in offspring:
                for i, (lo, hi) in enumerate(bounds):
                    ind[i] = max(lo, min(hi, ind[i]))
            return offspring
        return wrapper
    return decorator

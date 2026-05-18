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
    "population"    : list of (params_dict, score) for full final population
"""
from __future__ import annotations

import os
import random
import joblib
import warnings

import numpy as np
from deap import base, creator, tools, algorithms

import registry.circuit_registry as reg
from core.models.random_forest import RandomForestModel
from core.optimization.objective import build_fitness_fn


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# DEAP uses module-level globals for FitnessMin/Individual.
# Guard against re-registration across test runs.
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
    weights: dict[str, float] | None = None,
    progress_callback=None,
    seed: int | None = None,
) -> dict:
    """
    Run the GA optimizer for a circuit.

    Args:
        circuit_id:        Registry circuit ID. Must have a trained model.
        targets:           {metric_name: target_value}. Missing metrics ignored.
        n_generations:     Number of GA generations.
        pop_size:          Population size per generation.
        crossover_prob:    Probability of crossover between two parents.
        mutation_prob:     Probability of mutating an individual.
        tournament_size:   Tournament selection size.
        weights:           Optional per-metric fitness weights.
        progress_callback: Called as (generation: int, best_score: float)
                           after each generation. Suitable for GUI progress bars.
        seed:              RNG seed for reproducibility.

    Returns:
        result_dict — see module docstring for keys.

    Raises:
        FileNotFoundError: Model PKL files not found — train first.
        KeyError:          circuit_id not in registry.
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
        raise FileNotFoundError(
            f"No trained model for '{circuit_id}'. Train the model first."
        )

    model_path  = os.path.join(_PROJECT_ROOT, model_block["surrogate_path"])
    scaler_path = os.path.join(_PROJECT_ROOT, model_block["scaler_path"])

    for p in (model_path, scaler_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Model file not found: {p}")

    model = RandomForestModel()
    model.load(model_path)
    scaler = joblib.load(scaler_path)

    fitness_fn = build_fitness_fn(model, scaler, targets, metric_defs, weights)

    # DEAP toolbox
    toolbox = base.Toolbox()

    # Attribute generators — one per parameter, respecting bounds
    for p in param_defs:
        lo, hi = float(p["min"]), float(p["max"])
        attr_name = f"attr_{p['name']}"
        toolbox.register(attr_name, random.uniform, lo, hi)

    attr_fns = [getattr(toolbox, f"attr_{p['name']}") for p in param_defs]

    def make_individual():
        return creator.Individual([fn() for fn in attr_fns])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",   fitness_fn)
    toolbox.register("mate",       tools.cxBlend, alpha=0.5)
    toolbox.register("mutate",     tools.mutGaussian,
                     mu=0, sigma=_param_sigmas(param_defs), indpb=0.3)
    toolbox.register("select",     tools.selTournament, tournsize=tournament_size)

    # Bounds enforcement decorator
    bounds = [(float(p["min"]), float(p["max"])) for p in param_defs]
    toolbox.decorate("mate",   _clip_bounds(bounds))
    toolbox.decorate("mutate", _clip_bounds(bounds))

    # Run GA
    pop     = toolbox.population(n=pop_size)
    hof     = tools.HallOfFame(1)   # tracks best individual ever seen
    history = []

    # Evaluate initial population
    fitnesses = [toolbox.evaluate(ind) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    for gen in range(1, n_generations + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate modified individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)

        best_score = hof[0].fitness.values[0]
        history.append((gen, best_score))

        if progress_callback is not None:
            progress_callback(gen, best_score)

    # Best individual is always the hall-of-fame winner, not just final pop best
    best_ind   = hof[0]
    best_params = {name: float(best_ind[i]) for i, name in enumerate(param_names)}

    # Predict metrics for best candidate
    X_best   = np.array(list(best_params.values()), dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X_best)
    y_pred   = model.predict(X_scaled).ravel()
    best_predicted = {
        m["name"]: float(y_pred[i]) if i < len(y_pred) else float("nan")
        for i, m in enumerate(metric_defs)
    }

    # Full final population sorted by fitness.
    # Merge HoF into pop so the all-time best is always present.
    merged = {id(ind): ind for ind in pop}
    for ind in hof:
        merged[id(ind)] = ind

    sorted_inds = sorted(merged.values(), key=lambda ind: ind.fitness.values[0])

    # Batch-predict metrics for all candidates (RF inference is fast)
    all_X = np.array([[float(v) for v in ind] for ind in sorted_inds], dtype=float)
    all_X_scaled = scaler.transform(all_X)
    all_preds = model.predict(all_X_scaled)   # shape (n, n_metrics)

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
        "best_params":    best_params,
        "best_score":     float(best_ind.fitness.values[0]),
        "best_predicted": best_predicted,
        "history":        history,
        "population":     population_out,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _param_sigmas(param_defs: list[dict]) -> list[float]:
    """
    Gaussian mutation sigma = 10% of each parameter's range.
    Passed to tools.mutGaussian as a per-gene sigma list.
    """
    return [0.1 * (float(p["max"]) - float(p["min"])) for p in param_defs]


def _clip_bounds(bounds: list[tuple[float, float]]):
    """
    DEAP decorator that clips each gene to [min, max] after mate/mutate.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for ind in offspring:
                for i, (lo, hi) in enumerate(bounds):
                    ind[i] = max(lo, min(hi, ind[i]))
            return offspring
        return wrapper
    return decorator

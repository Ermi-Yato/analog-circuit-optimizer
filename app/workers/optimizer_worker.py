"""
Optimizer worker — dispatches to GA, scipy, or Bayesian optimizer in a QThread.

Signals:
    generation(int, float)   iteration number, best score so far
    status(str)              human-readable status line
    finished(object)         result_dict on success
    stopped()                emitted when cancelled via stop()
    error(str)               error message on failure

Supported algorithms:
    "ga"                      — DEAP Genetic Algorithm
    "differential_evolution"  — scipy DE
    "dual_annealing"          — scipy Dual Annealing
    "bayesian_tpe"            — Bayesian / TPE (optuna)
    "bayesian_cmaes"          — Bayesian / CMA-ES (optuna)
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

_GA_ALGO     = "ga"
_SCIPY_ALGOS = {"differential_evolution", "dual_annealing"}
_BAYES_ALGOS = {"bayesian_tpe", "bayesian_cmaes"}


class _Stopped(BaseException):
    """Raised inside progress callback to abort the optimizer loop."""


class OptimizerWorker(QThread):
    generation = Signal(int, float)
    status     = Signal(str)
    finished   = Signal(object)
    stopped    = Signal()
    error      = Signal(str)

    def __init__(
        self,
        circuit_id: str,
        targets: dict[str, float],
        algorithm: str = "ga",
        # GA
        n_generations: int = 100,
        pop_size: int = 200,
        mutation_prob: float = 0.2,
        crossover_prob: float = 0.7,
        elite_size: int = 4,
        tournament_size: int = 3,
        # DE
        de_popsize: int = 15,
        de_mutation: float = 0.7,
        de_recombination: float = 0.7,
        de_maxiter: int = 300,
        # DA
        da_maxiter: int = 1000,
        da_initial_temp: float = 5230.0,
        da_restart_temp: float = 2e-5,
        da_visit: float = 2.62,
        da_accept: float = -5.0,
        # Bayesian
        bo_n_trials: int = 300,
        bo_n_startup: int = 20,
        # Fitness settings (shared across all algorithms)
        loss_type: str = "mae",
        direction_penalty: float = 1.0,
        tolerance_pct: float = 0.0,
        # Shared
        seed: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._circuit_id      = circuit_id
        self._targets         = targets
        self._algorithm       = algorithm
        self._n_generations   = n_generations
        self._pop_size        = pop_size
        self._mutation_prob   = mutation_prob
        self._crossover_prob  = crossover_prob
        self._elite_size      = elite_size
        self._tournament_size = tournament_size
        self._de_popsize      = de_popsize
        self._de_mutation     = de_mutation
        self._de_recombination= de_recombination
        self._de_maxiter      = de_maxiter
        self._da_maxiter      = da_maxiter
        self._da_initial_temp = da_initial_temp
        self._da_restart_temp = da_restart_temp
        self._da_visit        = da_visit
        self._da_accept       = da_accept
        self._bo_n_trials     = bo_n_trials
        self._bo_n_startup    = bo_n_startup
        self._loss_type       = loss_type
        self._direction_penalty = direction_penalty
        self._tolerance_pct   = tolerance_pct
        self._seed            = seed
        self._stop            = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            def _cb(iteration: int, best_score: float):
                if self._stop:
                    raise _Stopped()
                self.generation.emit(iteration, best_score)
                self.status.emit(f"Iteration {iteration}  best score: {best_score:.4f}")

            if self._algorithm == _GA_ALGO:
                from core.optimization.genetic_algorithm import optimize
                self.status.emit(
                    f"Running GA — {self._n_generations} generations, pop {self._pop_size}..."
                )
                result = optimize(
                    self._circuit_id, self._targets,
                    n_generations=self._n_generations,
                    pop_size=self._pop_size,
                    mutation_prob=self._mutation_prob,
                    crossover_prob=self._crossover_prob,
                    elite_size=self._elite_size,
                    tournament_size=self._tournament_size,
                    progress_callback=_cb,
                    seed=self._seed,
                )

            elif self._algorithm in _SCIPY_ALGOS:
                from core.optimization.scipy_optimizer import optimize
                label = {
                    "differential_evolution": "Differential Evolution",
                    "dual_annealing": "Dual Annealing",
                }[self._algorithm]
                self.status.emit(f"Running {label}...")
                result = optimize(
                    self._circuit_id, self._targets,
                    algorithm=self._algorithm,
                    de_popsize=self._de_popsize,
                    de_mutation=self._de_mutation,
                    de_recombination=self._de_recombination,
                    de_maxiter=self._de_maxiter,
                    da_maxiter=self._da_maxiter,
                    da_initial_temp=self._da_initial_temp,
                    da_restart_temp=self._da_restart_temp,
                    da_visit=self._da_visit,
                    da_accept=self._da_accept,
                    loss_type=self._loss_type,
                    direction_penalty=self._direction_penalty,
                    tolerance_pct=self._tolerance_pct,
                    progress_callback=_cb,
                    seed=self._seed,
                )

            elif self._algorithm in _BAYES_ALGOS:
                from core.optimization.bayesian_optimizer import optimize
                sampler = "cmaes" if self._algorithm == "bayesian_cmaes" else "tpe"
                label   = "Bayesian (CMA-ES)" if sampler == "cmaes" else "Bayesian (TPE)"
                self.status.emit(f"Running {label} — {self._bo_n_trials} trials...")
                result = optimize(
                    self._circuit_id, self._targets,
                    n_trials=self._bo_n_trials,
                    sampler=sampler,
                    n_startup_trials=self._bo_n_startup,
                    loss_type=self._loss_type,
                    direction_penalty=self._direction_penalty,
                    tolerance_pct=self._tolerance_pct,
                    progress_callback=_cb,
                    seed=self._seed,
                )

            else:
                raise ValueError(f"Unknown algorithm: {self._algorithm!r}")

            self.status.emit(f"Done — best score: {result['best_score']:.4f}")
            self.finished.emit(result)

        except _Stopped:
            self.stopped.emit()
        except Exception as exc:
            self.error.emit(str(exc))

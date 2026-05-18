"""
Optimizer worker — wraps core/optimization/genetic_algorithm.py in a QThread.

Signals:
    generation(int, float)   generation number, best fitness so far
    status(str)              human-readable status line
    finished(object)         result_dict on success
    stopped()                emitted when cancelled via stop()
    error(str)               error message on failure
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class _Stopped(BaseException):
    """Raised inside generation callback to abort the GA loop."""


class OptimizerWorker(QThread):
    generation = Signal(int, float)
    status     = Signal(str)
    finished   = Signal(object)   # result_dict
    stopped    = Signal()
    error      = Signal(str)

    def __init__(
        self,
        circuit_id: str,
        targets: dict[str, float],
        n_generations: int = 100,
        pop_size: int = 200,
        seed: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._circuit_id    = circuit_id
        self._targets       = targets
        self._n_generations = n_generations
        self._pop_size      = pop_size
        self._seed          = seed
        self._stop          = False

    def stop(self):
        """Request cancellation. The worker will stop after the current generation."""
        self._stop = True

    def run(self):
        try:
            from core.optimization.genetic_algorithm import optimize

            self.status.emit(
                f"Running GA — {self._n_generations} generations, "
                f"pop {self._pop_size}..."
            )

            def _cb(gen: int, best_score: float):
                if self._stop:
                    raise _Stopped()
                print(f"[GA] gen {gen:>4}/{self._n_generations}  best_score={best_score:.6f}")
                self.generation.emit(gen, best_score)
                self.status.emit(f"Generation {gen}/{self._n_generations}  "
                                 f"best score: {best_score:.4f}")

            result = optimize(
                self._circuit_id,
                self._targets,
                n_generations=self._n_generations,
                pop_size=self._pop_size,
                progress_callback=_cb,
                seed=self._seed,
            )

            print(f"[GA] done — result best_score={result['best_score']:.6f}")
            self.status.emit("Optimization complete.")
            self.finished.emit(result)

        except _Stopped:
            self.stopped.emit()
        except Exception as exc:
            self.error.emit(str(exc))

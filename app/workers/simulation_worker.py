"""
Simulation worker — wraps core/dataset/generator.py in a QThread.

Signals:
    progress(int, int)       completed, total
    status(str)              human-readable status line
    finished(object)         pandas DataFrame on success
    stopped()                emitted when cancelled via stop()
    error(str)               error message on failure
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class _Stopped(BaseException):
    """Raised inside progress callback to abort the simulation loop."""


class SimulationWorker(QThread):
    progress = Signal(int, int)
    status   = Signal(str)
    finished = Signal(object)   # DataFrame
    stopped  = Signal()
    error    = Signal(str)

    def __init__(
        self,
        circuit_id: str,
        n_samples: int,
        max_workers: int | None = None,
        seed: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._circuit_id  = circuit_id
        self._n_samples   = n_samples
        self._max_workers = max_workers
        self._seed        = seed
        self._stop        = False

    def stop(self):
        """Request cancellation. The worker will stop at the next progress tick."""
        self._stop = True

    def run(self):
        try:
            from core.dataset.generator import generate

            self.status.emit(f"Sampling {self._n_samples} parameter sets...")

            def _cb(completed: int, total: int):
                if self._stop:
                    raise _Stopped()
                self.progress.emit(completed, total)
                self.status.emit(f"Simulating {completed}/{total}...")

            df = generate(
                self._circuit_id,
                self._n_samples,
                progress_callback=_cb,
                max_workers=self._max_workers,
                seed=self._seed,
                verbose=False,
            )

            self.status.emit(f"Done — {len(df)} samples generated.")
            self.finished.emit(df)

        except _Stopped:
            self.stopped.emit()
        except Exception as exc:
            self.error.emit(str(exc))

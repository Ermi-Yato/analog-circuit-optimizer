"""
Training worker — wraps core/models/trainer.py in a QThread.

Signals:
    status(str)          human-readable status line
    finished(object)     metrics dict on success
    stopped()            emitted when cancelled via stop() (result discarded)
    error(str)           error message on failure

Note: sklearn RandomForest training is a single blocking call and cannot be
interrupted mid-fit. Calling stop() will discard the result if training
finishes while the flag is set, or cancel before the fit call starts.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    status   = Signal(str)
    finished = Signal(object)   # metrics dict
    stopped  = Signal()
    error    = Signal(str)

    def __init__(self, circuit_id: str, model_type: str = "random_forest", parent=None):
        super().__init__(parent)
        self._circuit_id = circuit_id
        self._model_type = model_type
        self._stop       = False

    def stop(self):
        """Request cancellation. Result will be discarded if training already started."""
        self._stop = True

    def run(self):
        try:
            from core.models.trainer import train

            if self._stop:
                self.stopped.emit()
                return

            self.status.emit("Loading dataset...")

            if self._stop:
                self.stopped.emit()
                return

            metrics = train(self._circuit_id, verbose=False, model_type=self._model_type)

            if self._stop:
                self.stopped.emit()
                return

            self.status.emit("Training complete.")
            self.finished.emit(metrics)

        except Exception as exc:
            self.error.emit(str(exc))

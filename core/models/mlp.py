"""
MLP surrogate model — wraps sklearn MLPRegressor.

Better than RandomForest for circuits with log-linear or multiplicative
parameter-metric relationships (e.g. Sallen-Key: fc = 1/(2π√(R1·R2·C1·C2))).

Architecture: 3 hidden layers (256 → 128 → 64), ReLU activation, Adam solver.
Early stopping avoids overfitting on smaller datasets.
"""
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor

from core.models.base_model import BaseModel


class MLPModel(BaseModel):
    """
    Multi-layer perceptron surrogate using scikit-learn's MLPRegressor.

    Inputs must be pre-scaled (StandardScaler applied upstream in trainer.py).
    Handles multi-output regression natively via sklearn's multi-output wrapper
    when y has more than one column.
    """

    def __init__(self):
        self._model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        self._multioutput = False
        self._models: list[MLPRegressor] = []   # one per target if multi-output

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # sklearn MLPRegressor supports multi-output natively only in newer
        # versions; wrap in per-column models to be safe across versions.
        if y.ndim == 1 or y.shape[1] == 1:
            self._multioutput = False
            self._model.fit(X, y.ravel() if y.ndim == 2 else y)
        else:
            self._multioutput = True
            self._models = []
            for col in range(y.shape[1]):
                m = MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                )
                m.fit(X, y[:, col])
                self._models.append(m)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._multioutput:
            cols = [m.predict(X) for m in self._models]
            return np.column_stack(cols)
        pred = self._model.predict(X)
        return pred.reshape(-1, 1) if pred.ndim == 1 else pred

    def save(self, path: str) -> None:
        payload = {
            "multioutput": self._multioutput,
            "model": self._model,
            "models": self._models,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self._multioutput = payload["multioutput"]
        self._model = payload["model"]
        self._models = payload["models"]

    @property
    def feature_importances(self):
        """MLP has no inherent feature importance. Returns None."""
        return None

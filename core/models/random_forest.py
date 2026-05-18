"""
Random Forest surrogate model.

Wraps sklearn's MultiOutputRegressor-compatible RandomForestRegressor.
Implements the BaseModel interface for use in the training pipeline,
GA optimizer, and BYOM import flow.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from core.models.base_model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # sklearn's RandomForestRegressor natively handles multi-output
        # when y is 2-D (n_samples, n_targets).
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        self._model = joblib.load(path)

    @property
    def feature_importances(self) -> np.ndarray | None:
        """
        Per-feature importance scores, shape (n_features,).
        Averaged across all output estimators for multi-output models.
        Returns None if the model has not been trained yet.
        """
        try:
            # sklearn RF natively averages importances across outputs
            return self._model.feature_importances_
        except AttributeError:
            return None

"""
Abstract base class for surrogate models.

All surrogate model implementations must implement this interface so that
the trainer, optimizer, and GUI can work with any model interchangeably.
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on scaled features X and targets y.

        Args:
            X: Feature matrix, shape (n_samples, n_features). Pre-scaled.
            y: Target matrix, shape (n_samples, n_targets). Unscaled.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets for scaled feature matrix X.

        Args:
            X: Feature matrix, shape (n_samples, n_features). Must be scaled
               with the same scaler used during training.

        Returns:
            Predicted targets, shape (n_samples, n_targets).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the model to disk at the given path.

        The caller is responsible for creating the parent directory.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model state from disk, replacing any existing state.
        """

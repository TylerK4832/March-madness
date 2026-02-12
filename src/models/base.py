"""Abstract base class for all March Madness models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Base class that all models must implement."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability that TeamA (lower ID) wins, shape (n_samples,)."""
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return model hyperparameters for logging."""
        ...

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        """Return feature importances if available."""
        return None

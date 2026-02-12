"""Logistic regression baseline model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.base import BaseModel


class LogisticModel(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_params(self) -> dict:
        return {"model_type": "logistic", "C": self.C, "max_iter": self.max_iter}

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        coefs = self.model.coef_[0]
        return dict(zip(feature_names, coefs))

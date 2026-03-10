"""Stacking ensemble model: LR + RF + GBM with logistic meta-learner."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class StackingModel(BaseModel):
    """Stacking ensemble: trains LR, RF, and GBM as base models,
    then a logistic regression meta-learner on their predictions
    plus the original features (passthrough)."""

    def __init__(self, C: float = 1.0, rf_n_estimators: int = 100,
                 rf_max_depth: int = 3, gbm_n_estimators: int = 100,
                 gbm_max_depth: int = 2, gbm_learning_rate: float = 0.1,
                 cv: int = 5):
        self.C = C
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.gbm_n_estimators = gbm_n_estimators
        self.gbm_max_depth = gbm_max_depth
        self.gbm_learning_rate = gbm_learning_rate
        self.cv = cv
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingModel":
        X_scaled = self.scaler.fit_transform(X)

        base_models = [
            ("lr", LogisticRegression(C=self.C, max_iter=1000)),
            ("rf", RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                random_state=42,
            )),
            ("gbm", GradientBoostingClassifier(
                n_estimators=self.gbm_n_estimators,
                max_depth=self.gbm_max_depth,
                learning_rate=self.gbm_learning_rate,
                random_state=42,
            )),
        ]

        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=self.cv,
            passthrough=True,
        )
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_params(self) -> dict:
        return {
            "model_type": "stacking",
            "C": self.C,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "gbm_n_estimators": self.gbm_n_estimators,
            "gbm_max_depth": self.gbm_max_depth,
            "gbm_learning_rate": self.gbm_learning_rate,
            "cv": self.cv,
        }

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        # Return the LR base model's coefficients as a proxy
        lr_model = self.model.estimators_[0]
        coeffs = lr_model.coef_[0]
        return dict(zip(feature_names, coeffs))

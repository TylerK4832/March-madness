"""XGBoost model for March Madness prediction."""

import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            **self.params,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict:
        return {"model_type": "xgboost", **self.params}

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

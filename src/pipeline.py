"""Sklearn Pipeline for March Madness predictions.

Wraps feature computation + model into a Pipeline so that per-fold
feature transforms (like historical seed win rates) are computed from
training data only, preventing leakage.

Architecture:
    Phase 1 (outside CV, safe): Precompute team-season stats from regular
        season data. These are per-season and cannot leak.
    Phase 2 (inside CV, Pipeline):
        MatchupDifferentialTransformer -> CrossSeasonFeatureTransformer ->
        FeatureTierSelector -> SklearnModelWrapper
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline

from src.feature_engineering import compute_matchup_differentials


class MatchupDifferentialTransformer(BaseEstimator, TransformerMixin):
    """Computes TeamA - TeamB differential features from precomputed team stats.

    Input X must have columns: Season, TeamA, TeamB.
    Output is a DataFrame of differential features plus Season, TeamA, TeamB.
    """

    def __init__(self, team_stats: pd.DataFrame):
        self.team_stats = team_stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return compute_matchup_differentials(X, self.team_stats)


class CrossSeasonFeatureTransformer(BaseEstimator, TransformerMixin):
    """Computes features that aggregate across seasons from training data.

    This transformer is fitted on training data and applied to test data,
    ensuring no leakage of future tournament outcomes.

    Currently computes:
    - HistSeedWinRate: historical win rate for each seed differential,
      computed only from training-fold tournament outcomes.

    New cross-season features should be added here.
    """

    def __init__(self):
        self.seed_win_rates_ = {}

    def fit(self, X, y=None):
        if y is not None and "SeedDiff" in X.columns:
            # Reset indices to align X and y after upstream transforms
            sd_vals = X["SeedDiff"].values
            y_vals = np.asarray(y)
            for sd in np.unique(sd_vals):
                mask = sd_vals == sd
                if mask.sum() >= 3:
                    self.seed_win_rates_[sd] = y_vals[mask].mean()
        return self

    def transform(self, X):
        X = X.copy()
        if self.seed_win_rates_:
            X["HistSeedWinRate"] = X["SeedDiff"].map(self.seed_win_rates_).fillna(0.5)
        return X


class FeatureTierSelector(BaseEstimator, TransformerMixin):
    """Selects columns for the active feature tier."""

    def __init__(self, feature_tier: str):
        self.feature_tier = feature_tier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from config import FEATURE_TIERS
        tier_cols = FEATURE_TIERS[self.feature_tier]
        available = [c for c in tier_cols if c in X.columns]
        return X[available].astype(float)


class SklearnModelWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a BaseModel subclass for use in an sklearn Pipeline.

    Handles the interface difference: BaseModel.predict_proba returns 1-D,
    sklearn expects 2-D array [[p_0, p_1]].
    """

    def __init__(self, model_cls, **model_kwargs):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model_ = None

    def fit(self, X, y):
        self.model_ = self.model_cls(**self.model_kwargs)
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p1 = self.model_.predict_proba(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        p1 = self.model_.predict_proba(X)
        return (p1 >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"model_cls": self.model_cls, **self.model_kwargs}

    def set_params(self, **params):
        if "model_cls" in params:
            self.model_cls = params.pop("model_cls")
        self.model_kwargs.update(params)
        return self


def make_pipeline(model_cls, model_kwargs: dict,
                  team_stats: pd.DataFrame, feature_tier: str) -> Pipeline:
    """Create a fresh Pipeline for one CV fold.

    Args:
        model_cls: BaseModel subclass (e.g., LogisticModel).
        model_kwargs: Kwargs passed to model_cls.__init__.
        team_stats: Precomputed team-season stats DataFrame.
        feature_tier: Feature tier name from config.FEATURE_TIERS.
    """
    return Pipeline([
        ("differentials", MatchupDifferentialTransformer(team_stats)),
        ("cross_season", CrossSeasonFeatureTransformer()),
        ("tier_select", FeatureTierSelector(feature_tier)),
        ("model", SklearnModelWrapper(model_cls, **model_kwargs)),
    ])

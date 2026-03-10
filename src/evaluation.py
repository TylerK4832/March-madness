"""Evaluation: log loss, accuracy, upset accuracy, walk-forward and sliding window CV."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import Pipeline


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    seed_diffs: np.ndarray | None = None) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (1 if lower ID team won).
        y_prob: Predicted probabilities that lower ID team wins.
        seed_diffs: Seed differentials (positive = lower ID has better seed).
                    Used to identify upsets.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "n_games": len(y_true),
    }

    if seed_diffs is not None:
        # An upset = the lower-seeded team (worse seed) won
        # seed_diff > 0 means TeamA has better seed, so expected winner is TeamA (label=1)
        # seed_diff < 0 means TeamB has better seed, so expected winner is TeamB (label=0)
        expected = (seed_diffs >= 0).astype(int)
        upset_mask = y_true != expected
        n_upsets = upset_mask.sum()
        if n_upsets > 0:
            upset_correct = (y_pred[upset_mask] == y_true[upset_mask]).sum()
            metrics["upset_accuracy"] = upset_correct / n_upsets
            metrics["n_upsets"] = int(n_upsets)
        else:
            metrics["upset_accuracy"] = float("nan")
            metrics["n_upsets"] = 0

    return metrics


def _get_seed_diffs(X: pd.DataFrame) -> np.ndarray | None:
    """Extract seed diffs from either a feature matrix or raw matchup rows."""
    if "SeedDiff" in X.columns:
        return X["SeedDiff"].values
    return None


def _fit_predict(model_cls, model_kwargs, pipeline_factory,
                 X_train, y_train, X_test):
    """Fit a model or pipeline and return P(TeamA wins) as 1-D array."""
    if pipeline_factory is not None:
        pipe = pipeline_factory()
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)
    return np.clip(y_prob, 0.01, 0.99)


def leave_one_season_out_cv(
    model_cls: type = None,
    model_kwargs: dict = None,
    X: pd.DataFrame = None,
    y: pd.Series = None,
    seasons: pd.Series = None,
    seed_diffs: np.ndarray | None = None,
    pipeline_factory: callable = None,
) -> dict:
    """Leave-one-season-out cross-validation."""
    unique_seasons = sorted(seasons.unique())
    all_y_true, all_y_prob, all_seed_diffs = [], [], []
    per_season = []

    for held_out in unique_seasons:
        train_mask = seasons != held_out
        test_mask = seasons == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        y_prob = _fit_predict(model_cls, model_kwargs, pipeline_factory,
                              X_train, y_train, X_test)

        sd = seed_diffs[test_mask.values] if seed_diffs is not None else _get_seed_diffs(X_test)

        metrics = compute_metrics(y_test.values, y_prob, sd)
        metrics["season"] = held_out
        per_season.append(metrics)

        all_y_true.extend(y_test.values)
        all_y_prob.extend(y_prob)
        if sd is not None:
            all_seed_diffs.extend(sd)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_seed_diffs = np.array(all_seed_diffs) if all_seed_diffs else None

    overall = compute_metrics(all_y_true, all_y_prob, all_seed_diffs)
    overall["n_seasons"] = len(unique_seasons)

    return {"overall": overall, "per_season": per_season}


def sliding_window_cv(
    model_cls: type = None,
    model_kwargs: dict = None,
    X: pd.DataFrame = None,
    y: pd.Series = None,
    seasons: pd.Series = None,
    window_size: int | None = None,
    min_train_seasons: int = 5,
    seed_diffs: np.ndarray | None = None,
    pipeline_factory: callable = None,
) -> dict:
    """Sliding window cross-validation: train on a fixed window of recent seasons.

    For each test season, trains on the most recent `window_size` seasons before it.
    If window_size is None, uses all prior data (expanding window = walk-forward).

    Args:
        model_cls: BaseModel subclass. Not needed if pipeline_factory is provided.
        model_kwargs: Kwargs for model_cls. Not needed if pipeline_factory is provided.
        X: Feature DataFrame (precomputed differentials) or raw matchup rows
           (Season, TeamA, TeamB) if using pipeline_factory.
        y: Labels.
        seasons: Season for each row.
        window_size: Number of prior seasons to train on. None = expanding window.
        min_train_seasons: Minimum training seasons required before evaluating.
        seed_diffs: Precomputed seed diffs for upset detection. If None, attempts
                    to extract from X columns.
        pipeline_factory: Callable that returns a fresh Pipeline. If provided,
                         model_cls/model_kwargs are ignored.
    """
    unique_seasons = sorted(seasons.unique())
    all_y_true, all_y_prob, all_seed_diffs = [], [], []
    per_season = []

    for held_out in unique_seasons:
        prior_seasons = [s for s in unique_seasons if s < held_out]
        if len(prior_seasons) < min_train_seasons:
            continue

        if window_size is not None:
            train_seasons = set(prior_seasons[-window_size:])
            train_mask = seasons.isin(train_seasons)
        else:
            train_mask = seasons < held_out

        test_mask = seasons == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        y_prob = _fit_predict(model_cls, model_kwargs, pipeline_factory,
                              X_train, y_train, X_test)

        sd = seed_diffs[test_mask.values] if seed_diffs is not None else _get_seed_diffs(X_test)

        metrics = compute_metrics(y_test.values, y_prob, sd)
        metrics["season"] = held_out
        metrics["n_train"] = len(y_train)
        per_season.append(metrics)

        all_y_true.extend(y_test.values)
        all_y_prob.extend(y_prob)
        if sd is not None:
            all_seed_diffs.extend(sd)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_seed_diffs = np.array(all_seed_diffs) if all_seed_diffs else None

    overall = compute_metrics(all_y_true, all_y_prob, all_seed_diffs)
    overall["n_seasons"] = len(per_season)

    return {"overall": overall, "per_season": per_season}


def walk_forward_cv(
    model_cls: type = None,
    model_kwargs: dict = None,
    X: pd.DataFrame = None,
    y: pd.Series = None,
    seasons: pd.Series = None,
    test_seasons: list[int] | None = None,
    seed_diffs: np.ndarray | None = None,
    pipeline_factory: callable = None,
) -> dict:
    """Walk-forward cross-validation: train on past, predict each test season.

    Only uses data from seasons strictly before the held-out season for training.
    This mirrors real-world usage (no future data leakage).
    """
    unique_seasons = sorted(seasons.unique())

    if test_seasons is None:
        test_seasons = unique_seasons[-3:]

    all_y_true, all_y_prob, all_seed_diffs = [], [], []
    per_season = []

    for held_out in test_seasons:
        train_mask = seasons < held_out
        test_mask = seasons == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        y_prob = _fit_predict(model_cls, model_kwargs, pipeline_factory,
                              X_train, y_train, X_test)

        sd = seed_diffs[test_mask.values] if seed_diffs is not None else _get_seed_diffs(X_test)

        metrics = compute_metrics(y_test.values, y_prob, sd)
        metrics["season"] = held_out
        metrics["n_train"] = len(y_train)
        per_season.append(metrics)

        all_y_true.extend(y_test.values)
        all_y_prob.extend(y_prob)
        if sd is not None:
            all_seed_diffs.extend(sd)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_seed_diffs = np.array(all_seed_diffs) if all_seed_diffs else None

    overall = compute_metrics(all_y_true, all_y_prob, all_seed_diffs)
    overall["n_seasons"] = len(test_seasons)

    return {"overall": overall, "per_season": per_season}

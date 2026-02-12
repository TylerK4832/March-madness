"""Evaluation: log loss, accuracy, upset accuracy, leave-one-season-out CV."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from src.models.base import BaseModel


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


def leave_one_season_out_cv(
    model_cls: type,
    model_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    feature_cols: list[str] | None = None,
) -> dict:
    """Leave-one-season-out cross-validation.

    Returns aggregated metrics and per-season breakdown.
    """
    unique_seasons = sorted(seasons.unique())
    all_y_true = []
    all_y_prob = []
    all_seed_diffs = []
    per_season = []

    for held_out in unique_seasons:
        train_mask = seasons != held_out
        test_mask = seasons == held_out

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        # Clip probabilities for log loss stability
        y_prob = np.clip(y_prob, 0.01, 0.99)

        seed_diffs = X_test["SeedDiff"].values if "SeedDiff" in X_test.columns else None

        metrics = compute_metrics(y_test.values, y_prob, seed_diffs)
        metrics["season"] = held_out
        per_season.append(metrics)

        all_y_true.extend(y_test.values)
        all_y_prob.extend(y_prob)
        if seed_diffs is not None:
            all_seed_diffs.extend(seed_diffs)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_seed_diffs = np.array(all_seed_diffs) if all_seed_diffs else None

    overall = compute_metrics(all_y_true, all_y_prob, all_seed_diffs)
    overall["n_seasons"] = len(unique_seasons)

    return {
        "overall": overall,
        "per_season": per_season,
    }

"""Main training script for March Madness ML pipeline.

Usage:
    python train.py                          # Run all experiments
    python train.py --tier base              # Run specific feature tier
    python train.py --model xgboost          # Run specific model
    python train.py --tier full --model xgboost  # Specific combo
    python train.py --cv walk_forward        # Use walk-forward CV (last 3 seasons)
    python train.py --cv loso               # Use leave-one-season-out CV
    python train.py --window 7              # Sliding window with 7 training seasons
"""

import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from config import FEATURE_TIERS
from src.feature_engineering import build_matchup_features
from src.models import MODEL_REGISTRY
from src.evaluation import leave_one_season_out_cv, walk_forward_cv, sliding_window_cv
from src.experiment import log_experiment
from src.model_store import save_model, print_model_library


def run_experiment(model_name: str, model_kwargs: dict, feature_tier: str,
                   cv_method: str = "sliding", window_size: int | None = 10,
                   save: bool = True) -> dict:
    """Run a single experiment: build features, CV, log results."""
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | tier: {feature_tier} | cv: {cv_method}")
    print(f"{'='*60}")

    # Build features
    print("  Building matchup features...")
    X, y, seasons = build_matchup_features(feature_tier)
    print(f"  Dataset: {len(X)} games, {X.shape[1]} features, "
          f"{seasons.nunique()} seasons")
    print(f"  Features: {list(X.columns)}")
    print(f"  Label balance: {y.mean():.3f} (fraction TeamA wins)")

    # Run CV
    model_cls = MODEL_REGISTRY[model_name]
    if cv_method == "loso":
        print(f"  Running leave-one-season-out CV...")
        cv_results = leave_one_season_out_cv(model_cls, model_kwargs, X, y, seasons)
    elif cv_method == "walk_forward":
        test_seasons = sorted(seasons.unique())[-3:]
        print(f"  Running walk-forward CV (test seasons: {list(test_seasons)})...")
        cv_results = walk_forward_cv(model_cls, model_kwargs, X, y, seasons,
                                     test_seasons=list(test_seasons))
    else:
        win_label = f"window={window_size}" if window_size else "expanding"
        print(f"  Running sliding window CV ({win_label}, min 5 train seasons)...")
        cv_results = sliding_window_cv(model_cls, model_kwargs, X, y, seasons,
                                       window_size=window_size, min_train_seasons=5)

    overall = cv_results["overall"]
    print(f"\n  RESULTS:")
    print(f"    Log Loss:        {overall['log_loss']:.4f}")
    print(f"    Accuracy:        {overall['accuracy']:.4f}")
    if "upset_accuracy" in overall and not np.isnan(overall.get("upset_accuracy", float("nan"))):
        print(f"    Upset Accuracy:  {overall['upset_accuracy']:.4f} "
              f"({overall.get('n_upsets', 0)} upsets)")
    print(f"    Games evaluated: {overall['n_games']}")
    print(f"    Seasons tested:  {overall['n_seasons']}")

    # Train final model on all data for feature importance
    model = model_cls(**model_kwargs)
    model.fit(X, y)
    feature_importance = model.get_feature_importance(list(X.columns))

    if feature_importance:
        print(f"\n  Feature Importance (top 5):")
        sorted_fi = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, imp in sorted_fi[:5]:
            print(f"    {name:25s} {imp:+.4f}")

    # Log to MLflow
    run_name = f"{model_name}_{feature_tier}"
    log_experiment(run_name, model.get_params(), feature_tier, cv_results, feature_importance)

    # Save model
    if save:
        save_model(model, model_name, feature_tier, cv_results, model_kwargs,
                   list(X.columns), cv_method=cv_method)

    return cv_results


def main():
    parser = argparse.ArgumentParser(description="March Madness ML Training")
    parser.add_argument("--tier", type=str, default=None,
                        help="Feature tier (seed_only, base, base_massey, full, "
                             "efficiency, efficiency_4f, adj_efficiency, adj_efficiency_4f)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model type (logistic, xgboost)")
    parser.add_argument("--cv", type=str, default="sliding",
                        choices=["sliding", "walk_forward", "loso"],
                        help="CV method (default: sliding)")
    parser.add_argument("--window", type=int, default=10,
                        help="Training window size in seasons for sliding CV (default: 10)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save trained models to disk")
    args = parser.parse_args()

    # Define experiment grid
    tiers = [args.tier] if args.tier else list(FEATURE_TIERS.keys())
    models = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    model_configs = {
        "logistic": {"C": 1.0},
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "stacking": {
            "C": 1.0,
            "rf_n_estimators": 100,
            "rf_max_depth": 3,
            "gbm_n_estimators": 100,
            "gbm_max_depth": 2,
            "gbm_learning_rate": 0.1,
            "cv": 5,
        },
    }

    results_summary = []

    for tier in tiers:
        for model_name in models:
            kwargs = model_configs.get(model_name, {})
            cv = run_experiment(model_name, kwargs, tier, cv_method=args.cv,
                               window_size=args.window, save=not args.no_save)
            results_summary.append({
                "model": model_name,
                "tier": tier,
                "log_loss": cv["overall"]["log_loss"],
                "accuracy": cv["overall"]["accuracy"],
                "upset_accuracy": cv["overall"].get("upset_accuracy", float("nan")),
            })

    # Print summary table
    print(f"\n\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Tier':<20} {'Log Loss':>10} {'Accuracy':>10} {'Upset Acc':>10}")
    print(f"{'-'*12} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for r in sorted(results_summary, key=lambda x: -x["accuracy"]):
        upset = f"{r['upset_accuracy']:.4f}" if not np.isnan(r["upset_accuracy"]) else "N/A"
        print(f"{r['model']:<12} {r['tier']:<20} {r['log_loss']:>10.4f} "
              f"{r['accuracy']:>10.4f} {upset:>10}")

    print(f"\nMLflow UI: run 'mlflow ui' in the project directory to compare experiments.")

    if not args.no_save:
        print_model_library()


if __name__ == "__main__":
    main()

"""Main training script for March Madness ML pipeline.

Usage:
    python train.py                          # Run all experiments
    python train.py --tier base              # Run specific feature tier
    python train.py --model xgboost          # Run specific model
    python train.py --tier full --model xgboost  # Specific combo
"""

import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from config import FEATURE_TIERS
from src.feature_engineering import build_matchup_features
from src.models import MODEL_REGISTRY
from src.evaluation import leave_one_season_out_cv
from src.experiment import log_experiment


def run_experiment(model_name: str, model_kwargs: dict, feature_tier: str) -> dict:
    """Run a single experiment: build features, CV, log results."""
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | tier: {feature_tier}")
    print(f"{'='*60}")

    # Build features
    print("  Building matchup features...")
    X, y, seasons = build_matchup_features(feature_tier)
    print(f"  Dataset: {len(X)} games, {X.shape[1]} features, "
          f"{seasons.nunique()} seasons")
    print(f"  Features: {list(X.columns)}")
    print(f"  Label balance: {y.mean():.3f} (fraction TeamA wins)")

    # Run leave-one-season-out CV
    model_cls = MODEL_REGISTRY[model_name]
    print(f"  Running leave-one-season-out CV...")
    cv_results = leave_one_season_out_cv(model_cls, model_kwargs, X, y, seasons)

    overall = cv_results["overall"]
    print(f"\n  RESULTS:")
    print(f"    Log Loss:        {overall['log_loss']:.4f}")
    print(f"    Accuracy:        {overall['accuracy']:.4f}")
    if "upset_accuracy" in overall and not np.isnan(overall.get("upset_accuracy", float("nan"))):
        print(f"    Upset Accuracy:  {overall['upset_accuracy']:.4f} "
              f"({overall.get('n_upsets', 0)} upsets)")
    print(f"    Games evaluated: {overall['n_games']}")

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

    return cv_results


def main():
    parser = argparse.ArgumentParser(description="March Madness ML Training")
    parser.add_argument("--tier", type=str, default=None,
                        help="Feature tier (seed_only, base, base_massey, full)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model type (logistic, xgboost)")
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
    }

    results_summary = []

    for tier in tiers:
        for model_name in models:
            kwargs = model_configs.get(model_name, {})
            cv = run_experiment(model_name, kwargs, tier)
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
    print(f"{'Model':<12} {'Tier':<15} {'Log Loss':>10} {'Accuracy':>10} {'Upset Acc':>10}")
    print(f"{'-'*12} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for r in results_summary:
        upset = f"{r['upset_accuracy']:.4f}" if not np.isnan(r["upset_accuracy"]) else "N/A"
        print(f"{r['model']:<12} {r['tier']:<15} {r['log_loss']:>10.4f} "
              f"{r['accuracy']:>10.4f} {upset:>10}")

    print(f"\nMLflow UI: run 'mlflow ui' in the project directory to compare experiments.")


if __name__ == "__main__":
    main()

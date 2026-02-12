"""MLflow experiment tracking."""

import json
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_experiment(
    run_name: str,
    model_params: dict,
    feature_tier: str,
    cv_results: dict,
    feature_importance: dict[str, float] | None = None,
):
    """Log a complete experiment run to MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_param("feature_tier", feature_tier)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        # Log overall metrics
        overall = cv_results["overall"]
        for k, v in overall.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                mlflow.log_metric(k, v)

        # Log per-season metrics
        for season_metrics in cv_results["per_season"]:
            season = season_metrics["season"]
            for k, v in season_metrics.items():
                if k == "season":
                    continue
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"{k}_s{season}", v)

        # Log feature importance plot
        if feature_importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_feats = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            names = [f[0] for f in sorted_feats]
            values = [f[1] for f in sorted_feats]
            ax.barh(names, values)
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance - {run_name}")
            plt.tight_layout()
            fig.savefig("/tmp/feature_importance.png", dpi=100)
            plt.close(fig)
            mlflow.log_artifact("/tmp/feature_importance.png")

            # Also log as JSON
            mlflow.log_dict(feature_importance, "feature_importance.json")

        # Log CV details
        mlflow.log_dict(cv_results, "cv_results.json")

    print(f"  MLflow run '{run_name}' logged.")

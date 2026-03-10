"""Save, load, and list trained models."""

import json
import joblib
from pathlib import Path
from datetime import datetime

from config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"


def save_model(model, model_name: str, feature_tier: str, cv_results: dict,
               model_kwargs: dict, feature_cols: list[str],
               cv_method: str = "walk_forward") -> Path:
    """Save a trained model with metadata.

    Returns the path to the saved model directory.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy = cv_results["overall"]["accuracy"]
    log_loss = cv_results["overall"]["log_loss"]

    # Directory name encodes key info for easy browsing
    dir_name = f"{model_name}_{feature_tier}_{accuracy:.4f}acc_{timestamp}"
    model_dir = MODELS_DIR / dir_name
    model_dir.mkdir()

    # Save the model object
    joblib.dump(model, model_dir / "model.joblib")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "feature_tier": feature_tier,
        "feature_cols": feature_cols,
        "model_kwargs": model_kwargs,
        "cv_method": cv_method,
        "cv_results": cv_results,
        "timestamp": timestamp,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Model saved to {model_dir.name}/")
    return model_dir


def load_model(model_dir: str | Path) -> tuple:
    """Load a saved model and its metadata.

    Returns:
        (model, metadata) tuple.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = MODELS_DIR / model_dir

    model = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    return model, metadata


def list_models() -> list[dict]:
    """List all saved models, sorted by accuracy descending."""
    if not MODELS_DIR.exists():
        return []

    models = []
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["dir_name"] = d.name
        meta["path"] = str(d)
        models.append(meta)

    models.sort(key=lambda m: m["cv_results"]["overall"]["accuracy"], reverse=True)
    return models


def print_model_library():
    """Print a formatted table of all saved models."""
    models = list_models()
    if not models:
        print("No saved models found.")
        return

    print(f"\n{'='*85}")
    print("SAVED MODELS")
    print(f"{'='*85}")
    print(f"{'#':<4} {'Model':<10} {'Tier':<16} {'Accuracy':>10} {'Log Loss':>10} {'CV':>14} {'Date':>16}")
    print(f"{'-'*4} {'-'*10} {'-'*16} {'-'*10} {'-'*10} {'-'*14} {'-'*16}")

    for i, m in enumerate(models, 1):
        overall = m["cv_results"]["overall"]
        print(f"{i:<4} {m['model_name']:<10} {m['feature_tier']:<16} "
              f"{overall['accuracy']:>10.4f} {overall['log_loss']:>10.4f} "
              f"{m.get('cv_method', 'unknown'):>14} {m['timestamp']:>16}")

    print(f"\nModels directory: {MODELS_DIR}")

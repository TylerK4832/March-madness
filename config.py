"""Configuration and constants for March Madness ML pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# Seasons with detailed box scores available (2003+)
FIRST_DETAILED_SEASON = 2003

# Tournament typically starts around day 134-136
TOURNEY_START_DAY = 133

# Feature tier definitions for ablation studies
FEATURE_TIERS = {
    "seed_only": [
        "SeedDiff",
    ],
    "base": [
        "SeedDiff",
        "WinPctDiff",
        "ScoreMarginDiff",
        "FGPctDiff",
        "FG3PctDiff",
        "FTPctDiff",
        "ORDiff",
        "DRDiff",
        "AstDiff",
        "TODiff",
        "StlDiff",
        "BlkDiff",
    ],
    "base_massey": [
        "SeedDiff",
        "WinPctDiff",
        "ScoreMarginDiff",
        "FGPctDiff",
        "FG3PctDiff",
        "FTPctDiff",
        "ORDiff",
        "DRDiff",
        "AstDiff",
        "TODiff",
        "StlDiff",
        "BlkDiff",
        "OrdinalKenPomDiff",
        "OrdinalSagarinDiff",
    ],
    "full": [
        "SeedDiff",
        "WinPctDiff",
        "ScoreMarginDiff",
        "FGPctDiff",
        "FG3PctDiff",
        "FTPctDiff",
        "ORDiff",
        "DRDiff",
        "AstDiff",
        "TODiff",
        "StlDiff",
        "BlkDiff",
        "OrdinalKenPomDiff",
        "OrdinalSagarinDiff",
        "TOMarginDiff",
        "RebMarginDiff",
        "SoSProxyDiff",
    ],
    "efficiency": [
        "SeedDiff",
        "OEDiff",
        "DEDiff",
    ],
    "efficiency_4f": [
        "SeedDiff",
        "OEDiff",
        "DEDiff",
        "eFGPctDiff",
        "TOPctDiff",
        "ORPctDiff",
        "FTRateDiff",
    ],
    "adj_efficiency": [
        "SeedDiff",
        "AdjOEDiff",
        "AdjDEDiff",
    ],
    "adj_efficiency_4f": [
        "SeedDiff",
        "AdjOEDiff",
        "AdjDEDiff",
        "eFGPctDiff",
        "TOPctDiff",
        "ORPctDiff",
        "FTRateDiff",
    ],
}

# MLflow settings
MLFLOW_EXPERIMENT_NAME = "march-madness"
MLFLOW_TRACKING_URI = "mlruns"

# Bracket scoring rules (ESPN standard)
BRACKET_SCORING = {
    1: 10,   # Round of 64
    2: 20,   # Round of 32
    3: 40,   # Sweet 16
    4: 80,   # Elite 8
    5: 160,  # Final Four
    6: 320,  # Championship
}

# Monte Carlo simulations for pool-optimized bracket
MC_SIMULATIONS = 10000

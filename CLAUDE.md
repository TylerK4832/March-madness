# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Persona

You are an expert ML scientist specializing in sports analytics and probabilistic prediction. You approach this project with the rigor of a Kaggle grandmaster and the domain knowledge of a sports statistician.

### Principles

**Evaluation integrity above all.** Never trust a result until you understand the evaluation methodology. Default to sliding window CV (10yr window, 17 test seasons, 1,129 games). Be suspicious of small-sample results — 200 games is not enough to draw conclusions. Always report sample size alongside metrics.

**Simple models beat complex ones on small data.** NCAA tournament data is ~67 games/year. Regularized linear models will almost always outperform deep trees or neural nets. Only add complexity when you can demonstrate a statistically meaningful improvement on the full evaluation set.

**Features matter more than models.** Spend 80% of effort on feature engineering, 20% on model selection. The best features are ones that capture signal *complementary* to existing features, not redundant with them. Before adding a feature, ask: "Does this tell me something the model doesn't already know?"

**Beware multicollinearity.** With small datasets, correlated features add noise faster than signal. Fewer, orthogonal features > many correlated features. Always check if a new feature is redundant with an existing one (especially seed).

**Be honest about ceilings.** Tournament outcomes are inherently stochastic. A 1-seed loses to a 16-seed sometimes. Recognize when diminishing returns have set in and say so rather than chasing phantom improvements.

### Workflow

- Before running experiments, state your hypothesis and what you expect to see
- After running experiments, interpret results honestly — explain *why* something did or didn't work
- When something doesn't work, that's valuable information — document it
- Always update `notes/experiment-log.md` after completing any experiment, whether it succeeded or failed
- Never delete failed experiments from the log — negative results prevent repeating mistakes

## Project Overview

NCAA March Madness tournament prediction pipeline. Trains pairwise matchup models on historical Kaggle data, evaluates via sliding window cross-validation, and generates tournament brackets (chalk or Monte Carlo pool-optimized).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic test data (no Kaggle account needed)
python scripts/generate_synthetic_data.py

# Download real data (2003-2019, requires internet)
bash scripts/download_data.sh

# Train all model/tier combinations (8 runs: 2 models × 4 tiers)
python train.py

# Train specific model/tier
python train.py --model xgboost --tier full

# Generate bracket predictions
python predict.py --season 2024 --model xgboost --tier full
python predict.py --season 2024 --model xgboost --tier full --bracket pool

# View experiment results
mlflow ui
```

## Architecture

**Data flow:** Kaggle CSVs → `data_loader.py` → `feature_engineering.py` → model training → bracket generation

All matchups use **symmetric encoding**: lower TeamID is always "TeamA", label=1 means TeamA won. Features are computed as TeamA - TeamB differentials (except Massey ordinals which use B - A since lower rank = better).

### Feature Tiers (config.py `FEATURE_TIERS`)

Original tiers for ablation: `seed_only` → `base` (+ box score stats) → `base_massey` (+ KenPom/Sagarin) → `full` (+ derived: TOMargin, RebMargin, SoSProxy). Efficiency tiers: `efficiency` (Seed+OE+DE, best performer) → `efficiency_4f` (+ Four Factors) → `adj_efficiency` (opponent-adjusted) → `adj_efficiency_4f`.

### Models

Models implement `BaseModel` (src/models/base.py) with `fit()`, `predict_proba()`, `get_params()`, and optional `get_feature_importance()`. Registered in `MODEL_REGISTRY` (src/models/__init__.py). Currently: `logistic`, `xgboost`, and `stacking` (LR+RF+GBM ensemble).

### Evaluation

Sliding window CV (default) in `src/evaluation.py`. Also supports walk-forward and LOSO CV. Metrics: log loss, accuracy, upset accuracy. An "upset" is when the lower-seeded team wins.

### Bracket Generation (src/bracket.py)

- **Chalk:** always picks higher-probability team
- **Pool-optimized:** Monte Carlo (10k sims) maximizing expected ESPN scoring (10/20/40/80/160/320 per round)

### Data

CSV files go in `data/` (gitignored). Expected files: `MTeams.csv`, `MNCAATourneySeeds.csv`, `MRegularSeasonDetailedResults.csv`, `MNCAATourneyDetailedResults.csv`, `MMasseyOrdinals.csv`. Only seasons 2003+ are used (detailed box scores start there). Massey ordinals are filtered to pre-tournament rankings (day ≤ 133).

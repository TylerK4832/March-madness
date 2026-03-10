# Experiment Log

All results use **sliding window CV** (10-year training window, 17 test seasons 2008-2025, 1,129 tournament games) unless noted otherwise.

## Baseline

| Model | Tier | Accuracy | Log Loss | Notes |
|-------|------|----------|----------|-------|
| Logistic | seed_only | 69.62% | 0.5779 | True baseline — just seed differential |

## What's Working

### Tempo-adjusted efficiency (OE/DE)
The single biggest improvement over seed-only. Points per 100 possessions strips out tempo noise and captures true team quality. Complementary to seed — seed says "how good is this team?" while OE/DE says "how efficiently do they score/defend?"

| Model | Tier | Accuracy | Log Loss |
|-------|------|----------|----------|
| Stacking (LR+RF+GBM) | efficiency (Seed+OE+DE) | **71.04%** | 0.5702 |
| Logistic | efficiency | 70.06% | 0.5716 |
| Logistic | efficiency_4f (+ Four Factors) | 69.80% | 0.5731 |

### Stacking ensemble
LR + RF + GBM base models with logistic meta-learner and passthrough. Consistently +1% over logistic on the same features. The only model complexity that actually helped.

### Prediction blending (log loss only)
Averaging predictions from diverse models improves calibration but not accuracy.

| Blend | Accuracy | Log Loss |
|-------|----------|----------|
| 60% stacking_eff + 40% LR_base_massey | 70.77% | **0.5680** |
| 4-way average | 69.97% | 0.5682 |

Best log loss, but accuracy stayed at or below stacking alone.

## What Didn't Work

### Opponent-adjusted efficiency (AdjOE/AdjDE)
**Hypothesis:** Adjusting OE/DE for schedule strength (KenPom-style iterative algorithm) would improve predictions.
**Result:** Hurt accuracy. AdjOE/AdjDE became redundant with SeedDiff — both capture "team quality adjusted for schedule." Raw OE/DE works better *because* it captures something different from seed.

| Model | Tier | Accuracy | Log Loss |
|-------|------|----------|----------|
| Logistic | adj_efficiency (Seed+AdjOE+AdjDE) | 69.62% | 0.5716 |
| Logistic | adj_efficiency_4f | 69.80% | 0.5731 |

### Four Factors (eFG%, TO%, OR%, FT Rate)
Added on top of efficiency. Too correlated with OE/DE (they're components of efficiency). Added noise without new signal.

### Late-season momentum (last 10 games)
**Hypothesis:** Teams peaking in late February/March outperform season averages.
**Result:** No improvement. LateOE/LateDE didn't add signal beyond full-season OE/DE.

| Model | Tier | Accuracy |
|-------|------|----------|
| Stacking | efficiency_momentum (Seed+OE+DE+LateOE+LateDE) | 70.06% |
| Logistic | efficiency_momentum | 69.44% |

### Consistency / close-game performance
**Hypothesis:** Teams with low variance or strong close-game records are more reliable in single-elimination.
**Result:** No improvement. OEStd, CloseWinPct added nothing.

### XGBoost
Consistently underperformed logistic regression. ~660 training games per fold is too few for tree-based methods — they overfit.

| Model | Tier | Accuracy | Log Loss |
|-------|------|----------|----------|
| XGBoost | efficiency | 66.16% | 0.6247 |
| XGBoost | efficiency_4f | 66.96% | 0.6297 |

### More features in general
Kitchen-sink approaches (11+ features) always hurt. Multicollinearity is the enemy with small datasets. The best feature set is the smallest: Seed + OE + DE.

### Threshold tuning
Default 0.50 is already optimal. Tested 0.45-0.55 range, no improvement.

### Hyperparameter tuning on stacking
Tested: more trees, deeper RF, fewer CV folds, stronger regularization. Default config was already optimal.

## Evaluation Insights

### LOSO CV has temporal leakage (~3% inflation)
Leave-one-season-out trains on future data to predict past seasons. Inflates accuracy by ~3%. Not suitable for honest evaluation.

### Walk-forward CV is honest but high-variance
Training on all prior seasons, testing on last 3 (201 games). Single outlier seasons swing results dramatically. 2025 was 80.6% — an outlier that made everything look better than it was.

### Sliding window CV is the gold standard
10-year training window, 17 test seasons, 1,129 games. Stable, honest, large sample. This is what all numbers above use.

## To Investigate

- **External ratings as features:** Real KenPom/Sagarin/BPI rankings from Massey ordinals are already in the data but our current Massey tiers (base_massey) didn't beat efficiency. Could revisit with different combinations.
- **Conference strength signals:** Some conferences are systematically under/over-seeded. Conference tournament performance or regular season SOS could help.
- **Matchup-specific features:** Style matchups (e.g., pace mismatch, 3PT-heavy vs interior-heavy) rather than just overall quality.
- **Historical tournament performance:** Teams/coaches with deep March runs may have a systematic edge.
- **Larger training signal:** Current dataset is 2003-2025. Could incorporate pre-2003 data (compact results only, no box scores) for seed-based features.
- **Better calibration:** Isotonic regression or Platt scaling on CV predictions to improve log loss without changing accuracy.
- **Ensemble diversity:** Train models on truly different data views (e.g., one model on offensive stats only, another on defensive) and blend.

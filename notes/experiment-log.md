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

### Composite Massey ranking (H2)
Mean rank across 23 ranking systems with 18+ seasons of coverage. Provides robust "expert consensus" signal orthogonal to raw efficiency. Strongest single-feature addition we've found. Stronger regularization (C=0.05) helps logistic regression with this feature.

| Model | Tier | Accuracy | Log Loss |
|-------|------|----------|----------|
| LR (C=0.05) | efficiency_composite (Seed+OE+DE+Composite) | **70.95%** | **0.5661** |
| Stacking | efficiency_composite | 70.95% | 0.5697 |
| LR | Seed+Composite only | 70.33% | 0.5718 |

Key insight: composite rank helps LR close the gap to stacking because it provides a strong linear signal that stacking's trees were capturing non-linearly from other features.

### Model blending (H8)
Blending stacking(efficiency) + LR(efficiency_composite, C=0.05). These two models have complementary strengths: stacking wins accuracy, LR+composite wins log loss. **New best accuracy.**

| Blend | Accuracy | Log Loss |
|-------|----------|----------|
| 70% stacking + 30% LR(composite) | **71.30%** | 0.5679 |
| 60% stacking + 40% LR(composite) | 71.12% | 0.5674 |
| 30% stacking + 70% LR(composite) | 71.04% | **0.5663** |

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

### Strong regularization with richer features (H3)
**Hypothesis:** Stronger regularization (low C) could let logistic regression use more features without overfitting.
**Result:** Confirmed. C=0.05 is optimal for efficiency_composite (4 features). Also rescues larger feature sets: base_massey goes from 69.71% to 70.33% at C=0.05, full goes from 69.26% to 70.06% at C=0.01. However, no combination beats efficiency_composite at C=0.05. Stacking is insensitive to meta-learner regularization.

| Model | Tier | C | Accuracy | Log Loss |
|-------|------|---|----------|----------|
| LR | efficiency_composite | 0.05 | 70.95% | 0.5661 |
| LR | base_massey | 0.05 | 70.33% | 0.5698 |
| LR | full (17 features) | 0.01 | 70.06% | 0.5718 |

### Historical seed win rate via pipeline (H9)
**Hypothesis:** Per-fold historical seed matchup win rates (computed only from training data via pipeline) would improve predictions.
**Result:** Hurts across the board. Without leakage, per-fold sample sizes per seed differential are too small to be reliable. The earlier "improvement" (0.5702 → 0.5580 log loss) was entirely data leakage. Good validation that the pipeline prevents this.

### Isotonic calibration (H5)
**Hypothesis:** Post-hoc isotonic regression would improve log loss without changing accuracy.
**Result:** Hurts both metrics. Logistic regression already outputs well-calibrated probabilities. Isotonic regression overfits on the small training calibration set (~660 games).

| Model | Raw LL | Calibrated LL | Delta |
|-------|--------|---------------|-------|
| Stacking | 0.5702 | 0.5815 | +0.0113 (worse) |
| LR+composite | 0.5661 | 0.5807 | +0.0146 (worse) |

### Non-linear seed encoding (H1)
**Hypothesis:** SeedDiff² or historical seed win rates would capture the non-linear seed-win relationship.
**Result:** No accuracy improvement. SeedDiff², SeedDiff³, |SeedDiff| all hurt. Historical seed win rate showed large log loss improvement but was initially computed with data leakage; needs per-fold computation via pipeline (now possible). Logistic regression's sigmoid already captures non-linearity; stacking's trees learn it natively.

### More features in general
Kitchen-sink approaches (11+ features) always hurt. Multicollinearity is the enemy with small datasets. The best feature set is the smallest: Seed + OE + DE (+ CompositeRank).

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

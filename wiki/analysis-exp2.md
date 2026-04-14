# Experiment 2 Analysis — Balanced Sampling + Single-Target

## Starting Point

Experiment 1 results:
- T-number MAE = 0.70 (PASS), R² = 0.40
- Wind MAE = 18.2 kt (FAIL), R² = 0.42
- Pressure MAE = 39.4 hPa, R² = -1.13
- **Critical problem**: Prediction range collapse — model predicts T=[5.8, 8.9] but targets span [1.9, 8.0]

Root cause analysis in `analysis-exp1.md` identified three issues ranked by priority:
1. **Extreme class imbalance** (68.5% at T≥7.5) → prediction range collapse
2. **Discrete target regression mismatch** (7 Dvorak bins, not continuous) → stacked learning tasks
3. **Per-bin error scaling** inversely with bin frequency → confirms imbalance as root cause

## Interventions

### 1. Balanced Sampling (`dataset.py`)

Added `create_balanced_dataloaders()` with inverse-frequency `WeightedRandomSampler` across 5 T-number bins:

| Bin | Range | n (train) | Raw % | Weighted % |
|-----|-------|-----------|-------|------------|
| 0 | [1.0, 2.5) | ~130 | 0.3% | 20% |
| 1 | [2.5, 4.0) | ~820 | 2.2% | 20% |
| 2 | [4.0, 5.5) | ~2,200 | 5.8% | 20% |
| 3 | [5.5, 7.0) | ~5,800 | 15.4% | 20% |
| 4 | [7.0, 8.01) | ~28,800 | 76.3% | 20% |

Each bin contributes equal expected gradients per epoch, without discarding majority-class data (replacement sampling).

### 2. Single T-Number Output (`models.py`)

Added `TyphoonCNNv2` — same 4-block conv backbone as `TyphoonCNN`, but final FC layer outputs 1 value (T-number) instead of 3.

**Rationale**: Wind and pressure are deterministic functions of T-number via the Dvorak table. Predicting all three forces the model to learn the lookup table on top of T-number regression — two stacked tasks where the second has only 7 discrete output values. Eliminating the stacked task simplifies learning.

Wind and pressure are derived post-hoc via `dvorak_lookup()` in `evaluate.py`.

### 3. Fixed Scatter Plots (`evaluate.py`)

Added `dvorak_lookup()` function. `evaluate_model()` now detects single-output models and derives wind/pressure automatically. `generate_report()` generates proper predicted-vs-actual scatter plots.

## Results

| Metric | Exp 1 | Exp 2 | Change |
|--------|-------|-------|--------|
| T-number MAE | 0.70 | 0.73 | +0.03 (slight regression, expected) |
| T-number R² | 0.40 | 0.42 | +0.02 |
| Wind MAE | 18.2 kt | 16.2 kt | **-2.0 kt** |
| Wind R² | 0.42 | 0.39 | -0.03 |
| Pressure MAE | 39.4 hPa | 19.9 hPa | **-19.5 hPa** |
| Pressure R² | -1.13 | 0.32 | **+1.45** |

Training: CNNv2, batch_size=32, early stopped epoch 51, best epoch 36 (val loss 0.0140).

## Key Wins

1. **Prediction range collapse fixed**: Scatter plots confirm predictions now span the full T-number range (min prediction < 3.0, previously couldn't go below 5.8).

2. **Pressure dramatically improved**: From MAE=39.4 (R²=-1.13, worse than mean) to MAE=19.9 (R²=0.32). The 3-output model was predicting pressure below the true range floor; single-output + Dvorak lookup eliminated this.

3. **Wind improved by 2.0 kt**: From 18.2 to 16.2 — still above the 15 kt target, but significant progress.

## Why T-Number MAE Increased Slightly

Expected and acceptable. With balanced sampling, the model allocates capacity to minority bins (T < 5.5) where per-sample error is higher. The T-number MAE increases from 0.70 to 0.73 because the model no longer optimizes disproportionately for the easy majority class. This is a better-calibrated model even though the aggregate MAE is slightly worse.

## Remaining Gap

Wind MAE 16.2 kt is 1.2 kt above the 15 kt target. Scatter plots show the discrete 7-value Dvorak grid clearly — errors concentrate at bin boundaries where a small T-number error crosses a threshold and causes a large wind jump (e.g., T=7.5 boundary → 30 kt jump between 140 and 170).

This gap was addressed in Experiment 3 via isotonic calibration — see `analysis-exp3.md`.

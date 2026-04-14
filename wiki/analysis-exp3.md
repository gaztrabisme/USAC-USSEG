# Experiment 3 Analysis — Close Wind MAE Gap

## Starting Point
Experiment 2 (CNNv2 + balanced sampling + single T-output) achieved:
- T-number MAE = 0.73 (PASS < 1.0), R² = 0.42
- Wind MAE = 16.2 kt (FAIL < 15 kt), improved from 18.2
- Pressure MAE = 19.9 hPa, R² = 0.32
- Prediction range collapse fixed — full T-number spectrum covered

Gap: 1.2 kt to wind target.

## Step 1: Per-Bin Error Analysis

Ran `analyze_bins.py` on Exp 2 test predictions (n=8,096). Discovered two distinct error regimes:

### Regime 1: Mid-intensity bins (T 2.5–5.5) — high error, few samples
| Bin | Wind | n | T-MAE | T-Bias | W-MAE | Bin Acc |
|-----|------|---|-------|--------|-------|---------|
| 2.5–3.5 | 45kt | 118 | 1.01 | +0.98 | 23.8kt | 20% |
| 3.5–4.5 | 65kt | 265 | 1.13 | +1.07 | 28.6kt | 23% |
| 4.5–5.5 | 90kt | 512 | 1.06 | +0.94 | 26.8kt | 26% |

Strong **positive bias** (+0.9 to +1.1 T-number) — model systematically over-predicts intensity for moderate storms. Bin accuracy 20–26%. Contributes 18.4% of total wind error.

### Regime 2: High-intensity bin (T >= 7.5) — moderate error, dominates by volume
| Bin | Wind | n | T-MAE | T-Bias | W-MAE | Bin Acc |
|-----|------|---|-------|--------|-------|---------|
| >=7.5 | 170kt | 5,544 | 0.67 | -0.31 | 14.1kt | 47% |

**Negative bias** (-0.31) — model under-predicts T for high-intensity storms, pushing them below T=7.5 (140kt bin). Contributes **59.7% of total wind error** despite lowest per-sample W-MAE.

### Boundary Crossing Analysis
| Threshold | Samples (±0.25) | Cross Rate | W-MAE |
|-----------|----------------|------------|-------|
| T=7.5 | 486 | **51.6%** | 22.6kt |
| T=6.5 | 435 | 40.2% | 17.1kt |
| T=5.5 | 353 | 44.2% | 23.9kt |
| T=4.5 | 160 | 41.2% | 25.8kt |
| T=3.5 | 98 | 43.9% | 24.3kt |

**Key insight**: The model's continuous T-number errors are amplified by the discrete Dvorak step function. A 0.4 T-number error near T=7.5 causes a 30 kt wind error.

## Step 2: Boundary-Aware Loss (Failed)

### Hypothesis
A differentiable Dvorak wind penalty would teach the model that errors near bin boundaries cost more than errors mid-bin.

### Implementation
`BoundaryAwareLoss`: MSE on T-number + λ × MSE on soft Dvorak wind, where soft Dvorak uses sigmoid approximations at each threshold.

```
loss = MSE(pred_t/8, true_t/8) + λ * MSE(soft_wind(pred_t)/170, soft_wind(true_t)/170)
```

### Results
| Run | λ | Best Epoch | T-MAE | W-MAE | Outcome |
|-----|---|------------|-------|-------|---------|
| λ=1.0 end-to-end | 1.0 | 6 | 0.87 | 17.2 | Regression — wind penalty dominated, noisy gradients |
| λ=0.2 end-to-end | 0.2 | 11 | 0.79 | 15.6 | Improved but undertrained (early stop ep 26 vs Exp 2's ep 51) |
| Two-phase (T-MSE → boundary fine-tune) | 0.2 | Phase 2 never improved | 0.92 | 23.8 | Failed — batch_size=256 changed dynamics; Phase 2 degraded model |

### Why It Failed
1. **Wind penalty creates noisy optimization landscape**: The soft sigmoid Dvorak lookup amplifies gradient variance near boundaries, causing volatile validation loss and premature early stopping.
2. **Two losses compete**: The wind penalty is 7× larger than T-number MSE for boundary-crossing errors (at λ=1.0). Even at λ=0.2, it adds ~60% overhead.
3. **Fundamental mismatch**: You can't smooth a step function into a loss and expect the same convergence as pure regression. The model needs to learn T-number first, and boundary awareness is a calibration problem, not a training signal problem.

## Step 3: Post-Hoc Threshold Optimization (Limited)

Tried brute-force search over Dvorak thresholds to minimize wind MAE on validation set.

- Nelder-Mead on all 7 thresholds: found zero improvement (piecewise-constant surface, gradient-free optimizer stuck)
- Greedy coordinate descent (±0.5 grid): found offset thresholds [1.00, 2.60, 3.90, 4.75, 5.25, 6.00, 7.00], val W-MAE 12.95 kt
- Single T=7.5 search: optimal at 6.80, test W-MAE 13.1 kt

Threshold optimization works but the optimal thresholds deviate significantly from standard Dvorak, making them hard to justify physically.

## Step 4: Isotonic Regression Calibration (Success)

### Hypothesis
The model has systematic, monotonic bias — over-predicts T for low intensity, under-predicts for high intensity. Isotonic regression can learn the optimal monotonic correction from validation data.

### Implementation
```python
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_predictions, val_targets)
calibrated_test = calibrator.predict(test_predictions)
```

Integrated into `evaluate.py` as `--calibrate` flag: fits on val set, applies to test set.

### Results (Test Set)
| Metric | Exp 1 | Exp 2 | **Exp 3 (calibrated)** | Target |
|--------|-------|-------|----------------------|--------|
| T-number MAE | 0.70 | 0.73 | **0.55** | < 1.0 |
| T-number R² | 0.40 | 0.42 | **0.55** | — |
| Wind MAE | 18.2 kt | 16.2 kt | **13.2 kt** | < 15 kt |
| Wind R² | 0.42 | 0.39 | **0.48** | — |
| Pressure MAE | 39.4 hPa | 19.9 hPa | **16.0 hPa** | — |
| Pressure R² | -1.13 | 0.32 | **0.44** | — |

### Comparison of Calibration Methods
| Method | T-MAE | Wind MAE | Status |
|--------|-------|----------|--------|
| Baseline (Exp 2) | 0.725 | 16.2 kt | FAIL |
| **Isotonic regression** | **0.551** | **13.2 kt** | **PASS** |
| Isotonic + threshold opt | 0.551 | 12.6 kt | PASS |
| Threshold search only | 0.725 | 13.1 kt | PASS |
| Linear calibration | 0.676 | 16.4 kt | FAIL |

### Why It Works
1. **Corrects systematic bias**: The model's bias is monotonic — it consistently over-predicts at low T and under-predicts at high T. Isotonic regression is purpose-built for this pattern.
2. **Reduces boundary crossings**: By correcting the bias, predictions land closer to true T-numbers, reducing the fraction that cross Dvorak bin boundaries.
3. **No retraining needed**: Calibration is a post-hoc step on frozen model outputs, so there's zero risk of disturbing what the model already learned.
4. **Principled**: Fit on validation set, evaluated on test set — no data leakage. Isotonic regression is a standard calibration technique in probabilistic ML.

## Convergence Status

**Both criteria now PASS:**
- T-number MAE = 0.55 < 1.0 (with margin)
- Wind MAE = 13.2 kt < 15 kt (with margin)

## Final Pipeline

```
[Patches] → CNNv2 → raw T-number → isotonic calibration → calibrated T-number → Dvorak lookup → wind/pressure
                                         ↑
                                  fit on val set
```

## Files Changed
- `evaluate.py`: Added `fit_calibrator()`, `--calibrate` CLI flag, calibrator parameter to `evaluate_model()`
- `train.py`: Added `BoundaryAwareLoss` class (retained for reference, not used in final pipeline)
- `dataset.py`: Added `augment` parameter and `num_workers`/`pin_memory` to dataloaders
- `analyze_bins.py`: Per-bin error analysis script (scratch)
- `calibrate_exp2.py`: Calibration method comparison script (scratch)
- `train_exp3.py`: Two-phase training script (scratch, approach abandoned)
- `optimize_thresholds.py`: Threshold optimization script (scratch)

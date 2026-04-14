# Experiment 1 Analysis

## Root Cause Taxonomy

### RC1: Prediction Range Collapse (CRITICAL)
The model's **prediction range is severely compressed**:
- T-number: predicts 5.8–8.9, but targets span 1.9–8.0. Cannot predict T < 5.8.
- Wind: predicts 101–200 kt, targets span 30–170 kt. Cannot predict below 100 kt.
- Pressure: predicts 723–967 hPa, targets span 858–1000 hPa. Inverted — predicts *below* the true range floor.

The model has learned to predict "high intensity for everything" because 68.5% of samples are in the highest bin (T≥7.5 → 170kt, 858hPa). It's near-optimal on the dominant class but catastrophic on the minority.

### RC2: Extreme Class Imbalance (ROOT CAUSE of RC1)
| Wind bin | n | % |
|----------|---|---|
| 170 kt (T≥7.5) | 36,958 | 68.5% |
| 140 kt | 6,120 | 11.3% |
| 115 kt | 4,802 | 8.9% |
| 90 kt | 3,331 | 6.2% |
| 65 kt | 1,877 | 3.5% |
| 45 kt | 759 | 1.4% |
| 30 kt | 122 | 0.2% |

The MSE loss (even weighted) rewards accurate prediction on the 68.5% majority class. Minority classes contribute negligible gradients.

### RC3: Discrete Target Regression Mismatch (STRUCTURAL)
Wind and pressure each have only **7 unique values** — they're Dvorak lookup table bins, not continuous variables. The model regresses on continuous targets but the underlying relationship is a step function of T-number. Since wind/pressure add no information beyond T-number, the model must learn the Dvorak table *on top of* learning to predict T-number from images — two learning tasks stacked, with only 7 discrete output values for the second.

### RC4: Per-Bin Error Pattern (EVIDENCE)
| T-bin | n (test) | T-num MAE | Wind MAE | Bias direction |
|-------|----------|-----------|----------|----------------|
| [1.0, 2.5) | 19 | 3.64 | 78.1 kt | Massive over-prediction |
| [2.5, 4.0) | 230 | 2.80 | 64.9 kt | Severe over-prediction |
| [4.0, 5.5) | 665 | 1.70 | 46.5 kt | Moderate over-prediction |
| [5.5, 7.0) | 1,187 | 0.71 | 19.7 kt | Slight over-prediction |
| [7.0, 8.0] | 5,995 | 0.50 | 12.7 kt | Slight under-prediction |

The model's T-number prediction never goes below 5.8 — it physically cannot represent low-intensity storms. Error scales inversely with bin frequency, confirming imbalance as the root cause.

## Causal Model

```
Full-disk ADT scanning finds ~36 candidates/image
  → Most candidates are false positives at large ΔT
    → 62% of patches get T=8.0 (ADT saturation)
      → Training dominated by high-intensity samples
        → Model's prediction range collapses to [5.8, 8.9]
          → Wind/pressure inherit the bias, amplified by Dvorak step function
```

## Impact/Effort Ranking

| # | Intervention | Expected Impact | Effort | Priority |
|---|-------------|-----------------|--------|----------|
| 1 | **Downsample T=8.0 to balance classes** | HIGH — directly addresses RC1/RC2 | LOW — change dataset.py only | **DO FIRST** |
| 2 | **Predict T-number only, derive wind/pressure post-hoc** | HIGH — eliminates RC3, simplifies model task | LOW — change model output to 1, add Dvorak lookup in eval | **DO SECOND** |
| 3 | **Fix evaluate.py scatter plots** | MEDIUM — need real pred-vs-actual plots for diagnosis | LOW — pass predictions to generate_report | DO (quality of life) |
| 4 | Increase CNN depth/width | LOW — architecture is not the bottleneck | MEDIUM | SKIP for now |
| 5 | Augmentation (rotation, flip) | LOW-MEDIUM — may help generalize | LOW | Consider after #1-2 |

## Experiment Design: Experiment 2

### Changes
1. **Balanced sampling**: Cap T=8.0 class at N samples (where N = mean of other classes, ~3,000) or use inverse-frequency weighted sampler in DataLoader. This gives the model gradients from all intensity levels.
2. **Single-target prediction**: Output T-number only. Derive wind/pressure from T-number using the Dvorak table in evaluate.py. This eliminates the impossible task of regressing 7 discrete values.
3. **Fix scatter plots**: Pass predictions/targets to `generate_report()` for proper pred-vs-actual scatter.

### Measurement
- Before: T-num MAE=0.70, Wind MAE=18.2, Pressure MAE=39.4
- After: expect T-num MAE to increase slightly (0.8-1.0 as model spreads attention across all bins) but Wind MAE should drop significantly once derived from better-calibrated T-number predictions across the full range.

### Success Criteria
- T-number MAE < 1.0 on test set (must maintain)
- Wind MAE < 15 kt (via Dvorak lookup from predicted T-number)
- Prediction range covers full T-number spectrum (min pred < 3.0)

# Architecture

## Overview

Predicts typhoon intensity from satellite infrared imagery by learning to approximate the ADT (Advanced Dvorak Technique) algorithm. The model predicts a continuous T-number from a 240×240 temperature patch; wind speed and pressure are derived post-hoc via the Dvorak lookup table.

## Pipeline

```
GK-2A IR105 PNG + product.cbor
         │
    prepare_data.py          Scan fragment dirs, pair PNG+CBOR files
         │
    batch_adt.py             Run ADT on each image → ~36 storm candidates/image
         │                   Save 240×240 temperature patches + labels.csv
         │
    dataset.py               70/15/15 stratified split
         │                   Inverse-frequency WeightedRandomSampler (5 T-number bins)
         │
    train.py                 TyphoonCNNv2, single T-number output
         │                   TNumberMSELoss (T/8 scaling)
         │                   Adam + ReduceLROnPlateau + early stopping (patience=15)
         │
    evaluate.py --calibrate  Isotonic regression on val predictions → calibrated T-number
                             Dvorak lookup → wind (kt), pressure (hPa)
                             Scatter plots + evaluation_summary.json
```

## Data Format

- **Source**: GK-2A LRIT satellite data — PNG (IR105 channel, 2200×2200, 8-bit DN) paired with CBOR (calibration LUT: DN → Kelvin)
- **Calibration**: DN → Kelvin via LUT in `cbor['calibration']['IR105']`, then K → °C
- **Storm detection**: Cloud mask at T < -50°C, contour analysis, centroid extraction
- **Patches**: 240×240 float32 temperature arrays (°C), centered on detected storm candidates
- **Labels**: T-number (1.0–8.0), wind speed (25–170 kt), pressure (858–1009 hPa) — all from ADT

## Model: TyphoonCNNv2

Single T-number output. ~106K parameters.

```
Input: [1, 240, 240]
  Conv2d(1→16, 3×3) → BN → ReLU → MaxPool(2)     → [16, 120, 120]
  Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2)     → [32, 60, 60]
  Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2)     → [64, 30, 30]
  Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2)    → [128, 15, 15]
  AdaptiveAvgPool2d(1) → Flatten                    → [128]
  FC(128→64) → ReLU → Dropout(0.3)                 → [64]
  FC(64→1)                                          → [1]  (T-number)
```

Wind and pressure are **not predicted by the model**. They are derived from the predicted T-number using the Dvorak lookup table (7-bin step function) in `evaluate.py`.

## Calibration: Isotonic Regression

The raw CNN predictions have systematic monotonic bias: over-prediction at low T-numbers, under-prediction at high T-numbers. An isotonic regression is fit on validation set predictions → targets, then applied to test predictions before the Dvorak lookup. This corrects boundary-crossing errors that amplify through the discrete step function.

```
raw T-number → isotonic calibration (fit on val) → calibrated T-number → Dvorak lookup → wind/pressure
```

## Dataset Statistics

- 53,969 patches from 1,480 full-disk images (~36 candidates/image)
- Label distribution: 62% at T=8.0 (ADT saturation), long tail to T=1.8
- Balanced sampling addresses class imbalance during training
- Split: 37,778 train / 8,096 val / 8,095 test (stratified by T-number bin)

## Key Design Decisions

1. **Single T-number output** (not 3-output): Wind and pressure are deterministic functions of T-number via the Dvorak table. Regressing all three forces the model to learn the lookup table on top of learning T-number — two stacked tasks with only 7 discrete output values for the second.

2. **Balanced sampling** (not downsampling): WeightedRandomSampler with inverse-frequency weights ensures all T-number bins contribute equal expected gradients, without discarding majority-class data.

3. **Post-hoc calibration** (not boundary-aware training): Boundary-aware loss was attempted (differentiable soft-Dvorak penalty) but failed — noisy gradients caused premature early stopping. Isotonic regression achieves the same goal as a clean post-processing step.

4. **ADT approximation** (not ground truth): Labels come from the ADT algorithm applied to the same temperature data the model trains on. This is a learned ADT replacement, not a true intensity estimator. Ground-truth labels would require best-track data from JTWC/JMA.

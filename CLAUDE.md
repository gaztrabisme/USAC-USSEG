# Typhoon Intensity Estimation — V1

## What This Is

Neural network pipeline that predicts typhoon intensity (T-number, wind speed, pressure) from GK-2A satellite infrared temperature patches. Learns to approximate the ADT algorithm from temperature patches (labels are ADT-derived, not ground-truth observations).

## Pipeline

```
[11 Fragment Dirs (PNG+CBOR)] → prepare_data.py → batch_adt.py → dataset.py → train.py → evaluate.py --calibrate
```

## Files

| File | Purpose |
|------|---------|
| `applyADT.py` | ADT implementation — `load_temperature_data_from_png()`, `auto_detect_storm_and_apply_adt()` |
| `prepare_data.py` | Scan fragment dirs, pair GK2A_IR105 PNG + product.cbor files (`scan_and_pair()`) |
| `batch_adt.py` | Run ADT on all pairs, generate 240×240 patches + labels.csv. Two-phase sidecar pattern (resumable). |
| `dataset.py` | PyTorch Dataset with 70/15/15 stratified splits. `create_balanced_dataloaders()` for inverse-frequency weighted sampling. |
| `models.py` | TyphoonCNNv2 (~106K params, single T-number output — primary). Also TyphoonCNN (3-output, Exp 1) and TyphoonMLP (baseline only). |
| `train.py` | Training loop: `TNumberMSELoss`, Adam, ReduceLROnPlateau, early stopping. Checkpoint resume via `--resume`. Also contains `BoundaryAwareLoss` (Exp 3 attempt, not used in final pipeline). |
| `evaluate.py` | Per-target MAE/RMSE/R², scatter plots, summary JSON. `--calibrate` fits isotonic regression on val set for post-hoc T-number correction. `dvorak_lookup()` derives wind/pressure from T-number. |

## Running

```bash
# 1. Scan and pair satellite data (from 5080 machine or local mount)
python -c "from prepare_data import scan_and_pair; print(scan_and_pair('/path/to/GK-2A/data'))"

# 2. Generate labeled dataset
python batch_adt.py      # (call generate_dataset programmatically)

# 3. Train (CNNv2 = single T-number output + balanced sampling)
python train.py --model cnnv2 --data_dir dataset/ --checkpoint_dir checkpoints/ --epochs 100

# 4. Evaluate (--calibrate fits isotonic regression on val set)
python evaluate.py --model cnnv2 --data_dir dataset/ --checkpoint checkpoints/best_model.pt --output_dir results/ --calibrate
```

## Tests

```bash
python -m pytest tests/ -v
```

149 tests across `tests/test_data_pipeline.py`, `tests/test_ml_pipeline.py`, and `tests/test_v2_pipeline.py`.

## Key Parameters

- Temperature normalization: [-90, +40]°C mapped to [0, 1]
- Loss scaling: T-number / 8 (single-target MSE)
- Early stopping patience: 15 epochs on val loss
- Patch size: 240×240 single-channel float32
- Balanced sampling: 5 T-number bins with inverse-frequency weights

## Convergence Criteria (MET as of Exp 3)

- Primary: T-number MAE < 1.0 on test set — **0.55 (PASS)**
- Secondary: Wind speed MAE < 15 knots — **13.2 kt (PASS)**

## Experiment History

| Exp | Change | T-MAE | Wind MAE | Notes |
|-----|--------|-------|----------|-------|
| 1 | CNN, 3-output, uniform sampling | 0.70 | 18.2 kt | Prediction range collapse (T=[5.8,8.9]) |
| 2 | CNNv2 single-output, balanced sampling, Dvorak post-hoc | 0.73 | 16.2 kt | Range fixed, boundary errors remain |
| 3 | Isotonic calibration on Exp 2 model | **0.55** | **13.2 kt** | Both criteria met |

See `wiki/` for detailed analysis of each experiment.

---

## V2: Storm Bounding Box Localization

Separate pipeline in `v2/` that predicts the bounding box of the largest storm candidate in full-disk IR images.

### Pipeline

```
[PNG+CBOR] → applyADT.py (detect_largest_storm_bbox) → v2/batch_bbox.py → v2/dataset.py → v2/train.py → v2/evaluate.py
```

### V2 Files

| File | Purpose |
|------|---------|
| `v2/batch_bbox.py` | Batch process full-disk images → 512×512 .npy + normalized bbox labels |
| `v2/dataset.py` | BboxDataset with optional coordinated augmentation (disabled in best config) |
| `v2/models.py` | StormBboxNet — ResNet18 backbone (1-ch adapter), center+size head → minmax output |
| `v2/train.py` | SmoothL1/GIoU/cxywh loss, phased unfreezing, checkpoint resume, `--no-augment` |
| `v2/evaluate.py` | IoU metrics, histogram, center scatter plots |

### Running V2

```bash
# 1. Generate bbox labels (on 5080 machine)
python -c "from v2.batch_bbox import generate_dataset; from prepare_data import scan_and_pair; generate_dataset(scan_and_pair('/path/to/data'), 'dataset_v2/')"

# 2. Train (--no-augment is critical — augmentation hurts with small dataset)
python -m v2.train --data_dir dataset_v2/ --checkpoint_dir checkpoints_v2/ --epochs 100 --no-augment

# 3. Evaluate
python -m v2.evaluate --data_dir dataset_v2/ --checkpoint checkpoints_v2/best_model.pt --output_dir results_v2/
```

### V2 Convergence Criteria (primary MET as of Exp 4)

- Primary: Mean IoU > 0.3 on test set — **0.370 (PASS)**
- Secondary: IoU@0.5 accuracy > 50% — **36.5% (not yet met)**

### V2 Experiment History

| Exp | Change | Mean IoU | IoU@0.5 | Notes |
|-----|--------|----------|---------|-------|
| 1 | SmoothL1, minmax output, augment | 0.139 | 5.0% | Size variance collapse |
| 2b | Center+size parameterization | 0.256 | 17.6% | Valid boxes, better position |
| **4** | **No augmentation** | **0.370** | **36.5%** | **Primary criterion met** |

### V2 Tests

29 tests in `tests/test_v2_pipeline.py`.

---

## Dependencies

- PyTorch, torchvision
- numpy, scikit-learn, matplotlib
- rasterio, cbor2, opencv-python (for applyADT.py)

## Data

53,969 patches from 1,480 full-disk GK-2A images (11 fragment directories).
Dataset on 5080 machine (`bppc@100.106.185.34:projects/USAC/GK-2A/data/`).
Each fragment contains `IMAGES/GK-2A/<timestamp>/` dirs with:
- `GK2A_IR105_*.png` — 2200×2200 grayscale 8-bit DN infrared image
- `product.cbor` — calibration LUT (DN → Kelvin)

1,480 complete pairs, 957 orphans in each direction (skipped).
Skip dirs: `GK-2A_inference`, `GOES-18_cleaned`. Syncthing-ignored locally (`.stignore`).

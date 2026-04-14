# Typhoon Intensity Estimation

Predicts typhoon intensity from GK-2A satellite infrared imagery using a CNN that approximates the [Advanced Dvorak Technique](https://en.wikipedia.org/wiki/Dvorak_technique) (ADT).

Given a 240×240 cloud-top temperature patch, the model predicts a **T-number** (1.0–8.0), which maps to wind speed and sea-level pressure via the standard Dvorak lookup table.

## Results

| Metric | Value | Target |
|--------|-------|--------|
| T-number MAE | 0.55 | < 1.0 |
| Wind speed MAE | 13.2 kt | < 15 kt |
| Pressure MAE | 16.0 hPa | — |

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib rasterio cbor2 opencv-python

# Train
python train.py --model cnnv2 --data_dir dataset/ --checkpoint_dir checkpoints/ --epochs 100

# Evaluate (--calibrate applies isotonic regression for post-hoc correction)
python evaluate.py --model cnnv2 --data_dir dataset/ --checkpoint checkpoints/best_model.pt \
    --output_dir results/ --calibrate

# Tests
python -m pytest tests/ -v    # 123 tests
```

## How It Works

1. **Data**: GK-2A IR105 satellite images (PNG) + CBOR calibration files → temperature arrays
2. **Labeling**: ADT algorithm scans each full-disk image, detects storm candidates (~36/image), assigns T-number/wind/pressure
3. **Training**: CNNv2 (4 conv blocks, ~106K params) predicts T-number only, with balanced sampling to handle severe class imbalance (62% of patches at T=8.0)
4. **Calibration**: Isotonic regression fit on validation predictions corrects systematic bias (model over-predicts at low intensity, under-predicts at high intensity)
5. **Evaluation**: Calibrated T-number → Dvorak lookup table → wind speed (kt) and pressure (hPa)

> **Note**: Labels are ADT-derived from the same temperature data the model trains on. This is a learned ADT approximation, not a ground-truth intensity estimator.

## Project Structure

```
├── applyADT.py          # ADT algorithm implementation
├── prepare_data.py      # Scan and pair satellite files
├── batch_adt.py         # Batch ADT processing → patches + labels
├── dataset.py           # PyTorch Dataset, balanced sampling
├── models.py            # TyphoonCNNv2 (primary), CNN, MLP
├── train.py             # Training loop with checkpointing
├── evaluate.py          # Evaluation with isotonic calibration
├── tests/               # 123 tests (data pipeline + ML pipeline)
└── wiki/                # Design docs and experiment analyses
    ├── architecture.md
    ├── analysis-exp1.md # Exp 1: range collapse diagnosis
    ├── analysis-exp2.md # Exp 2: balanced sampling + single output
    ├── analysis-exp3.md # Exp 3: calibration (convergence achieved)
    ├── active-work.md
    └── log.md           # Session log
```

## Experiment History

| Exp | Key Change | T-MAE | Wind MAE |
|-----|-----------|-------|----------|
| 1 | 3-output CNN, uniform sampling | 0.70 | 18.2 kt |
| 2 | Single T-output, balanced sampling | 0.73 | 16.2 kt |
| 3 | + Isotonic calibration | **0.55** | **13.2 kt** |

See `wiki/analysis-exp*.md` for detailed reasoning behind each experiment.

## Data

53,969 patches extracted from 1,480 GK-2A full-disk infrared images across 11 data fragments. Dataset generation is handled by `prepare_data.py` → `batch_adt.py`. Raw satellite data is stored on a separate machine and not included in this repository.

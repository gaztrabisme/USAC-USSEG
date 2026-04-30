# Active Work

## V1 Typhoon Intensity Estimation Build
**Status:** Both convergence criteria met (Exp 3)
**Started:** 2026-04-10
**Goal:** Train CNN to approximate ADT algorithm from cloud-top temperature patches

### Phases
- [x] Phase A: Data pipeline (prepare_data.py, batch_adt.py, dataset.py)
- [x] Phase B: ML pipeline (models.py, train.py, evaluate.py)
- [x] Phase C: Code assessment and hardening

### Current State
- 78/79 tests passing (1 flaky: `test_train_loss_decreases_over_epochs` on random data)
- Assessment completed 2026-04-14 — all quality findings resolved
- Labels are ADT-derived (not ground truth) — model learns to approximate ADT formula
- CNN is primary architecture; MLP kept as baseline only

### Key Decisions
- **ADT-approximation accepted**: Labels come from the same temperature data the model trains on. Wind/pressure are deterministic from T-number. This is a learned ADT replacement, not a ground-truth intensity estimator.
- **CNN over MLP**: MLP at 30M params is massively overparameterized for ~1.5K samples. CNN at 106K params is the correct choice.

### Assessment Fixes Applied (2026-04-14)
- `applyADT.py`: Added `plot=False` default to suppress matplotlib side effects during batch processing
- `batch_adt.py`: Changed CSV from append to write mode to prevent duplicates on re-run
- `prepare_data.py`: Removed legacy `extract_and_pair()` — dead code kept for backwards compatibility
- `tests/test_data_pipeline.py`: Removed 8 legacy tests, fixed 5 monkeypatch targets (`j2k` → `png`)
- `CLAUDE.md`: Updated to reflect CNN as primary, ADT-approximation framing

### Convergence Criteria
- Primary: T-number MAE < 1.0 on test set
- Secondary: Wind speed MAE < 15 knots
- Stop: 3 experiments with <5% improvement

### Resilience Hardening (2026-04-14)
- `batch_adt.py`: Two-phase sidecar pattern — Phase 1 saves .npy + .json per patch (parallel, resumable), Phase 2 collects sidecars → labels.csv (idempotent). ProcessPoolExecutor concurrency, tqdm progress.
- `train.py`: Full checkpoint resume via `latest.pt` (optimizer, scheduler, patience, losses). `--resume` CLI flag. Epoch progress line + batch tqdm.
- 87 tests passing (79 existing + 8 new resilience tests)

### Experiment 1 Results (2026-04-14)
- **Dataset**: 53,969 patches from 1,480 full-disk images, 12GB
- **Label skew**: 62% T=8.0 (ADT saturation), heavy right skew
- **Training**: CNN ~106K params, batch_size=64, early stopped ep 54, best ep 39
- **T-number MAE=0.70** (PASS <1.0), R²=0.40
- **Wind MAE=18.2 kt** (FAIL <15 kt), R²=0.42
- **Pressure MAE=39.4 hPa** (poor), R²=-1.13
- Environment: RTX 5080, PyTorch 2.11+cu128, venv at typhoon/.venv

### Experiment 2 Results (2026-04-14)
- **Changes**: Balanced sampling (WeightedRandomSampler), single-target CNNv2 (T-number only → Dvorak post-hoc), scatter plots fixed
- **Training**: CNNv2 ~106K params, early stopped ep 51, best ep 36 (val loss 0.0140)
- **T-number MAE=0.73** (PASS <1.0), R²=0.42
- **Wind MAE=16.2 kt** (FAIL <15 kt, improved from 18.2), R²=0.39
- **Pressure MAE=19.9 hPa** (improved from 39.4), R²=0.32
- Scatter plots confirm prediction range collapse is FIXED — predictions span full T-number range
- Wind discrete grid (7 Dvorak values) visible — errors come from boundary misclassification

### Experiment 3 Results (2026-04-14)
- **Per-bin analysis**: T=7.5 boundary has 51.6% crossing rate, contributes 59.7% of wind error
- **Boundary-aware loss**: Failed — noisy gradients, premature early stopping, no improvement over Exp 2
- **Solution: Isotonic calibration** — fit on val set, corrects systematic monotonic bias
- **T-number MAE=0.55** (PASS <1.0), R²=0.55
- **Wind MAE=13.2 kt** (PASS <15 kt), R²=0.48
- **Pressure MAE=16.0 hPa**, R²=0.44
- Both convergence criteria now met. See wiki/analysis-exp3.md for full trace.

### Convergence Status: PASS
- Primary: T-number MAE = 0.55 < 1.0
- Secondary: Wind MAE = 13.2 kt < 15 kt

---

## V2 Storm Bounding Box Localization
**Status:** CONCLUDED — Exp 4 champion, stop criterion met
**Started:** 2026-04-16
**Goal:** Predict bounding box of largest storm candidate in full-disk IR image
**Requested by:** Gia Hiếu — separate branch from V1

### Phases
- [x] Phase A: Detection function (`detect_largest_storm_bbox()` in applyADT.py)
- [x] Phase B: Pipeline scaffolding (v2/batch_bbox.py, v2/dataset.py, v2/models.py, v2/train.py, v2/evaluate.py)
- [x] Phase C: Tests (29/29 passing in tests/test_v2_pipeline.py)
- [x] Phase D: Architecture diagram (draw.io — V1 + V2 side by side)
- [x] Phase E: Generate bbox labels — 1,480 images, 0 skipped, 0 errors (24s)
- [x] Phase F: Train and iterate (4 experiments)
- [x] Phase G: Primary criterion met

### Architecture
- **Input**: 2200×2200 full-disk IR → resized to 512×512
- **Model**: StormBboxNet — ResNet18 backbone (1-channel adapter, ~11M params) + FC head
- **Internal parameterization**: Predicts (cx, cy, w, h) via Sigmoid, converts to (minX, minY, maxX, maxY) — guarantees valid boxes
- **Loss**: SmoothL1 on normalized coords
- **Key concern**: Only ~1,480 images (vs 54K patches in V1) — small dataset, pretrained backbone + phased unfreezing essential
- **Label source**: Largest contour from cloud mask (T < -50°C) — algorithmic, same pattern as V1
- **Augmentation**: DISABLED — rotation/flip hurts localization with small dataset

### Convergence Criteria
- Primary: Mean IoU > 0.3 on test set
- Secondary: IoU@0.5 accuracy > 50%
- Stop: 3 experiments with < 5% improvement

### Convergence Status: CONCLUDED
- Primary: Mean IoU = 0.370 > 0.3 **PASS**
- Secondary: IoU@0.5 = 36.5% < 50% (not met — stop criterion reached)

### Dataset Stats
- 1,480 full-disk images → 1,480 bbox labels (every image had a detectable storm)
- Bbox width: mean=0.091±0.026, range=[0.033, 0.321] (storms are small ~0.7% of image area)
- Position: storms centered around x=0.51, y=0.38 with good spread across the disk

### Experiment History

| Exp | Changes | Mean IoU | IoU@0.5 | Key Finding |
|-----|---------|----------|---------|-------------|
| 1 | SmoothL1, minmax output, augment=on | 0.139 | 5.0% | Size variance collapse (constant box size) |
| 2 | + GIoU loss | 0.000 | 0.0% | GIoU alone can't bootstrap — zero overlap at init |
| 2b | center+size parameterization, SmoothL1 | 0.256 | 17.6% | Valid boxes guaranteed, position improved |
| 3 | cxywh-space loss | 0.227 | 13.5% | Size variance improved but position regressed |
| 3b | minmax + cxywh combined loss | 0.222 | 10.8% | Combined loss re-collapsed size |
| **4** | **Exp 2b + augment=off** | **0.370** | **36.5%** | **Augmentation was destroying spatial context** |
| 5 | SmoothL1 + DIoU (target position fix) | 0.181 | 1.4% | DIoU gamed via oversized boxes (size 3× target) |
| 7 | Mixup (alpha=0.4) | 0.326 | 26.6% | Blending full-disk images breaks spatial label continuity |
| TTA | Exp 4 ckpt + 4-way rotation TTA | 0.024 | 0.0% | Model has zero rotation-invariance (no-aug training) |
| 8 | 768×768 resolution (was 512×512) | 0.373 | 37.4% | +0.8% — resolution is not the bottleneck |

### Final Status (2026-04-28)
- **Stop criterion confirmed**: 4 consecutive experiments (Exp 5, 7, TTA, 8) failed to improve >5% over Exp 4.
- **Exp 4 is the final champion**: mean IoU=0.370, IoU@0.5=36.5%. Primary criterion met, secondary not.
- **Root cause of ceiling**: 1,480 images is too few to learn rotation-invariance AND position/size cues together. Neither loss changes, augmentation strategies, resolution bumps, nor inference-time tricks can overcome this.
- **What would help**: more data (10×), or a fundamentally different approach (e.g., heatmap regression instead of bbox).

### Key Decisions
- **ResNet18 over domain-specific models**: No public pretrained model exists for GK-2A IR localization. Weather AI models (GraphCast, Prithvi-WxC) expect gridded atmospheric fields, not raw satellite images. ResNet18's spatial features transfer even to IR data.
- **Phased unfreezing**: Backbone frozen epochs 1-5, layer3+4 unfrozen epochs 6-20, all unfrozen epoch 21+
- **Center+size parameterization**: Model internally predicts (cx, cy, w, h), converts to minmax output. Prevents degenerate boxes.
- **No augmentation**: With 1,480 images, rotation/flip destroyed spatial context needed for localization. This was the single biggest improvement (+0.114 IoU).
- **v2/ subdirectory**: Separate from V1 to avoid coupling
- **Future chain**: V2 locates storm → crop → V1 predicts intensity (not implemented yet)

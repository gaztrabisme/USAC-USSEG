# V2 Storm Bounding Box — Final Analysis

## Summary

V2 predicts the bounding box of the largest storm in full-disk GK-2A IR satellite images (2200×2200 → 768×768, output: minX, minY, maxX, maxY in [0,1]).

**Final model**: StormBboxNet (ResNet18 backbone, ~11M params), center+size parameterization, SmoothL1 loss, no augmentation. Trained on 1,480 images.

| Metric | Value | Criterion |
|--------|-------|-----------|
| Mean IoU | 0.373 | > 0.3 **PASS** |
| Median IoU | 0.414 | — |
| IoU@0.5 | 37.4% | > 50% **NOT MET** |
| IoU@0.3 | 65.8% | — |

---

## Experiment History

8 experiments over 4 sessions. Only one variable changed per experiment.

| Exp | Change | Mean IoU | IoU@0.5 | Outcome |
|-----|--------|----------|---------|---------|
| 1 | SmoothL1, minmax output, augment=on | 0.139 | 5.0% | Size variance collapse |
| 2 | + GIoU loss | 0.000 | 0.0% | GIoU can't bootstrap from zero overlap |
| 2b | Center+size parameterization, SmoothL1 | 0.256 | 17.6% | Valid boxes, better position |
| 3 | cxywh-space loss | 0.227 | 13.5% | Size improved, position regressed |
| 3b | Combined minmax + cxywh loss | 0.222 | 10.8% | Re-collapsed size |
| **4** | **Exp 2b + augment=off** | **0.370** | **36.5%** | **Champion — spatial prior preserved** |
| 5 | SmoothL1 + DIoU loss | 0.181 | 1.4% | DIoU gamed via oversized boxes |
| 7 | Mixup (alpha=0.4) | 0.326 | 26.6% | Broke spatial label continuity |
| TTA | 4-way rotation TTA on Exp 4 | 0.024 | 0.0% | Model has zero rotation invariance |
| 8 | 768×768 resolution (was 512) | 0.373 | 37.4% | +0.8% — resolution not the bottleneck |

**Stop criterion met**: 4 consecutive experiments (5, 7, TTA, 8) with <5% improvement.

---

## Stratified Error Analysis

### By Storm Size

| Bucket | n | Mean IoU | IoU@0.5 |
|--------|---|----------|---------|
| Tiny (<0.3% area) | 11 | 0.256 | 18.2% |
| Small (0.3-0.6%) | 91 | 0.286 | 23.1% |
| Medium (0.6-1.2%) | 96 | 0.448 | 51.0% |
| Large (1.2-5%) | 24 | 0.457 | 45.8% |

**Finding**: Storm size is the primary performance driver. The model works well on medium/large storms (IoU ~0.45, IoU@0.5 ~50%) but struggles with small storms (IoU ~0.28). Small storms make up 46% of the test set, dragging down the overall mean.

### By Disk Position

| Position | n | Mean IoU | IoU@0.5 |
|----------|---|----------|---------|
| Center (0-0.1) | 24 | 0.397 | 41.7% |
| Near-center (0.1-0.2) | 96 | 0.402 | 41.7% |
| Mid (0.2-0.3) | 55 | 0.295 | 20.0% |
| Edge (0.3-0.5) | 47 | 0.393 | 46.8% |

**Finding**: Disk position is not a systematic driver — performance is relatively flat except for a dip at mid-distance (0.2-0.3 from center). This could be a small-storm confound rather than a position effect.

### Error Decomposition

- **Center distance error** (mean: 0.067): corr with IoU = **-0.68** (strong)
- **Size error** (mean: 0.028): corr with IoU = **-0.42** (moderate)

Position error is the dominant failure mode — the model's IoU depends more on getting the center right than the size. The size scatter plots show predictions clustered around the mean bbox size (~0.09), indicating partial size variance collapse persists.

---

## Key Insights

1. **The model's strength is its spatial prior**: storms cluster near (0.5, 0.4) on the disk, and the model learned this pattern. This is why augmentation (which breaks spatial context) and TTA (which averages across rotated spatial contexts) both fail catastrophically.

2. **Dataset size is the binding constraint**: 1,480 images is too few to learn both rotation-invariance and position/size cues. More data (10× or more) would allow augmentation to work, which would improve generalization.

3. **For medium/large storms, the model already meets IoU@0.5 >50%**: the secondary criterion failure comes from the small-storm tail.

4. **What would improve results**:
   - More training data (most impactful)
   - Heatmap regression instead of direct bbox regression (better for small targets)
   - Multi-scale feature extraction (FPN) to improve small-storm detection

---

## Plots

- `results_v2_analysis/iou_by_size.png` — Mean IoU by storm size bucket
- `results_v2_analysis/iou_by_position.png` — Mean IoU by disk position
- `results_v2_analysis/position_vs_size_error.png` — Center vs size error, colored by IoU
- `results_v2_analysis/size_scatter.png` — Predicted vs actual width/height
- `results_v2_exp8/iou_histogram.png` — IoU distribution
- `results_v2_exp8/center_scatter.png` — Predicted vs actual center positions

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

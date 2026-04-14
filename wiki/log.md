# Wiki Log

## [2026-04-10] session | Initial setup
- Created wiki, ingested existing codebase (applyADT.py, read_j2k.py, read_cbor.py)
- Starting V1 build: 6 files across 2 phases with TDD+Adversarial patterns
- Data: 11 zip archives in Data/, J2K+CBOR pairs

## [2026-04-10] session | V1 build complete
- Implemented all 6 pipeline files with TDD+Adversarial patterns
- 87/87 tests passing (32 data pipeline + 55 ML pipeline)
- Adversarial review: fixed CRITICAL device mismatch in evaluate.py, cleaned dead code in batch_adt.py
- Verification: CNN has ~106K params (not ~250K as estimated — architecture is correct, estimate was off)
- Created CLAUDE.md for the project
- Next: extract real data, run batch ADT, train models

## [2026-04-14] session | Experiment 1 analysis
- Deep error analysis revealed prediction range collapse: model predicts T=[5.8,8.9] but targets span [1.9,8.0]
- Root cause: 68.5% of samples at Wind=170kt (T≥7.5), model optimizes for dominant class
- Wind/pressure have only 7 unique values each (Dvorak lookup bins) — regression on discrete targets
- Per-bin error: T<2.5 has MAE=3.64, T>7.0 has MAE=0.50 — inversely proportional to bin size
- Designed Experiment 2: balanced sampling + single T-number output + Dvorak post-hoc derivation
- See wiki/analysis-exp1.md for full taxonomy and experiment design

## [2026-04-14] session | First training run
- Synced pipeline code to 5080 (projects/USAC/GK-2A/typhoon/)
- Created venv with PyTorch 2.11+cu128 on RTX 5080 (16GB)
- batch_adt.py: 1,480 images → 53,969 patches in 24s (0 errors). ~36 detections/image.
- Label distribution: 62% at T=8.0 (ADT saturation), long tail from T=1.8-7.9
- CNN training: early stopped at epoch 54, best at epoch 39 (val loss 0.0127)
- Results:
  - T-number MAE=0.70 (PASS, target <1.0), R²=0.40
  - Wind MAE=18.2 kt (FAIL, target <15 kt), R²=0.42
  - Pressure MAE=39.4 hPa (poor), R²=-1.13
- Primary criterion met. Wind/pressure need improvement.

## [2026-04-14] session | Experiment 2 — balanced sampling + single-target
- Implemented 3 interventions from analysis-exp1.md:
  1. `dataset.py`: Added `create_balanced_dataloaders()` with inverse-frequency WeightedRandomSampler (5 T-number bins)
  2. `models.py`: Added TyphoonCNNv2 — same conv backbone, 1 output (T-number only)
  3. `evaluate.py`: Added `dvorak_lookup()` for post-hoc wind/pressure derivation; fixed scatter plots
  4. `train.py`: Added `TNumberMSELoss`, `--model cnnv2` CLI, single-target training path
- 122/123 tests passing (1 pre-existing flaky)
- Training: early stopped ep 51, best ep 36 (val loss 0.0140)
- Results:
  - T-number MAE=0.73 (PASS), R²=0.42 (up from ~0 effective)
  - Wind MAE=16.2 kt (FAIL, improved from 18.2), R²=0.39
  - Pressure MAE=19.9 hPa (improved from 39.4), R²=0.32
- Key win: prediction range collapse FIXED — scatter plot shows predictions spanning full T-number range
- Remaining gap: 1.2 kt to wind target, driven by Dvorak bin boundary errors

## [2026-04-14] session | Assessment and hardening
- Assessed pipeline against ML heuristics and engineering best practices
- CRITICAL finding: labels are ADT-derived from same input data (circular). Accepted as ADT-approximation.
- Fixed: plotting side effects in applyADT.py (plot=False default), CSV append→write in batch_adt.py
- Removed legacy extract_and_pair() and 8 associated tests
- Fixed 5 broken monkeypatch targets in batch_adt tests (j2k→png)
- Documented CNN as primary architecture, MLP as baseline only
- 78/79 tests passing (1 pre-existing flaky test on random synthetic data)

## [2026-04-14] session | Experiment 3 — close wind MAE gap
- Per-bin error analysis (analyze_bins.py): T=7.5 boundary has 51.6% crossing rate, ≥7.5 bin contributes 59.7% of wind error. Systematic bias: +1.0 for T<5.5, -0.31 for T≥7.5.
- Attempted boundary-aware loss (BoundaryAwareLoss with differentiable soft-Dvorak):
  - λ=1.0: regression (wind penalty dominated, T-MAE 0.87, W-MAE 17.2)
  - λ=0.2: marginal (T-MAE 0.79, W-MAE 15.6, undertrained)
  - Two-phase (T-MSE → boundary fine-tune): failed — Phase 2 never improved
  - **Conclusion**: boundary-aware training signal is too noisy for the discrete Dvorak structure
- Attempted Dvorak threshold optimization: Nelder-Mead found zero improvement (piecewise-constant landscape). Greedy grid search found shifted thresholds but deviate from physical Dvorak.
- **Solution: isotonic regression calibration** (fit on val, apply to test). Corrects systematic monotonic bias.
- Results (calibrated):
  - T-number MAE=0.55 (PASS), R²=0.55
  - Wind MAE=13.2 kt (PASS), R²=0.48
  - Pressure MAE=16.0 hPa, R²=0.44
- **Both convergence criteria met.**
- Integrated into evaluate.py as `--calibrate` flag with `fit_calibrator()` function
- Also improved DataLoader performance: num_workers=4, pin_memory, persistent_workers (GPU util 76%→99%)
- See wiki/analysis-exp3.md for full research trace

# Changelog

## Phase 0: Repo setup (2026-10-21)

- Moved the old Tkinter/Flask app into `legacy/` (binaries gitignored).
- Scaffolded the `src/beaconai/` package layout, `tests/`, `config/`, and support folders.
- Added uv (Python 3.11), ruff, pytest, pre-commit, Makefile, and GitHub Actions CI.
- Added a validated config loader (`src/beaconai/config.py`) with tests.
- Metrics: none yet (no ML work in this phase).

## Phase 1: Data pipeline (2026-10-21)

- `make data` (`dvc repro`) downloads DDXPlus from Figshare and writes model-ready arrays to `data/processed/ddxplus/`.
- Typed loaders for the evidence and condition catalogs, with English labels for every evidence and value code.
- Sparse encoding: binary evidences as 0/1, 0-10 scale evidences as one scaled column, categorical as one-hot, multi-choice as multi-hot, plus age buckets, scaled age, and sex. Every column has a readable label in `features.json`.
- Official train/validate/test split kept. Stratified dev subsample (`data.dev_sample`, at least 200 rows per class where possible).
- Unit tests cover each evidence type on tiny hand-made examples, not the real files.
- Metrics:
  - Full dataset: 1,025,602 train, 132,448 validate, 134,529 test patients.
  - Dev subsample (`dev_sample: 50000`): 50,000 rows per split, all 49 classes present in each split.
  - 922 features: 208 binary, 6 scale, 22 categorical one-hot, 676 multi-choice multi-hot, 8 age buckets, 1 scaled age, 1 sex.
  - 49 condition classes. About 22 active features per patient. Every patient has a non-empty ground-truth differential.

## Phase 2: Baseline models and evaluation (2026-10-21)

- Three models behind one wrapper (fit, predict_proba, save, load): logistic regression, XGBoost, PyTorch MLP. All hyperparameters and grids in `config/base.yaml`.
- Grid search (11 trials) on the dev subsample, validation split only. Every trial logged to MLflow with params, metrics, a confusion plot, and the model. Winners refit on the full training set.
- Champion registered as `beaconai-classifier` (alias `champion`) by validation top-3 accuracy. All three models tie at 1.000, so ties are broken by top-1, then macro F1 (`train.tie_breakers`): XGBoost.
- Test split scored once on the full dataset. A lock file blocks repeat scoring; scoring on the dev subsample is refused.
- New metric: overlap with the DDXPlus reference differential (precision@k, raw recall@k, normalized recall@k = hits / min(k, |D|)). Normalized recall@3 is the headline.
- macOS: torch and XGBoost now share torch's OpenMP runtime (set by the Makefile); two runtimes in one process crashed training.
- Metrics (test, full data, 134,529 patients):
  - XGBoost (champion): top-1 0.998, top-3 1.000, top-5 1.000, macro F1 0.997, DDx normalized recall@3 0.697. Fit 184 s, 0.99 ms per patient.
  - Logistic regression: top-1 0.996, macro F1 0.995, DDx normalized recall@3 0.703. Fit 14 s, 0.08 ms.
  - MLP: top-1 0.996, macro F1 0.996, DDx normalized recall@3 0.693. Fit 54 s, 0.05 ms.
  - Main confusion: acute rhinosinusitis predicted as chronic (13.4% of acute cases). Only safety-relevant confusion: unstable angina predicted as stable angina (1.5%).
- With full evidence the task is close to saturated; partial-evidence performance (Phase 8) and calibration (Phase 3) are where the models should be compared next. See `docs/phase2_analysis.md`.

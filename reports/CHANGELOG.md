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

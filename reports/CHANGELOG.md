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

## Phase 3: Calibration and abstention (2026-10-21)

- Calibrators (temperature scaling, isotonic, or none) for XGBoost and logistic regression, fit on half of validation and scored on the other half. Calibrator state is plain JSON.
- New partial-evidence validation set: each patient keeps the initial evidence plus 3 random other evidences.
- Champion re-picked by the lowest partial-evidence ECE, with top-3 floors of 0.99 (full evidence) and 0.65 (partial evidence): **logistic regression + temperature scaling** replaces raw XGBoost (`beaconai-classifier` version 8, alias `champion`).
- One prediction interface: `CalibratedClassifier.predict_differential` returns the top 3 conditions with calibrated probabilities, or `Abstain` with a reason. Abstention threshold 0.84.
- Every (model, method) pair logged to MLflow (`beaconai-phase3` experiment) with metrics, reliability diagrams, and the calibrated model.
- Metrics (full data, held-out validation, 66,223 patients):
  - Champion ECE: full evidence 0.0006, partial evidence 0.082 (raw 0.097). Top-3: full 1.000, partial 0.761.
  - XGBoost ECE: full 0.0003, partial 0.556 (top-1 on partial 0.282). Calibration does not fix it.
  - "Not asked is read as no": the champion loses 23.9 points of top-3 accuracy from full to partial evidence.
  - At threshold 0.84: 99.0% of full-evidence patients answered (top-3 1.000); 44.3% of partial-evidence patients answered (top-1 0.904, top-3 0.952).
- Registry aliases split: `champion` / `dev` now mean the calibrated model and are moved only by `make calibrate`; raw winners from `make train` use `raw-champion` / `raw-dev`.
- Caveat: validation was also used for Phase 2 tuning and early stopping, so these held-out numbers are somewhat optimistic until confirmed on test.
- Decision record: `docs/phase3_calibration.md`.

## Phase 3b: Evidence dropout, partial-evidence calibration, final test evaluation (2026-10-22)

- Logistic regression and XGBoost retrained with random evidence dropout. Each training patient is kept and also added once with evidences dropped. Each copied row draws a dropout rate from 0.5 to 0.95, and the initial evidence, age and sex are always kept.
- Calibrators can now be fit on partial-evidence rows. 20 candidates (plain and dropout models × full and partial calibration rows × method), same selection rule as Phase 3.
- Champion: **logistic regression + dropout + isotonic calibration fit on partial rows** (`beaconai-classifier` version 13, alias `champion`).
- New evidence-count curve in `reports/calibration.md`: top-3 and ECE with the initial evidence plus 0 to 12 others, Phase 3 vs 3b champion.
- Abstention threshold 0.60: the 95% top-3 rule is now applied to patients who give only the initial evidence (`abstention.suggest_on_extra: 0`).
- `make eval` is now the one-time test evaluation of the calibrated champion only, locked per evaluation name (`reports/test_locks/classifier_final.json`). Phase 2's evaluator no longer touches test.
- Metrics (validation held-out half, initial + 3 evidences): top-3 0.761 to 0.983, top-1 0.609 to 0.898, ECE 0.082 to 0.0027. Full evidence: top-3 1.000, ECE 0.0009. Initial evidence only: top-3 0.188 to 0.585.
- **Test (scored once, 134,529 patients):** full evidence top-1 0.995, top-3 1.000, ECE 0.0010; initial + 3 evidences top-1 0.899, top-3 0.984, ECE 0.0028; at threshold 0.60, 89.6% of partial-evidence patients answered with top-3 0.999.
- Decision record: `docs/phase3b_dropout.md`.

## Phase 4: Explainability (2026-10-22)

- Exact explanations for the calibrated logistic regression champion. Each contribution is `coef[c, j] * (x[j] - m[j])` in logit space, before isotonic calibration, and baseline score plus contributions equals the model's score. A test cross-checks every contribution against `shap.LinearExplainer` with the same baseline.
- Baseline patient: no evidence, and the training-set mean for age and sex (average age 39.7, 49% male, from 1,025,602 training patients; `reports/explain_baseline.json`). Age and sex are each reported as one grouped contribution.
- `ChampionExplainer.explain_evidence` returns the prediction, the top 5 supporting and top 3 opposing contributors per shown condition, and a contrast "why A rather than B" for the top two.
- Calibrated ties are now broken by the model's own score in `predict_differential` and in explanations. Before this, slots 2 and 3 and the contrast's runner-up were often chosen by label order: 99.9% of held-out patients had a tied third slot.
- Metrics (full data, held-out validation): calibration changes the top-1 for 0.05% (full evidence) and 1.5% (partial evidence) of patients, and the shown set (conditions at 1% or more) for 0.05% and 4.0%. Latency end to end: median 0.6 ms, p95 0.7 ms (budget 50 ms). Automatic sanity flags: 2 of 80 top features (croup age, expected).
- Global view plots for 8 conditions, plus an author review (not clinically validated) noting a DDXPlus quirk: skin and rash questions rank high for pneumonia and influenza.
- MLflow tags unified for dropout models (`phase` "3b", `model_type` "logreg+dropout").
- Decision record: `docs/phase4_explainability.md`.

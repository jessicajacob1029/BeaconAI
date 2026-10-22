# Phase 3b decision record: evidence dropout, partial-evidence calibration, final test

> BeaconAI is decision support, not diagnosis. All numbers below come from synthetic data (DDXPlus) and say nothing about real patients.

Validation numbers are from `reports/calibration.md` (full dataset, the same held-out half of validation and the same partial-evidence rule as Phase 3). Test numbers are from `reports/final_evaluation.md`. That file holds the one and only test evaluation of this classifier, run on 2026-10-22.

## Decision

- **Champion: logistic regression retrained with evidence dropout, plus isotonic calibration fit on partial-evidence rows.** Registered as `beaconai-classifier` version 13, alias `champion`.
- **Abstention threshold: 0.60.**
- **Test (scored once): partial-evidence top-3 0.984, ECE 0.0028; full-evidence top-3 1.000, ECE 0.0010.**
- From here on, any classifier change is validated on validation data only. The test lock for this evaluation is `reports/test_locks/classifier_final.json`.

## What changed

Phase 3 found that the models read every unasked question as "no". With only the initial evidence plus 3 others, the Phase 3 champion's top-3 accuracy fell to 0.761.

1. **Evidence dropout during training.** Each training patient is kept as is and also added once with evidences dropped. The initial evidence, age and sex are always kept. Every other evidence (a whole question with all its values) is dropped with a probability that each copied row draws from 0.5 to 0.95. Training therefore covers both few-symptom and many-symptom patients, about 2.05M rows on full data. Hyperparameters are the Phase 2 winners. Early stopping uses only the calibration fit half, never the held-out half. (It uses the same partial-evidence rows the partial-row calibrators are later fit on. That makes no difference here: XGBoost ran to its round cap, and logistic regression does not early-stop.)
2. **Calibration fit on partial-evidence rows.** As well as fitting calibrators on the calibration half as is (the Phase 3 behavior), they are also fit on its partial-evidence version.

## Results on held-out validation (66,223 patients)

The four corners of the comparison, best method in each:

| Base model | Calibrated on | Best pair | ECE partial | Top-3 partial | ECE full | Top-3 full |
|---|---|---|---:|---:|---:|---:|
| Phase 2 models (Phase 3 baseline) | full rows | logreg + temperature | 0.082 | 0.761 | 0.0006 | 1.000 |
| Phase 2 models | partial rows | logreg + temperature | 0.023 | 0.761 | 0.0028 | 1.000 |
| Dropout retrain | full rows | xgboost + isotonic | 0.0046 | 0.982 | 0.0005 | 1.000 |
| **Dropout retrain** | **partial rows** | **logreg + isotonic** | **0.0027** | **0.983** | **0.0009** | **1.000** |

What this shows:

- **The comparison slightly favors the baseline.** The Phase 2 models early-stopped on all of validation, including the held-out half, which makes the baseline numbers a little optimistic. The dropout models never saw the held-out half. The conclusions below hold despite that.
- **Dropout does the heavy lifting.** It raises partial-evidence top-3 from 0.761 to 0.983 and top-1 from 0.609 to 0.898. Calibration cannot do that, because it can only reshape probabilities, not fix the ranking.
- **Partial-row calibration alone helps calibration but not accuracy,** and it can damage full-evidence calibration. For the Phase 2 XGBoost, temperature scaling fit on partial rows pushes full-evidence ECE to 0.85. The dropout models are well calibrated on both settings, so fitting on partial rows costs them almost nothing on full evidence.
- **XGBoost and logistic regression are close after dropout.** XGBoost has slightly higher top-3 (0.989 vs 0.983) but higher ECE (0.0036 vs 0.0027). Logistic regression wins under the selection rule (lowest partial-evidence ECE among pairs meeting both top-3 floors). It is also faster (82 s to train vs 580 s) and easier to explain. XGBoost hit its 400-round cap without early stopping on both dev and full data, so it may improve with more rounds; that is a follow-up, not a reason to change the rule.

### How accuracy grows with what the patient tells us

Held-out patients keep the initial evidence plus n other random evidences:

| Other evidences | Top-3 (Phase 3) | Top-3 (Phase 3b) | ECE (Phase 3) | ECE (Phase 3b) |
|---:|---:|---:|---:|---:|
| 0 | 0.188 | 0.585 | 0.211 | 0.084 |
| 1 | 0.445 | 0.848 | 0.159 | 0.031 |
| 2 | 0.633 | 0.949 | 0.115 | 0.007 |
| 3 | 0.761 | 0.983 | 0.082 | 0.003 |
| 5 | 0.908 | 0.999 | 0.046 | 0.002 |
| 8 | 0.980 | 1.000 | 0.026 | 0.002 |
| 12 | 0.998 | 1.000 | 0.010 | 0.002 |

The 3b champion first passes 95% top-3 with 3 extra evidences (0.949 with 2). Its calibration is good from 2 extra evidences on (ECE 0.007 or lower), but weaker with the initial evidence alone (ECE 0.084). Every point beats the Phase 3 champion. Accurate, honest probabilities after a few answers are what the Phase 8 question selector needs.

## Abstention threshold

The Phase 3 rule is the smallest threshold whose answered patients reach 95% top-3 accuracy. On the initial-plus-3 set that rule now gives 0: those patients are already above 95% at every threshold. The threshold only matters for the hardest case, a patient who has given just the initial evidence. So the rule is now computed there (`abstention.suggest_on_extra: 0`), which gives **0.60**:

| Known evidences | Answered at 0.60 | Top-1 on answered | Top-3 on answered |
|---|---:|---:|---:|
| Initial only (validation) | 30.1% | 0.751 | 0.953 |
| Initial + 3 (validation) | 89.7% | 0.953 | 0.999 |
| Full record (validation) | 99.9% | 0.995 | 1.000 |

For comparison, the Phase 3 threshold of 0.84 would answer only 16.0% of first-turn patients (same held-out rows; computed when choosing the threshold). The threshold was chosen and is reported on the same held-out rows; test agrees closely (89.6% vs 89.7% of initial-plus-3 patients answered). An abstention is not a dead end: in Phase 8 it becomes "ask the next most informative question".

## Final test evaluation (134,529 patients, scored once)

| Evidence | Top-1 | Top-3 | Top-5 | Macro F1 | ECE | Answered at 0.60 | Top-3 on answered |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.995 | 1.000 | 1.000 | 0.994 | 0.0010 | 100.0% | 1.000 |
| Initial + 3 | 0.899 | 0.984 | 0.993 | 0.905 | 0.0028 | 89.6% | 0.999 |

Test agrees closely with validation (partial top-3 0.984 vs 0.983, ECE 0.0028 vs 0.0027). So choosing the champion and threshold on validation did not overfit noticeably. This is the first and only look at test for this classifier. `make eval` now refuses to run it again. (Phase 2 scored its raw models on test once, on 2026-10-21. That record, `reports/model_comparison.md` with `reports/test_eval_lock.json`, is kept as history; no code uses that lock any more.)

## Notes and follow-ups

- **Differential overlap on full evidence dropped** (DDx nR@3 0.527 on test, vs about 0.70 for the Phase 2 models). With a complete record the champion is near-certain, so the 2nd and 3rd slots carry little probability and are not chosen to match the DDXPlus reference differential. The Phase 5 `min_display_prob` rule (hide items below 1% in the API) covers how this is shown. If a reference-like differential matters later, it needs its own objective.
- **The yes/no/not-asked encoding was not built.** It would need a second column per evidence, changes to the Phase 1 encoder and to `PatientEvidence` (callers must say which questions were answered "no"), and a simulation of which absent evidences were asked during training. Phase 8 is the natural place, because the agent knows exactly which questions it asked.
- **XGBoost + dropout used all 400 rounds.** A larger round budget may help it. It would be validated on validation only.
- **Phase 2's `make eval` no longer touches test.** `make eval` is now this one-time champion evaluation, with one lock per evaluation name in `reports/test_locks/`. A later evaluation (for example a Phase 8 agent simulation) gets its own name and its own one-time run.

## Method

- Validation split per class into a calibration fit half and a held-out half (`calibration.fit_fraction: 0.5`, seed 42). Both are identical to Phase 3.
- Partial evidence: the initial evidence plus 3 random other evidences, with age and sex kept. The generator was refactored to share code with dropout, and a golden test proves its output is unchanged from Phase 3.
- 20 candidates: 4 base models (logreg, xgboost, each plain and dropout) × (raw once, plus temperature and isotonic fit on full rows and on partial rows). Selection: lowest partial-evidence ECE among pairs with top-3 at least 0.99 on full evidence and at least 0.65 on partial evidence.
- Every run is in MLflow: `beaconai-phase3b` (dropout retraining), `beaconai-phase3` (calibration), `beaconai-final` (the test evaluation).

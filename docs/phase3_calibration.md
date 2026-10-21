# Phase 3 decision record: calibration, champion, and abstention

> BeaconAI is decision support, not diagnosis. All numbers below come from synthetic data (DDXPlus) and say nothing about real patients.

Numbers are from `reports/calibration.md` (full dataset, validation split, 2026-10-21). Calibrators were fit on 66,225 validation patients, and every number here comes from the other 66,223. The test split was not used.

## Decision

- **Champion: logistic regression + temperature scaling** (`beaconai-classifier` version 8, alias `champion`). This replaces the Phase 2 champion (raw XGBoost).
- **Abstention threshold: 0.84.**

## How the champion was chosen

The rule: among (model, method) pairs whose held-out top-3 accuracy is at least 0.99 on full evidence and at least 0.65 on partial evidence, pick the lowest partial-evidence ECE. Partial evidence means each patient keeps the initial evidence plus 3 other random evidences, which is closer to what a user gives us than a complete DDXPlus record.

| Model | Method | Temperature | ECE full | ECE partial | Top-1 partial | Top-3 partial | Eligible |
|---|---|---:|---:|---:|---:|---:|---|
| XGBoost | none | | 0.0003 | 0.556 | 0.282 | 0.707 | yes |
| XGBoost | temperature | 1.00 | 0.0004 | 0.556 | 0.282 | 0.707 | yes |
| XGBoost | isotonic | | 0.0004 | 0.577 | 0.284 | 0.374 | no (partial floor) |
| Logistic regression | none | | 0.0004 | 0.097 | 0.609 | 0.761 | yes |
| **Logistic regression** | **temperature** | **1.05** | **0.0006** | **0.082** | **0.609** | **0.761** | **yes** |
| Logistic regression | isotonic | | 0.0003 | 0.187 | 0.633 | 0.798 | yes |

What this shows:

- **With full evidence, every option is already well calibrated** (ECE 0.0003 to 0.0006). Calibration has almost nothing to fix there, which is why the fitted temperatures are about 1.0.
- **With partial evidence, XGBoost is dangerously overconfident.** Its top choice is right only 28% of the time, yet it puts most of its probability on that choice (ECE 0.556). Logistic regression is right 61% of the time and much more honest about its uncertainty (ECE 0.097 raw, 0.082 after temperature scaling).
- **Calibration fitted on full evidence cannot repair partial-evidence behavior.** The temperatures are learned where the models are already calibrated, so they barely move partial-evidence probabilities. Isotonic calibration makes partial evidence worse for both models, and for XGBoost it also breaks the ranking (top-3 0.374).
- **The Phase 2 margin for XGBoost (top-1 0.998 vs 0.996) was measured with complete records.** That is not how the system will be used. With partial evidence, logistic regression is both more accurate and far better calibrated.
- On the dev subsample, XGBoost was much less overconfident on partial evidence (ECE 0.072). The full-data XGBoost, trained on 20 times more patients, learned sharper rules that fail harder when answers are missing. Logistic regression won on both datasets.

## Abstention threshold

BeaconAI abstains when the calibrated probability of its top condition is below the threshold (equal is answered). Thresholds were compared on the held-out patients with the champion:

| Threshold | Answered (partial) | Top-1 on answered | Top-3 on answered | Answered (full) |
|---:|---:|---:|---:|---:|
| 0.30 | 86.2% | 0.678 | 0.821 | 100.0% |
| 0.50 | 67.9% | 0.778 | 0.893 | 100.0% |
| 0.60 | 60.5% | 0.818 | 0.913 | 99.8% |
| 0.70 | 53.9% | 0.854 | 0.929 | 99.5% |
| 0.80 | 47.2% | 0.887 | 0.943 | 99.2% |
| **0.84** | **44.3%** | **0.904** | **0.952** | **99.0%** |
| 0.90 | 39.3% | 0.929 | 0.964 | 98.7% |

**0.84** is the smallest threshold at which answered partial-evidence patients reach 95% top-3 accuracy (`abstention.target_accuracy`). Why this trade-off:

- Showing a confident differential that misses the true condition is worse than saying more information is needed, especially for a triage tool.
- Abstaining is not a dead end. In Phase 8 an abstention becomes "ask the next most informative question", so a cautious threshold costs a few questions, not a failed conversation.
- With complete answers the threshold rarely matters: 99% of patients are still answered.

## Limitation: "not asked" is read as "no"

The models were trained on complete records, where a missing evidence means the patient does not have it. In a conversation, most evidences are simply not asked yet, and the models read them as "no". For the champion, moving from full to partial evidence:

- top-3 accuracy drops from 1.000 to 0.761 (23.9 points)
- ECE rises from 0.0006 to 0.082

For raw XGBoost the drop is 29.3 points of top-3 accuracy and ECE rises to 0.556. This is the main gap to close before Phase 8. A note in the roadmap asks for retraining with random evidence dropout, or a yes/no/not-asked encoding, with partial-evidence ECE compared against this baseline.

## Open questions

- **Fitting calibrators on partial evidence.** Calibrators here are fit on full-evidence validation rows, as specified. Fitting them on a partial-evidence version of the fit rows would target the setting we care about. That is worth testing after the retraining above, because both change the same probabilities.
- **Near-zero items in a differential.** With complete evidence the champion is often near-certain, so the 2nd and 3rd items can have probabilities like 4e-12. Which conditions fill those slots then comes from class order, not from the model, but they still look like real alternatives. A display rule (for example, hide items below 1%) belongs with the API and UI work.

## Method

- Validation was split per class into a fit part and a held-out part (`calibration.fit_fraction: 0.5`, seed 42).
- ECE is top-label ECE with 15 equal-width bins, weighted by bin size. NLL is reported alongside it in `reports/calibration.md`.
- Partial evidence: per patient, the initial evidence plus 3 other random evidences (an evidence is one question with all its selected values), with age and sex kept. Same seed for every model and method.
- Every (model, method) pair is a nested MLflow run in the `beaconai-phase3` experiment with its metrics and the calibrated model. The champion is registered with tags for its calibration method and held-out ECE.
- `CalibratedClassifier.predict_differential(PatientEvidence(evidence=[...], age=..., sex=...))` returns the top 3 conditions with calibrated probabilities, or an `Abstain` with a reason. An unknown evidence code raises a clear `ValueError`.

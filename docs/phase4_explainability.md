# Phase 4 decision record: explanations for the calibrated logistic regression champion

> BeaconAI is decision support, not diagnosis. All numbers below come from synthetic data (DDXPlus) and say nothing about real patients.

The generated report is `reports/explainability.md` (`make explain`, full dataset, champion `beaconai-classifier` version 13). The author review of the global view is `docs/phase4_sanity_review.md`, embedded at the end of the report. It is not a clinical validation.

## What was built

- **Exact explanations.** The champion's base model is a multinomial logistic regression, so its score for condition c is a sum: `intercept[c] + sum_j coef[c, j] * x[j]`. Each feature's contribution is `coef[c, j] * (x[j] - m[j])`, measured against a baseline patient m. Baseline score plus contributions equals the model's score exactly; tests check this, and check that each contribution matches `shap.LinearExplainer` with the same baseline.
- **Baseline patient.** No evidence (every symptom and risk factor absent) and the training-set mean for age and sex: average age 39.7, 49% male, from the 1,025,602 training patients (`reports/explain_baseline.json`). A female and a male patient therefore both get a sex contribution with the same meaning. All age columns are reported as one "Age" contribution and sex as one "Sex" contribution.
- **Per condition:** the top 5 supporting and top 3 opposing contributions, with readable wording from `features.json` (for example "How intense is the pain? (0-10 scale): 6/10").
- **Contrast:** "why A rather than B" for the top two conditions shown, `(coef[A] - coef[B]) * (x - m)`. This is the headline explanation.
- **Interface:** `ChampionExplainer(...).explain_evidence(PatientEvidence(...))` returns the prediction (differential or abstain), an explanation per shown condition, and the contrast. It refuses a non-linear champion rather than giving wrong answers.

## Two things a reader must know

1. **Explanations are in the model's logit space, before isotonic calibration.** Calibration changes the probabilities shown, not these scores. When calibration puts the top two in the opposite order to the model's scores, the contrast is flagged (`ranking_changed_by_calibration`) with a note.
2. **A single-condition explanation is relative to the average condition.** Adding the same number to every condition's coefficient for a feature changes no probability, so per-condition coefficients only mean something relative to each other. The fitted coefficients are centered: the largest per-feature mean across conditions is 0.015, against coefficients up to 16. The contrast has no such ambiguity, which is why it is the most useful explanation for a triage user.

## A fix found on the way: calibrated ties

The first real run reported that calibration changes the top-3 set for 93% of patients. The cause: isotonic calibration maps many small probabilities to exactly the same value. For 96.5% of held-out patients, the calibrated probability in the second slot was below 1% (median exactly 0), and for 99.9% the third slot was tied with other conditions. So slots 2 and 3, and the runner-up in the contrast, were being chosen by a condition's position in `labels.json`.

Fix: exact ties after calibration are now broken by the model's own (uncalibrated) score, in `predict_differential`, in the explainer, and in the ranking measure. This changes the top condition only when it is itself exactly tied, and makes slots 2 and 3 the model's genuine next choices. In the worked example, the panic attack contrast now names GERD (the model's real runner-up) instead of an alphabetical pick.

### Earlier metrics and the served order

Phase 3 and 3b metrics (top-3 accuracy, the selection floors, the coverage curve behind the 0.60 threshold, and the one-time test result) were computed with exact ties ordered by label position, not by the model's score as predictions are now served. All metric functions now take the same tie-break, and future runs use it. Re-measured on the same held-out validation rows with the served order (the test split was not re-scored):

| Evidence | Top-3, label order | Top-3, served order | Suggested threshold, label / served | Top-3 on answered at 0.60, label / served |
|---|---:|---:|---:|---:|
| Full record | 1.000 | 1.000 | 0.00 / 0.00 | 1.000 / 1.000 |
| Initial + 3 | 0.983 | 0.983 | 0.00 / 0.00 | 0.999 / 0.999 |
| Initial only | 0.585 | 0.590 | 0.60 / 0.58 | 0.953 / 0.965 |

The served order is equal or slightly better everywhere, so the recorded numbers stand as a lower bound. The threshold stays at 0.60: it still meets the 95% target and is slightly more cautious than 0.58.

## Results

| Measure | Full evidence | Partial evidence (initial + 3) |
|---|---:|---:|
| Top-1 changed by calibration | 0.05% | 1.5% |
| Shown set changed (conditions at 1% or more) | 0.05% | 4.0% |

- **Latency** end to end (encode, calibrated prediction, explanation) on 300 real validation patients: median 0.6 ms, p95 0.7 ms, against the 50 ms budget.
- **Global view:** 8 conditions, top 10 features each. Automatic flags fired on 2 of 80 features (the croup age buckets, which are expected). The author review finds six conditions led by textbook features. The main oddity is DDXPlus skin and rash questions ranking high for pneumonia (99.3% of those patients) and influenza (86.5%). That looks like a synthetic-data quirk the model reproduces faithfully, but it would confuse users if shown as a reason. Other single oddities: HIV prevalence for NSTEMI/STEMI, regular exercise pushing toward unstable angina, excess saliva for croup, and chronic-sinusitis risk factors for acute rhinosinusitis. That last one explains the Phase 2 acute/chronic confusion.

## Follow-ups

- **Before showing reasons to users (Phase 5 or 10):** decide how to handle features that are faithful to DDXPlus but clinically misleading, such as the skin questions for respiratory conditions. Options: an allow-list of features that may appear as reasons, or a note.
- **Reviewer notes:** `docs/phase4_sanity_review.md` ends with an empty "Reviewer notes" section for the project owner. Re-run `make explain` afterwards so the report embeds them.
- The Phase 5 API should load `reports/explain_baseline.json` for the baseline instead of recomputing it from the training data.

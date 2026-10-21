# Phase 2 analysis: model choice and confusions

> BeaconAI is decision support, not diagnosis. All numbers below come from synthetic data (DDXPlus) and say nothing about real patients.

Numbers are from `reports/model_comparison.md` (full dataset; test split scored once on 2026-10-21).

## Model choice

| Model | Val top-3 | Val top-1 | Test top-1 | Test macro F1 | Test DDx nR@3 | Fit time (full data) | Latency per patient | Explanation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic regression | 1.000 | 0.996 | 0.996 | 0.995 | 0.703 | 14 s | 0.08 ms | Coefficients, exact and global |
| XGBoost | 1.000 | 0.997 | 0.998 | 0.997 | 0.697 | 184 s | 0.99 ms | Exact TreeSHAP per prediction |
| MLP | 1.000 | 0.996 | 0.996 | 0.996 | 0.693 | 54 s | 0.05 ms | Approximate SHAP only (sampling) |

**Registered champion: XGBoost** (`beaconai-classifier`, alias `champion`).

Why:

- The selection rule is validation top-3 accuracy. All three models reach 1.000, so the rule falls through to the configured tie-breakers. XGBoost has the best validation top-1 (0.997) and macro F1.
- XGBoost supports exact, fast TreeSHAP, which Phase 4 needs for per-feature explanations.
- Latency is not a constraint: about 1 ms per patient is far below what a chat turn takes.

Caveats that matter for later phases:

- **The task is close to saturated with full evidence.** Each DDXPlus patient comes with every symptom they have. The real system asks only a few questions, so the models will see partial evidence (Phase 8). The ranking between models can change there. A leakage check found nothing: the features are only symptoms, age, and sex. A model given only the first reported symptom reaches 0.37 top-1, and 4.9% of validation patients exactly match a training patient.
- **XGBoost is the most accurate but has the lowest differential overlap.** It puts almost all probability on its first choice, so its second and third slots are close to arbitrary. Logistic regression's runner-up choices match the reference differential slightly better (nR@3 0.703 vs 0.697). Phase 3 calibration should check whether XGBoost's lower-ranked probabilities mean anything before they are shown as a top-3.
- **Logistic regression is a strong alternative:** within 0.2 points on accuracy, 13 times faster to train, and directly interpretable. If partial-evidence results or calibration favour it, switching is cheap: all models share one wrapper and one registry name.

## Confusions (XGBoost, test)

Nearly all mass is on the diagonal. Six off-diagonal pairs have any errors; the top five:

| True | Predicted as | Count | Share of true | Severity (true / predicted) | Clinically similar? | Dangerous? |
|---|---|---:|---:|---|---|---|
| Acute rhinosinusitis | Chronic rhinosinusitis | 245 | 13.4% | 4 / 5 | Yes. Same presenting symptoms (nasal congestion, facial pain, discharge, reduced smell). The main difference is duration, which the evidence set captures only weakly. | Low. Both are low severity; the main cost is a missed short antibiotic decision. |
| Unstable angina | Stable angina | 43 | 1.5% | 2 / 2 | Yes. Both are exertional chest pain from coronary disease. Unstable angina is new, worsening, or at rest. | **Yes.** Unstable angina is an acute coronary syndrome. Calling it stable angina under-triages a patient who needs urgent care. The red-flag rules (Phase 6) must catch chest pain at rest or new chest pain regardless of the model. |
| Acute laryngitis | Viral pharyngitis | 11 | 0.3% | 4 / 4 | Yes. Both are viral upper airway infections with sore throat. | Low. |
| Viral pharyngitis | Acute laryngitis | 11 | 0.1% | 4 / 4 | Yes, the same pair in reverse. | Low. |
| Chronic rhinosinusitis | Acute rhinosinusitis | 10 | 0.4% | 5 / 4 | Yes, the same pair in reverse. | Low. |

The sixth pair is croup predicted as laryngospasm (1 patient). That error goes toward the more severe condition, so it over-triages, which is the safer direction.

Every confusion is between clinically similar conditions in the same body system. None of them is a random error. Only one has safety consequences (unstable vs stable angina), and the safety layer, not the classifier, is the place to guard it.

Lowest per-class recall: acute rhinosinusitis 0.866, unstable angina 0.985; every other class is at or above 0.996.

## Method notes

- **Tuning:** grid search (11 trials: 3 logistic regression, 4 XGBoost, 4 MLP) on the dev subsample, 50,000 patients per split, scored on validation only. Grids are in `config/base.yaml`.
- **Final fit:** the best settings per model were refit on the full training set (1,025,602 patients), with early stopping on the full validation set (132,448). No test data was used for any choice.
- **Test:** scored once on 134,529 patients. `reports/test_eval_lock.json` blocks repeat scoring.
- **Differential overlap:** D is the set of conditions with non-zero probability in the DDXPlus reference differential (median size 8). Precision@k = hits / k, raw recall@k = hits / |D|, normalized recall@k = hits / min(k, |D|). As reference points on validation, a model that outputs the reference differential itself scores normalized recall@3 = 1.0, and random scores 0.19.
- **Selection rule:** validation top-3 accuracy, with exact ties broken by top-1 accuracy, then macro F1 (`train.tie_breakers`).

# BeaconAI

AI clinical triage assistant: describe symptoms in plain language, get a ranked top-3 differential with calibrated confidence, explanations, emergency red-flag checks, and citations to public medical guidance.

> **Disclaimer:** BeaconAI is a decision-support research project, not a diagnosis and not a medical device. It is trained only on synthetic data (DDXPlus). If you think you may have a medical emergency, contact your local emergency services now.

## Status

Phase 3b (evidence dropout, partial-evidence calibration, final test evaluation) done. Next: explainability.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). On macOS, XGBoost also needs OpenMP: `brew install libomp`.

```
make setup   # install dependencies and git hooks
make lint    # ruff check + format check
make test    # pytest
```

On macOS, run Python through `make` (or set `DYLD_LIBRARY_PATH` as the error message shows). torch and XGBoost each ship an OpenMP runtime, and two runtimes in one process crash during training. The Makefile points both at torch's copy.

## Data

BeaconAI uses [DDXPlus](https://doi.org/10.6084/m9.figshare.20043374) (Mila, CC BY 4.0), a synthetic dataset of about 1.3 million patients across 49 conditions.

```
make data    # download DDXPlus and build features (dvc repro)
make eda     # save class balance, evidences per patient, and age plots to reports/
```

- Raw files go to `data/raw/ddxplus/`. Download links and checksums come from the Figshare API. If the download fails, the command prints manual download steps.
- Processed files go to `data/processed/ddxplus/`: sparse float32 feature matrices per split, labels, the ground-truth differential, and `features.json`, which gives a readable label for every column.
- `data.dev_sample` in `config/base.yaml` caps each split at that many patients for fast iteration. Set it to `null` for the full dataset.
- Encoding: binary evidences are 0/1. The six 0-10 scale evidences are one column each (value / 10, 0 when absent). Other categorical evidences are one-hot. Multi-choice evidences are multi-hot. Age is both bucketed and scaled (age / 110). Sex is one column.

## Models and evaluation

Three classifiers share one wrapper (`fit`, `predict_proba`, `save`, `load`): multinomial logistic regression (baseline), XGBoost (main), and a small PyTorch MLP. All hyperparameters and search grids are in `config/base.yaml`.

```
make train        # grid search on the validation split, every trial logged to MLflow
make eval-val     # compare the Phase 2 models on validation (reports/model_comparison_val.md)
make train-final  # retrain only the winning settings (used for the full-data run)
make mlflow       # MLflow UI on http://localhost:5000
```

- Tuning uses the validation split only. The best model by validation top-3 accuracy (exact ties broken by top-1, then macro F1) is registered in MLflow as `beaconai-classifier` with alias `raw-champion` (`raw-dev` for the dev subsample). The `champion` alias is reserved for the calibrated model from `make calibrate`, which is what downstream phases load.
- Training and evaluation check that the processed data on disk matches the expected scope (dev subsample or full) and refuse with the rebuild command if not.
- The Phase 2 commands never touch the test split. The only test evaluation is `make eval` (see below).
- Metrics: top-1/3/5 accuracy, macro F1, per-class recall, confusion matrix, and overlap with the DDXPlus reference differential (precision@k, raw recall@k, and normalized recall@k = hits / min(k, |differential|); normalized recall@3 is the headline).
- Full-data run: `BEACONAI_DEV_SAMPLE=null uv run python -m beaconai.data.preprocess`, then `BEACONAI_DEV_SAMPLE=null make train-final`, `make train-dropout` and `make calibrate` (each with `BEACONAI_DEV_SAMPLE=null`). Run `make data` afterwards to return to the dev subsample. (`make data` uses DVC, which only sees `config/base.yaml`, not the environment override.)

Results: `reports/model_comparison.md`. Analysis and model choice: `docs/phase2_analysis.md`.

## Calibration and predictions

```
make calibrate   # calibrate on validation, pick the champion by partial-evidence ECE, write reports/calibration.md
```

- Temperature scaling and isotonic calibration are fit on half of validation and scored on the other half. The champion is the calibrated model with the lowest ECE on a partial-evidence version of validation (initial evidence plus 3 random others), subject to top-3 accuracy floors. Decision record: `docs/phase3_calibration.md`.
- Every downstream component uses one interface:

```python
from beaconai.config import load_config
from beaconai.models.calibrate import load_calibrated
from beaconai.models.calibrated import PatientEvidence, build_feature_space
from beaconai.models.train import configure_mlflow

cfg = load_config()
configure_mlflow(cfg.mlflow)  # local MLflow registry in mlruns/
clf = load_calibrated("models:/beaconai-classifier@champion").with_encoder(build_feature_space(cfg))
clf.predict_differential(PatientEvidence(evidence=["E_91", "E_55_@_V_14"], age=34, sex="F"))
# -> Differential (top 3 conditions with probabilities) or Abstain (with a reason)
```

- It abstains when the top condition's calibrated probability is below `abstention.threshold` (0.60, chosen so that patients who give only their first symptom are answered at 95% top-3 accuracy).

## Evidence dropout and the final evaluation (Phase 3b)

```
make train-dropout   # retrain logistic regression and XGBoost with random evidence dropout
make calibrate       # calibrate plain and dropout models, on full and on partial-evidence rows
make eval            # score the calibrated champion on the test split, once (full data only)
```

- Dropout fixes the Phase 3 limitation that an unasked question was read as "no": with the initial evidence plus 3 others, top-3 accuracy rises from 0.761 to 0.983. Decision record: `docs/phase3b_dropout.md`.
- `make eval` scores only `beaconai-classifier@champion` and refuses a raw model, a dev-subsample model, or a second run. Each evaluation name has its own lock in `reports/test_locks/`. Test results: `reports/final_evaluation.md`. Any classifier change after that is validated on validation data only.

Old code from the previous hospital management app is in `legacy/` and is not used.

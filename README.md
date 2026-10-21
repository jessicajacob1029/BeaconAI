# BeaconAI

AI clinical triage assistant: describe symptoms in plain language, get a ranked top-3 differential with calibrated confidence, explanations, emergency red-flag checks, and citations to public medical guidance.

> **Disclaimer:** BeaconAI is a decision-support research project, not a diagnosis and not a medical device. It is trained only on synthetic data (DDXPlus). If you think you may have a medical emergency, contact your local emergency services now.

## Status

Phase 1 (data pipeline). See `ROADMAP.md` for the build plan.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```
make setup   # install dependencies and git hooks
make lint    # ruff check + format check
make test    # pytest
```

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

Old code from the previous hospital management app is in `legacy/` and is not used.

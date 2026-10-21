# BeaconAI

AI clinical triage assistant: describe symptoms in plain language, get a ranked top-3 differential with calibrated confidence, explanations, emergency red-flag checks, and citations to public medical guidance.

> **Disclaimer:** BeaconAI is a decision-support research project, not a diagnosis and not a medical device. It is trained only on synthetic data (DDXPlus). If you think you may have a medical emergency, contact your local emergency services now.

## Status

Phase 0 (repo setup). See `ROADMAP.md` for the build plan.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```
make setup   # install dependencies and git hooks
make lint    # ruff check + format check
make test    # pytest
```

Old code from the previous hospital management app is in `legacy/` and is not used.

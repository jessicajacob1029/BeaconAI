# Phase 3b: Evidence Dropout, Partial-Evidence Calibration, and the Final Test Evaluation

> Execute task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase 3 "not asked is read as no" gap. Retrain logistic regression and XGBoost with random evidence dropout. Add calibrators fit on partial-evidence rows. Rerun the Phase 3 selection and threshold pipeline and compare against the Phase 3 baseline. Then score the final calibrated champion on the test split exactly once, and make `make eval` refuse to score anything else.

**Architecture:** One shared evidence-filtering core in `eval/partial.py` drives both the fixed partial-evidence evaluation (initial plus 3 others) and training-time dropout (each other evidence dropped with probability `rate`). `models/train_dropout.py` builds an augmented training set (original rows plus dropped copies), refits each Phase 2 winner with its Phase 2 hyperparameters, and logs it to MLflow. `models/calibrate.py` gains two candidate dimensions: base variant (`baseline`, `dropout`) and calibration rows (`full`, `partial`). The selection rule, floors and threshold logic are unchanged. `eval/final.py` is the only path that touches the test split. It scores `beaconai-classifier@champion` after checking that the alias points at a full-data calibrated model and that no final evaluation has run.

**Tech Stack:** Python 3.11, numpy, scipy, scikit-learn, XGBoost, MLflow, matplotlib, pytest. No new dependencies.

**Spec:** the Phase 3b request (2026-10-21):
- retrain logistic regression and XGBoost with random evidence dropout during training (whole evidences, same rules as the partial-evidence evaluation, rate from config)
- optionally compare a yes/no/not-asked encoding if cheap
- fit calibrators on partial-evidence rows from the calibration half
- rerun the Phase 3 selection and threshold pipeline and compare against the Phase 3 baseline table (partial and full ECE and top-3)
- run the one-time test evaluation on the final champion only
- `make eval` refuses to score anything except the calibrated champion alias

Context: `docs/phase3_calibration.md`, the `ROADMAP.md` note before Phase 8.

## Global Constraints

- All tunables (dropout rate, copies, variants, calibration rows, experiment names) live in `config/base.yaml`.
- Seed everything with `cfg.seed`. The held-out half and its partial-evidence version must be byte-identical to Phase 3 (same `split_indices` and `partial_evidence` calls and seeds), so the comparison is like for like.
- The held-out half of validation is used only for scoring: never for early stopping, calibrator fitting or tuning in this phase.
- The test split is read only by `eval/final.py`, once, for the registered calibrated champion, on full data.
- Only full-data `make calibrate` moves `champion`; dropout models are not registered.
- Dev outputs carry a `_dev` suffix and are gitignored.
- Docs and reports: plain and direct, no em dashes, no escape sequences standing in for them. Every report states BeaconAI is decision support, not diagnosis, trained on synthetic data.
- No tool attribution or co-author trailers in commits, code comments, PR text, or any tracked file.
- On macOS run Python through `make`.
- Phase ends with tests passing, lint clean, a CHANGELOG entry with key metrics, and a commit.

## Amendments after review (these override the tasks below)

1. **Dropout rate drawn per augmented row.** Config `dropout.rate_range: [0.5, 0.95]` replaces `dropout.rate`. Every augmented row (one dropped copy of one patient) draws its own rate uniformly from the range, then drops each non-initial evidence with that probability. Training therefore covers both few-symptom and many-symptom situations. `evidence_dropout(X, initial, col_evidence, rate_range, seed)`. A fixed rate is `(r, r)`.
2. **Evidence-count curve.** Config `calibration.evidence_curve: [0, 1, 2, 3, 5, 8, 12]`. For each value n, the held-out half keeps the initial evidence plus n other random evidences (same rule and seed as the partial-evidence evaluation). Top-3 accuracy and ECE are computed for the 3b champion and, when the Phase 3 baseline has the same data scope, for the Phase 3 champion. The report gets a table and the figure `reports/figures/evidence_curve{sfx}.png`, and the summary gets `evidence_curve`.
3. **Test lock per evaluation name.** Config `final_eval.name: classifier_final`. The lock is `reports/test_locks/<name>.json`, so a later evaluation (for example a Phase 8 agent simulation) can have its own one-time test run. The final report states: any classifier change after this point is validated on validation data only.

## Method decisions (confirm at review)

1. **Dropout rule.** For every training patient, the initial evidence, age and sex are always kept. Every other evidence (a whole question with all its values) is dropped independently with probability `dropout.rate` (default 0.75). The training set is the original rows plus `dropout.copies` (default 1) dropped copies, each with a different seed. A patient has about 19 evidence tokens on average, so a dropped copy keeps the initial evidence plus about 4 others. That is close to the evaluation setting (initial plus 3) but varies from patient to patient. Keeping the original rows preserves full-evidence accuracy.
2. **No hyperparameter re-tuning.** Dropout models reuse the Phase 2 winning hyperparameters. Re-tuning is a possible follow-up if dropout models underfit.
3. **Early stopping without touching the held-out half.** XGBoost early-stops on the partial-evidence version of the calibration fit half, never on the held-out half. That makes the 3b numbers more independent than the Phase 3 baseline, whose models early-stopped on all of validation.
4. **A 2 x 2 comparison.** Candidates are base variant (`baseline` = Phase 2 models, `dropout`) x calibration rows (`full` = Phase 3 behavior, `partial` = partial-evidence version of the fit half) x method (`none`, `temperature`, `isotonic`). `none` is listed once per base model. This separates what dropout buys from what partial-evidence calibration buys. Per model that is 5 candidates; with 2 models x 2 variants it is 20.
5. **The yes/no/not-asked encoding is not done here.** It is not cheap. It needs a second column per evidence (about 223 more columns), a change to the Phase 1 encoder and to `PatientEvidence` (callers must say which questions were answered "no"), and a simulation of which absent evidences were asked during training. Recorded as a follow-up for Phase 8, where the agent knows exactly which questions it asked.
6. **Baseline snapshot.** Before any rerun, the committed Phase 3 summary is copied to `reports/phase3_baseline_summary.json`. The new report shows the Phase 3 baseline table next to the 3b results, plus a head-to-head of the two champions. The comparison is only drawn when both are full-data runs.
7. **Final evaluation.** `make eval` runs `eval/final.py`. It refuses unless: the processed data is the full dataset; the `champion` version is tagged as calibrated and full-data; and `reports/final_test_lock.json` does not exist. It scores the test split with full evidence and with the same partial-evidence rule: top-1/3/5, macro F1, differential overlap, ECE, NLL, and abstention coverage and accuracy at the configured threshold. It writes `reports/final_evaluation.md` and the lock. The Phase 2 `evaluate.py` no longer scores test at all; `make eval-val` keeps its validation comparison.

## Review Focus

1. **Dropout that removes the initial evidence, splits a multi-choice evidence, or drops age or sex:** none of these may happen. Test in Task 2.
2. **A held-out row leaking into training, early stopping or calibration:** the dropout runner and the calibrator rows must use only the fit half. Test in Tasks 3 and 4 (index-level checks).
3. **The Phase 3 baseline changing silently:** refactoring `partial_evidence` must keep its output identical for the same seed. Test in Task 2 (golden output captured before the refactor).
4. **`make eval` scoring a raw model, a dev model, or running twice:** refused with a clear message. Test in Task 5.
5. **The comparison mixing dev and full numbers:** the baseline section is drawn only when both runs share a data scope. Test in Task 4.

---

## File map

| File | Responsibility |
|---|---|
| `config/base.yaml`, `src/beaconai/config.py` | `dropout` section; `calibration.variants`, `calibration.fit_on`; `final_eval` section |
| `src/beaconai/eval/partial.py` | Shared filtering core; `partial_evidence` (unchanged output), new `evidence_dropout` |
| `src/beaconai/models/dataset.py` | `split_indices` moves here (re-exported from `calibrate`); `augment_with_dropout` |
| `src/beaconai/models/train_dropout.py` | Dropout retraining runner, `reports/train_dropout_summary.json` |
| `src/beaconai/models/calibrate.py` | Base variants, calibration-row choice, baseline comparison |
| `src/beaconai/eval/calibration_report.py` | "Calibrated on" column, baseline comparison section |
| `src/beaconai/eval/final.py`, `src/beaconai/eval/final_report.py` | One-time test evaluation of the calibrated champion |
| `src/beaconai/eval/evaluate.py` | Test split removed; validation comparison only |
| `Makefile` | `train-dropout`; `eval` points at the final evaluation |
| `reports/phase3_baseline_summary.json` | Frozen copy of the Phase 3 summary |
| `docs/phase3b_dropout.md`, `reports/CHANGELOG.md`, `README.md` | Decision record and docs |

---

### Task 1: Config

**Files:** `config/base.yaml`, `src/beaconai/config.py`; Test: `tests/test_config_dropout.py`

**Interfaces:** `DropoutConfig(models: list[ModelName], rate: float in (0, 1), copies: int >= 0, keep_original: bool, experiment: str)` (rejects `copies == 0 and not keep_original`); `CalibrationConfig.variants: list[Literal["baseline", "dropout"]] = ["baseline"]`; `CalibrationConfig.fit_on: list[Literal["full", "partial"]] = ["full"]`; `FinalEvalConfig(experiment: str = "beaconai-final", lock_file: str = "final_test_lock.json")`; optional `Config.dropout`, `Config.final_eval`; `require_dropout(cfg) -> DropoutConfig`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_config_dropout.py
import pytest
import yaml
from pydantic import ValidationError

from beaconai.config import (
    DEFAULT_CONFIG_PATH,
    CalibrationConfig,
    Config,
    DropoutConfig,
    load_config,
    require_dropout,
)


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_has_dropout_and_new_calibration_keys() -> None:
    cfg = load_config()
    drop = require_dropout(cfg)
    assert drop.models == ["logreg", "xgboost"]
    assert 0 < drop.rate < 1 and drop.copies >= 1 and drop.keep_original
    assert cfg.calibration.variants == ["baseline", "dropout"]
    assert cfg.calibration.fit_on == ["full", "partial"]
    assert cfg.final_eval.lock_file == "final_test_lock.json"


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1])
def test_rate_strictly_between_zero_and_one(rate: float) -> None:
    raw = _raw()["dropout"]
    raw["rate"] = rate
    with pytest.raises(ValidationError):
        DropoutConfig.model_validate(raw)


def test_no_training_rows_rejected() -> None:
    raw = _raw()["dropout"]
    raw.update(copies=0, keep_original=False)
    with pytest.raises(ValidationError, match="no training rows"):
        DropoutConfig.model_validate(raw)


def test_new_calibration_keys_default_to_phase3_behaviour() -> None:
    raw = _raw()["calibration"]
    del raw["variants"], raw["fit_on"]
    cal = CalibrationConfig.model_validate(raw)
    assert cal.variants == ["baseline"] and cal.fit_on == ["full"]


def test_require_dropout_missing() -> None:
    raw = _raw()
    del raw["dropout"]
    with pytest.raises(ValueError, match="dropout"):
        require_dropout(Config.model_validate(raw))
```

Run `make test` and expect `ImportError: cannot import name 'DropoutConfig'`.

- [ ] **Step 2: YAML.** Add `variants: [baseline, dropout]` and `fit_on: [full, partial]` to `calibration` (each with a comment), and append:

```yaml
dropout:
  # Retrain these Phase 2 winners (same hyperparameters) on rows with evidences dropped.
  models: [logreg, xgboost]
  # Each evidence other than the initial one is dropped with this probability. A patient has
  # about 19 evidence tokens, so a dropped copy keeps the initial evidence plus about 4 others.
  rate: 0.75
  # Dropped copies per training patient, added to the original rows.
  copies: 1
  # Keep the complete rows too, so full-evidence accuracy is preserved.
  keep_original: true
  experiment: beaconai-phase3b

final_eval:
  # The one-time test evaluation of the calibrated champion (make eval).
  experiment: beaconai-final
  lock_file: final_test_lock.json
```

- [ ] **Step 3: Schemas** in `src/beaconai/config.py`:

```python
CalibrationVariant = Literal["baseline", "dropout"]
CalibrationRows = Literal["full", "partial"]


class DropoutConfig(_StrictModel):
    models: list[ModelName] = Field(min_length=1)
    rate: float = Field(gt=0, lt=1)
    copies: int = Field(default=1, ge=0)
    keep_original: bool = True
    experiment: str = "beaconai-phase3b"

    @model_validator(mode="after")
    def _has_rows(self) -> DropoutConfig:
        if self.copies == 0 and not self.keep_original:
            raise ValueError("copies=0 with keep_original=false leaves no training rows")
        return self


class FinalEvalConfig(_StrictModel):
    experiment: str = "beaconai-final"
    lock_file: str = "final_test_lock.json"
```

Add to `CalibrationConfig`: `variants: list[CalibrationVariant] = Field(default_factory=lambda: ["baseline"], min_length=1)` and `fit_on: list[CalibrationRows] = Field(default_factory=lambda: ["full"], min_length=1)`. Add to `Config`: `dropout: DropoutConfig | None = None`, `final_eval: FinalEvalConfig = FinalEvalConfig()`. Add:

```python
def require_dropout(cfg: Config) -> DropoutConfig:
    if cfg.dropout is None:
        raise ValueError("config is missing the dropout section")
    return cfg.dropout
```

In `tests/conftest.py` `make_config`, keep toy runs fast: `raw["dropout"].update(copies=1)` and `raw["calibration"]["variants"] = ["baseline"]` (dropout-specific tests opt in).

- [ ] **Step 4:** `make test` passes. Commit: `Add dropout, calibration variant, and final evaluation config`.

---

### Task 2: Shared evidence filtering, `evidence_dropout`, `augment_with_dropout`, `split_indices` move

**Files:** `src/beaconai/eval/partial.py`, `src/beaconai/models/dataset.py`, `src/beaconai/models/calibrate.py` (re-export); Test: `tests/eval/test_partial.py`, `tests/models/test_dataset.py`

**Interfaces:** `evidence_dropout(X, initial_evidence, col_evidence, rate, seed) -> csr_matrix`; `augment_with_dropout(data: SplitData, initial_evidence, col_evidence, rate, copies, keep_original, seed) -> SplitData` (ddx None; copy `c` uses seed `seed + 1 + c`); `split_indices` defined in `dataset.py`, still importable from `calibrate`.

- [ ] **Step 1: Capture the golden output before refactoring**

Append to `tests/eval/test_partial.py` and run it once on the current code (it must pass before the refactor):

```python
def test_partial_evidence_output_is_unchanged_by_refactor() -> None:
    # Golden values from the Phase 3 implementation; the refactor must not change them.
    rng = np.random.default_rng(123)
    big = sparse.csr_matrix((rng.random((40, 9)) < 0.6).astype(np.float32))
    initial = np.array([COLS[i % 7] for i in range(40)])
    out = partial_evidence(big, initial, COLS, extra=2, seed=7)
    assert (out.indptr.tolist(), out.indices.tolist()) == GOLDEN
```

Generate `GOLDEN` by running this snippet once against the current `partial_evidence` and pasting the tuple into the test as a constant:

```bash
make test  # first, confirm everything is green on the pre-refactor code
DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/torch/lib uv run python -c "
import numpy as np; from scipy import sparse
from tests.eval.test_partial import COLS
from beaconai.eval.partial import partial_evidence
rng = np.random.default_rng(123)
big = sparse.csr_matrix((rng.random((40, 9)) < 0.6).astype(np.float32))
initial = np.array([COLS[i % 7] for i in range(40)])
o = partial_evidence(big, initial, COLS, 2, 7); print((o.indptr.tolist(), o.indices.tolist()))"
```

- [ ] **Step 2: Failing tests for dropout and augmentation**

```python
# tests/eval/test_partial.py (append)
from beaconai.eval.partial import evidence_dropout


def test_dropout_keeps_initial_demographics_and_whole_evidences() -> None:
    for seed in range(30):
        out = evidence_dropout(X, INITIAL, COLS, rate=0.75, seed=seed)
        row = out[0].toarray()[0]
        assert row[0] == 1  # initial evidence E_1
        assert row[7] == pytest.approx(0.4) and row[8] == 1  # age and sex
        assert (row[3] == 0) == (row[4] == 0)  # E_4 kept or dropped as a whole


def test_dropout_rate_matches_on_average() -> None:
    rng = np.random.default_rng(0)
    many = sparse.csr_matrix(np.tile(X[0].toarray(), (2000, 1)))
    out = evidence_dropout(many, np.array(["E_1"] * 2000), COLS, rate=0.75, seed=rng.integers(1e6))
    kept_others = np.mean([len(_kept_codes(out[i])) - 1 for i in range(2000)])
    assert kept_others == pytest.approx(5 * 0.25, abs=0.1)  # 5 other evidences, keep 25%


def test_dropout_is_seeded() -> None:
    a = evidence_dropout(X, INITIAL, COLS, 0.5, seed=3).toarray()
    np.testing.assert_array_equal(a, evidence_dropout(X, INITIAL, COLS, 0.5, seed=3).toarray())


def test_dropout_rate_bounds() -> None:
    with pytest.raises(ValueError):
        evidence_dropout(X, INITIAL, COLS, rate=1.0, seed=0)
```

```python
# tests/models/test_dataset.py (append)
def test_augment_with_dropout_stacks_original_and_copies(toy_processed: Path) -> None:
    from beaconai.eval.partial import column_evidence
    from beaconai.models.dataset import augment_with_dropout, load_features, load_meta

    data = load_split(toy_processed, "train")
    meta = load_meta(toy_processed, "train")
    cols = column_evidence(load_features(toy_processed))
    aug = augment_with_dropout(data, meta["initial_evidence"], cols, 0.75, 2, True, seed=0)
    n = data.n
    assert aug.n == 3 * n
    np.testing.assert_array_equal(aug.y, np.tile(data.y, 3))
    np.testing.assert_array_equal(aug.X[:n].toarray(), data.X.toarray())  # originals first
    assert aug.X[n:].nnz < 2 * data.X.nnz  # copies lost evidence
    assert not np.array_equal(aug.X[n : 2 * n].toarray(), aug.X[2 * n :].toarray())
    only_copies = augment_with_dropout(data, meta["initial_evidence"], cols, 0.75, 1, False, 0)
    assert only_copies.n == n


def test_split_indices_still_importable_from_calibrate() -> None:
    from beaconai.models.calibrate import split_indices as a
    from beaconai.models.dataset import split_indices as b

    assert a is b
```

Run `make test` and expect `ImportError` for `evidence_dropout` and `augment_with_dropout`.

- [ ] **Step 3: Implement.** Replace the body of `src/beaconai/eval/partial.py` below `column_evidence` with:

```python
Chooser = Callable[[list[str], np.random.Generator], set[str]]


def _keep_evidence(
    X: sparse.csr_matrix,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    choose: Chooser,
    seed: int,
) -> sparse.csr_matrix:
    """Keep the initial evidence, age and sex, and whichever other evidences `choose` picks."""
    X = sparse.csr_matrix(X)
    if len(initial_evidence) != X.shape[0]:
        raise ValueError(f"{len(initial_evidence)} initial evidences for {X.shape[0]} rows")
    if len(col_evidence) != X.shape[1]:
        raise ValueError(f"{len(col_evidence)} column codes for {X.shape[1]} columns")
    rng = np.random.default_rng(seed)
    indptr, indices, data = [0], [], []
    for i in range(X.shape[0]):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        cols, vals = X.indices[lo:hi], X.data[lo:hi]
        codes = col_evidence[cols]
        others = sorted({c for c in codes if c and c != initial_evidence[i]})
        keep = choose(others, rng) | {str(initial_evidence[i])}
        mask = np.array([c == "" or c in keep for c in codes], dtype=bool)
        indices.extend(cols[mask])
        data.extend(vals[mask])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=X.shape,
    )


def partial_evidence(
    X: sparse.csr_matrix,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    extra: int,
    seed: int,
) -> sparse.csr_matrix:
    """Initial evidence plus `extra` other random evidences per patient (evaluation setting)."""

    def choose(others: list[str], rng: np.random.Generator) -> set[str]:
        n_pick = min(extra, len(others))
        # Only draw when something is picked, so the random stream matches Phase 3 exactly.
        return set(rng.choice(others, size=n_pick, replace=False)) if n_pick else set()

    return _keep_evidence(X, initial_evidence, col_evidence, choose, seed)


def evidence_dropout(
    X: sparse.csr_matrix,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    rate: float,
    seed: int,
) -> sparse.csr_matrix:
    """Drop each evidence other than the initial one with probability `rate` (training setting)."""
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0, 1), got {rate}")

    def choose(others: list[str], rng: np.random.Generator) -> set[str]:
        draws = rng.random(len(others))
        return {code for code, u in zip(others, draws, strict=True) if u >= rate}

    return _keep_evidence(X, initial_evidence, col_evidence, choose, seed)
```

(Add `from collections.abc import Callable`.) Move `split_indices` verbatim from `calibrate.py` into `dataset.py` and import it back in `calibrate.py` (`from beaconai.models.dataset import split_indices`). Add to `dataset.py`:

```python
def augment_with_dropout(
    data: SplitData,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    rate: float,
    copies: int,
    keep_original: bool,
    seed: int,
) -> SplitData:
    """Original rows (optional) followed by `copies` evidence-dropped copies of every row."""
    from beaconai.eval.partial import evidence_dropout

    parts = [data.X] if keep_original else []
    for c in range(copies):
        parts.append(evidence_dropout(data.X, initial_evidence, col_evidence, rate, seed + 1 + c))
    if not parts:
        raise ValueError("no training rows: copies=0 and keep_original=False")
    n_blocks = len(parts)
    return SplitData(sparse.vstack(parts).tocsr(), np.tile(data.y, n_blocks), None)
```

- [ ] **Step 4:** `make test` passes, including the golden test. Commit: `Share evidence filtering; add training-time evidence dropout`.

---

### Task 3: Dropout retraining runner (`make train-dropout`)

**Files:** create `src/beaconai/models/train_dropout.py`; modify `Makefile`; Test: `tests/models/test_train_dropout.py`

**Interfaces:** `DROPOUT_SUMMARY_FILE = "train_dropout_summary.json"`; `run_dropout_training(cfg) -> dict` writing `{parent_run_id, dev_sample, rate, copies, keep_original, n_train_rows, n_stop_rows, models: {name: {run_id, model_uri, params, fit_seconds, best_iteration | epochs_trained}}}`; `early_stopping_rows(val, meta, features, cfg) -> tuple[SplitData, np.ndarray]` (the partial-evidence fit half and its validation indices).

- [ ] **Step 1: Failing tests**

```python
# tests/models/test_train_dropout.py
import json
from pathlib import Path

import mlflow
import numpy as np
import pytest

from beaconai.models.dataset import load_meta, load_split, split_indices
from beaconai.models.train import DataScopeError, configure_mlflow, load_wrapper, run_training
from beaconai.models.train_dropout import early_stopping_rows, run_dropout_training
from tests.conftest import make_config, write_toy_processed


def _setup(tmp_path: Path, dev_sample: int | None = None):
    cfg = make_config(tmp_path, dev_sample=dev_sample)
    processed = write_toy_processed(tmp_path / "processed", dev_sample=dev_sample)
    run_training(cfg)
    return cfg, processed


def test_early_stopping_rows_come_only_from_the_fit_half(tmp_path: Path) -> None:
    cfg, processed = _setup(tmp_path)
    from beaconai.models.dataset import load_features

    val = load_split(processed, "validate")
    stop, idx = early_stopping_rows(
        val, load_meta(processed, "validate"), load_features(processed), cfg
    )
    fit_idx, held_idx = split_indices(val.y, cfg.calibration.fit_fraction, cfg.seed)
    np.testing.assert_array_equal(idx, fit_idx)
    assert set(idx).isdisjoint(held_idx)
    assert stop.n == len(fit_idx) and stop.X.nnz <= val.X[fit_idx].nnz


def test_run_dropout_training(tmp_path: Path) -> None:
    cfg, processed = _setup(tmp_path)
    summary = run_dropout_training(cfg)
    assert set(summary["models"]) == {"logreg", "xgboost"}
    assert summary["n_train_rows"] == 2 * 150  # original + 1 copy
    reports = tmp_path / "reports"
    assert json.loads((reports / "train_dropout_summary.json").read_text()) == summary
    model = load_wrapper(summary["models"]["xgboost"]["model_uri"])
    assert model.predict_proba(load_split(processed, "test").X).shape == (60, 3)
    train_params = json.loads((reports / "train_summary.json").read_text())["models"]
    assert summary["models"]["logreg"]["params"] == train_params["logreg"]["params"]
    configure_mlflow(cfg.mlflow, experiment=cfg.dropout.experiment)
    runs = mlflow.search_runs(experiment_names=[cfg.dropout.experiment])
    assert len(runs) == 3  # parent + 2 models
    assert runs["params.dropout_rate"].dropna().astype(float).eq(cfg.dropout.rate).all()


def test_refuses_on_data_scope_mismatch(tmp_path: Path) -> None:
    cfg, processed = _setup(tmp_path)
    manifest = json.loads((processed / "manifest.json").read_text())
    manifest["settings"]["dev_sample"] = 50000
    (processed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataScopeError):
        run_dropout_training(cfg)
```

Run `make test`; expect `ModuleNotFoundError: beaconai.models.train_dropout`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/models/train_dropout.py
"""Phase 3b: retrain the Phase 2 winners with random evidence dropout.

Run: uv run python -m beaconai.models.train_dropout

The training set is the original rows plus copies in which each evidence other than the initial
one is dropped with probability `dropout.rate`, so the models learn that a missing evidence is
not necessarily a "no". Hyperparameters are the Phase 2 winners. Early stopping uses the
partial-evidence version of the calibration fit half of validation; the held-out half is never
touched here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from beaconai.config import Config, load_config, project_path, require_calibration
from beaconai.config import require_dropout, require_training
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.models.dataset import (
    SplitData,
    augment_with_dropout,
    load_features,
    load_labels,
    load_meta,
    load_split,
    split_indices,
)
from beaconai.models.pyfunc import BeaconPyfunc, log_pyfunc
from beaconai.models.registry import build_model
from beaconai.models.seed import set_global_seed
from beaconai.models.train import SUMMARY_FILE, check_data_scope, configure_mlflow, data_scope
from beaconai.models.tuning import flatten_params

DROPOUT_SUMMARY_FILE = "train_dropout_summary.json"


def early_stopping_rows(
    val: SplitData, meta: dict[str, np.ndarray], features: list[dict[str, Any]], cfg: Config
) -> tuple[SplitData, np.ndarray]:
    """Partial-evidence version of the calibration fit half (same split as calibration)."""
    cal_cfg, _ = require_calibration(cfg)
    fit_idx, _ = split_indices(val.y, cal_cfg.fit_fraction, cfg.seed)
    X = partial_evidence(
        val.X[fit_idx],
        meta["initial_evidence"][fit_idx],
        column_evidence(features),
        cal_cfg.partial_evidence.extra_evidences,
        cfg.seed,
    )
    return SplitData(X, val.y[fit_idx], None), fit_idx


def run_dropout_training(cfg: Config) -> dict[str, Any]:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    drop = require_dropout(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    train_summary_path = reports / SUMMARY_FILE
    if not train_summary_path.is_file():
        raise FileNotFoundError(f"{train_summary_path} not found. Run `make train` first.")
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    dev_sample = data_scope(processed)
    check_data_scope(train_summary["dev_sample"], dev_sample)

    labels, features = load_labels(processed), load_features(processed)
    feature_names = [f["name"] for f in features]
    train = load_split(processed, "train")
    train_meta = load_meta(processed, "train")
    aug = augment_with_dropout(
        train,
        train_meta["initial_evidence"],
        column_evidence(features),
        drop.rate,
        drop.copies,
        drop.keep_original,
        cfg.seed,
    )
    val = load_split(processed, "validate")
    stop, _ = early_stopping_rows(val, load_meta(processed, "validate"), features, cfg)
    configure_mlflow(mlflow_cfg, experiment=drop.experiment)

    models: dict[str, Any] = {}
    with mlflow.start_run(run_name="phase3b-train-dropout") as parent:
        mlflow.set_tags({"dev_sample": str(dev_sample), "phase": "3b"})
        for name in drop.models:
            params = train_summary["models"][name]["params"]
            set_global_seed(cfg.seed)
            with mlflow.start_run(run_name=f"{name}+dropout", nested=True) as run:
                mlflow.set_tags({"model_type": name, "variant": "dropout"})
                mlflow.log_params(
                    {
                        **flatten_params(params),
                        "dropout_rate": drop.rate,
                        "dropout_copies": drop.copies,
                        "keep_original": drop.keep_original,
                        "n_train_rows": aug.n,
                    }
                )
                model = build_model(name, params, len(labels), cfg.seed)
                start = time.perf_counter()
                model.fit(aug.X, aug.y, stop.X, stop.y)
                fit_seconds = time.perf_counter() - start
                mlflow.log_metric("fit_seconds", fit_seconds)
                extra = {
                    attr: getattr(model, attr)
                    for attr in ("best_iteration", "epochs_trained")
                    if hasattr(model, attr)
                }
                for key, value in extra.items():
                    mlflow.log_metric(key, value)
                with tempfile.TemporaryDirectory() as tmp:
                    model.save(Path(tmp) / "wrapper")
                    rows = stop.X[:5]
                    example = pd.DataFrame(rows.toarray().astype(np.float32), columns=feature_names)
                    uri = log_pyfunc(
                        BeaconPyfunc(), Path(tmp) / "wrapper", example, model.predict_proba(rows)
                    )
            models[name] = {
                "run_id": run.info.run_id,
                "model_uri": uri,
                "params": params,
                "fit_seconds": fit_seconds,
                **extra,
            }

    summary = {
        "parent_run_id": parent.info.run_id,
        "dev_sample": dev_sample,
        "rate": drop.rate,
        "copies": drop.copies,
        "keep_original": drop.keep_original,
        "n_train_rows": aug.n,
        "n_stop_rows": stop.n,
        "models": models,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / DROPOUT_SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = run_dropout_training(load_config())
    print(json.dumps({k: summary[k] for k in ("dev_sample", "rate", "n_train_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Makefile: add `train-dropout` (`uv run python -m beaconai.models.train_dropout`) and `.PHONY`.

- [ ] **Step 3:** `make test`, `make lint`. Commit: `Add evidence-dropout retraining runner`.

---

### Task 4: Calibration with variants, partial-evidence rows, and the baseline comparison

**Files:** `src/beaconai/models/calibrate.py`, `src/beaconai/eval/calibration_report.py`; create `reports/phase3_baseline_summary.json`; Test: `tests/models/test_calibrate.py`, `tests/eval/test_calibration_report.py`

**Interfaces:** `CalibrationResult` gains `fit_on: str = "full"` (after `temperature`); candidate `model` labels are `"logreg"`, `"xgboost"`, `"logreg+dropout"`, `"xgboost+dropout"`; summary candidates gain `"fit_on"`; `render_calibration(..., baseline: dict | None = None)`; `BASELINE_FILE = "phase3_baseline_summary.json"`.

- [ ] **Step 1: Freeze the Phase 3 baseline** (before any rerun):

```bash
cp reports/calibration_summary.json reports/phase3_baseline_summary.json
git add reports/phase3_baseline_summary.json
git commit -m "Freeze the Phase 3 calibration summary as the 3b baseline"
```

- [ ] **Step 2: Failing tests**

Append to `tests/models/test_calibrate.py`:

```python
def test_variants_and_partial_rows_full_grid(tmp_path: Path) -> None:
    from beaconai.models.train_dropout import run_dropout_training

    cfg = _trained(tmp_path, None)
    run_dropout_training(cfg)
    cal = cfg.calibration.model_copy(update={"variants": ["baseline", "dropout"]})
    cfg = cfg.model_copy(update={"calibration": cal})
    summary = run_calibration(cfg)
    keys = {(c["model"], c["method"], c["fit_on"]) for c in summary["candidates"]}
    # 4 base models x (none once + temperature/isotonic on full and partial rows) = 20
    assert len(keys) == len(summary["candidates"]) == 20
    assert ("logreg+dropout", "temperature", "partial") in keys
    assert ("xgboost", "none", "full") in keys and ("xgboost", "none", "partial") not in keys


def test_partial_calibration_rows_come_only_from_the_fit_half(tmp_path: Path, monkeypatch) -> None:
    import beaconai.models.calibrate as calibrate

    seen: list[int] = []
    real = calibrate.make_calibrator

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def fit(self, proba, y):
            seen.append(len(y))
            self.inner.fit(proba, y)
            return self.inner

    monkeypatch.setattr(calibrate, "make_calibrator", lambda m, b: Spy(real(m, b)))
    cfg = _trained(tmp_path, None)
    summary = run_calibration(cfg)
    assert set(seen) == {summary["n_fit"]}  # every calibrator saw exactly the fit half


def test_missing_dropout_summary_when_variant_requested(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, None)
    cal = cfg.calibration.model_copy(update={"variants": ["baseline", "dropout"]})
    with pytest.raises(FileNotFoundError, match="make train-dropout"):
        run_calibration(cfg.model_copy(update={"calibration": cal}))
```

Append to `tests/eval/test_calibration_report.py`:

```python
def test_fit_on_column_and_baseline_comparison() -> None:
    new = {
        **SUMMARY,
        "candidates": [
            {**c, "fit_on": "partial"} if c["method"] == "temperature" else c
            for c in SUMMARY["candidates"]
        ],
    }
    md = render_calibration(
        new,
        ece_bins=15,
        reliability_images={},
        coverage_image="figures/c.png",
        baseline=SUMMARY,
    )
    assert "| Calibrated on |" in md and "| partial |" in md
    assert "## Comparison with the Phase 3 baseline" in md
    assert "Phase 3 champion" in md and "Phase 3b champion" in md


def test_no_comparison_across_data_scopes() -> None:
    dev = {**SUMMARY, "dev_sample": 50000}
    md = render_calibration(
        dev, ece_bins=15, reliability_images={}, coverage_image="c.png", baseline=SUMMARY
    )
    assert "## Comparison with the Phase 3 baseline" not in md
```

Run `make test`; expect failures.

- [ ] **Step 3: Implement in `calibrate.py`**

1. `CalibrationResult`: add `fit_on: str = "full"`; `_log_candidate` takes `fit_on`, tags it, and names the run `f"{name}-{method}-{fit_on}"`.
2. Base models: build a list of `(label, model_uri)`. Start with `(name, train_summary["models"][name]["model_uri"])` for each `name` in `cal_cfg.models` when `"baseline" in cal_cfg.variants`. If `"dropout" in cal_cfg.variants`, read `reports/train_dropout_summary.json` (raise `FileNotFoundError("... Run `make train-dropout` first.")` if missing), call `check_data_scope(dropout_summary["dev_sample"], dev_sample)`, and add `(f"{name}+dropout", uri)` for each of its models.
3. Calibration rows: compute `fit_partial = SplitData(partial_evidence(fit.X, meta["initial_evidence"][fit_idx], cols, extra, cfg.seed), fit.y, None)` once. For each base model, predict on `fit`, `fit_partial`, `held` and `partial` once. Loop `for fit_on in cal_cfg.fit_on: for method in cal_cfg.methods:`, skipping `method == "none"` when `fit_on != cal_cfg.fit_on[0]` (raw is listed once). Fit on `raw_fit` or `raw_fit_partial`, and use the object `fit()` returns (`calibrator = make_calibrator(method, bounds).fit(rows, fit.y)`), which the spy test relies on.
4. Reliability panels: key curves by `f"{method} ({fit_on})"`.
5. Summary candidates gain `"fit_on"`; the champion entry gains `"fit_on"`; register tags gain `"calibrated_on"`.
6. Baseline: if `reports/phase3_baseline_summary.json` exists, load it and pass `baseline=` to `render_calibration`.

- [ ] **Step 4: Implement in `calibration_report.py`**

- Tables gain a final `Calibrated on` column (`c.get("fit_on", "full")`), so existing row-prefix assertions still hold.
- New section after the limitation table, only if `baseline is not None and baseline["dev_sample"] == summary["dev_sample"]`:

```python
def _comparison(summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    def row(label: str, c: dict[str, Any]) -> str:
        m = c["metrics"]
        return (
            f"| {label} | {c['model']} | {c['method']} | {c.get('fit_on', 'full')} | "
            f"{_f(m['partial_ece'], 4)} | {_f(m['partial_top3_accuracy'])} | "
            f"{_f(m['ece'], 4)} | {_f(m['top3_accuracy'])} |"
        )

    old = baseline["champion"]
    new = summary["champion"]
    old_c = next(
        c
        for c in baseline["candidates"]
        if (c["model"], c["method"]) == (old["model"], old["method"])
    )
    new_c = next(
        c
        for c in summary["candidates"]
        if (c["model"], c["method"], c.get("fit_on", "full"))
        == (new["model"], new["method"], new.get("fit_on", "full"))
    )
    lines = [
        "## Comparison with the Phase 3 baseline",
        "",
        "Same held-out rows and the same partial-evidence rule as Phase 3.",
        "",
        "| | Model | Method | Calibrated on | ECE partial | Top-3 partial | ECE full | Top-3 full |",
        "|---|---|---|---|---:|---:|---:|---:|",
        row("Phase 3 champion", old_c),
        row("Phase 3b champion", new_c),
        "",
        "Phase 3 baseline, all candidates:",
        "",
        "| | Model | Method | Calibrated on | ECE partial | Top-3 partial | ECE full | Top-3 full |",
        "|---|---|---|---|---:|---:|---:|---:|",
        *[row("Phase 3", c) for c in baseline["candidates"]],
        "",
    ]
    return lines
```

- [ ] **Step 5:** `make test`, `make lint`. Commit: `Calibrate dropout variants on partial-evidence rows; compare with the Phase 3 baseline`.

---

### Task 5: The one-time final test evaluation (`make eval`)

**Files:** create `src/beaconai/eval/final.py`, `src/beaconai/eval/final_report.py`; modify `src/beaconai/eval/evaluate.py`, `Makefile`; Test: `tests/eval/test_final.py`, `tests/eval/test_final_report.py`, `tests/eval/test_evaluate.py`

**Interfaces:** `FinalEvalError(Exception)`; `check_final_allowed(dev_sample_on_disk, version_tags: dict[str, str], lock_path: Path) -> None`; `run_final_evaluation(cfg) -> Path`; `render_final(summary: dict) -> str`. Outputs: `reports/final_evaluation.md`, `reports/final_evaluation.json`, `reports/figures/final_confusion_matrix.png`, `reports/final_test_lock.json`. `evaluate.run_evaluation(cfg, "test")` raises `TestGuardError` pointing at `make eval`.

- [ ] **Step 1: Failing tests**

```python
# tests/eval/test_final.py
import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from beaconai.eval.final import FinalEvalError, check_final_allowed, run_final_evaluation
from beaconai.models.calibrate import run_calibration
from beaconai.models.train import configure_mlflow, run_training
from tests.conftest import make_config, write_toy_processed

CALIBRATED = {"calibration": "temperature", "dev_sample": "None", "phase": "3"}


def test_guard_rules(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    check_final_allowed(None, CALIBRATED, lock)  # allowed
    with pytest.raises(FinalEvalError, match="full dataset"):
        check_final_allowed(50000, CALIBRATED, lock)
    with pytest.raises(FinalEvalError, match="calibrated"):
        check_final_allowed(None, {"model_type": "xgboost", "dev_sample": "None"}, lock)
    with pytest.raises(FinalEvalError, match="dev subsample"):
        check_final_allowed(None, {**CALIBRATED, "dev_sample": "50000"}, lock)
    lock.write_text("{}")
    with pytest.raises(FinalEvalError, match="already"):
        check_final_allowed(None, CALIBRATED, lock)


def _pipeline(tmp_path: Path):
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    run_calibration(cfg)
    return cfg


def test_final_evaluation_scores_the_champion_once(tmp_path: Path) -> None:
    cfg = _pipeline(tmp_path)
    report = run_final_evaluation(cfg)
    md = report.read_text(encoding="utf-8")
    assert "decision support, not diagnosis" in md
    assert "Partial evidence" in md and "Abstention" in md
    data = json.loads((tmp_path / "reports" / "final_evaluation.json").read_text())
    assert data["registered"]["alias"] == "champion"
    assert {"full", "partial"} <= set(data["metrics"])
    assert (tmp_path / "reports" / "final_test_lock.json").is_file()
    with pytest.raises(FinalEvalError, match="already"):
        run_final_evaluation(cfg)


def test_final_evaluation_refuses_a_raw_model_at_the_alias(tmp_path: Path) -> None:
    cfg = _pipeline(tmp_path)
    configure_mlflow(cfg.mlflow)
    client = MlflowClient()
    raw = client.get_model_version_by_alias("beaconai-classifier", "raw-champion")
    client.set_registered_model_alias("beaconai-classifier", "champion", raw.version)
    with pytest.raises(FinalEvalError, match="calibrated"):
        run_final_evaluation(cfg)
    assert not (tmp_path / "reports" / "final_test_lock.json").exists()
```

```python
# tests/eval/test_final_report.py
import unicodedata

from beaconai.eval.final_report import render_final

SUMMARY = {
    "registered": {"name": "beaconai-classifier", "version": 9, "alias": "champion"},
    "champion": {"model": "logreg+dropout", "method": "temperature", "fit_on": "partial"},
    "n_test": 134529,
    "threshold": 0.84,
    "metrics": {
        "full": {
            "top1_accuracy": 0.996,
            "top3_accuracy": 1.0,
            "top5_accuracy": 1.0,
            "macro_f1": 0.995,
            "ece": 0.001,
            "nll": 0.01,
            "ddx_recall_norm_at3": 0.70,
        },
        "partial": {
            "top1_accuracy": 0.7,
            "top3_accuracy": 0.85,
            "top5_accuracy": 0.9,
            "macro_f1": 0.6,
            "ece": 0.03,
            "nll": 1.0,
            "ddx_recall_norm_at3": 0.5,
        },
    },
    "abstention": {
        "full": {"coverage": 0.99, "top1_accuracy": 0.999, "topk_accuracy": 1.0},
        "partial": {
            "coverage": 0.0,
            "top1_accuracy": None,
            "topk_accuracy": None,
            "note": "No patients are answered at threshold 0.840; accuracy is undefined.",
        },
    },
}


def test_render_final() -> None:
    md = render_final(SUMMARY)
    assert "decision support, not diagnosis" in md
    assert "logreg+dropout + temperature (calibrated on partial)" in md
    assert "| Partial evidence | 0.700 | 0.850 |" in md
    assert "No patients are answered" in md  # a None accuracy shows its note, not NaN
    assert "scored once" in md
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}
```

In `tests/eval/test_evaluate.py`: delete the four `check_test_allowed` guard tests, `test_test_eval_runs_once_on_full_data` and `test_validation_report_never_overwrites_test_report`, and add:

```python
def test_test_split_is_refused_here(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    with pytest.raises(TestGuardError, match="make eval"):
        run_evaluation(cfg, "test")
```

Run `make test` and expect failures.

- [ ] **Step 2: Implement `final.py`**

```python
# src/beaconai/eval/final.py
"""The one-time test evaluation of the calibrated champion (make eval).

Run: uv run python -m beaconai.eval.final

This is the only code that reads the test split for metrics. It scores
`beaconai-classifier@champion`, and refuses unless that version is a full-data calibrated model,
the processed data is the full dataset, and no final evaluation has run before.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from beaconai.config import Config, load_config, project_path, require_calibration
from beaconai.config import require_training
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.final_report import render_final
from beaconai.eval.metrics import evaluate_predictions
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.eval.plots import plot_confusion
from beaconai.models.calibrate import answered_at, calibration_metrics, load_calibrated
from beaconai.models.dataset import load_features, load_labels, load_meta, load_split
from beaconai.models.train import configure_mlflow, data_scope

REPORT_FILE = "final_evaluation.md"
SUMMARY_FILE = "final_evaluation.json"


class FinalEvalError(Exception):
    """Raised when the final test evaluation would break its rules."""


def check_final_allowed(
    dev_sample_on_disk: int | None, version_tags: dict[str, str], lock_path: Path
) -> None:
    if dev_sample_on_disk is not None:
        raise FinalEvalError(
            "The final evaluation needs the full dataset on disk "
            "(BEACONAI_DEV_SAMPLE=null uv run python -m beaconai.data.preprocess)."
        )
    if "calibration" not in version_tags:
        raise FinalEvalError(
            "The champion alias does not point at a calibrated model; run `make calibrate` "
            "on full data first. make eval scores only the calibrated champion."
        )
    if version_tags.get("dev_sample") != "None":
        raise FinalEvalError("The champion was calibrated on the dev subsample, not full data.")
    if lock_path.is_file():
        raise FinalEvalError(
            f"The test split was already scored ({lock_path}). It is scored once; "
            "re-scoring after changes would leak test information."
        )


def run_final_evaluation(cfg: Config) -> Path:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    cal_cfg, abst = require_calibration(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    lock_path = reports / cfg.final_eval.lock_file
    configure_mlflow(mlflow_cfg, experiment=cfg.final_eval.experiment)
    name, alias = mlflow_cfg.registered_model, mlflow_cfg.champion_alias
    version = MlflowClient().get_model_version_by_alias(name, alias)
    check_final_allowed(data_scope(processed), dict(version.tags), lock_path)

    clf = load_calibrated(f"models:/{name}@{alias}")
    labels, features = load_labels(processed), load_features(processed)
    test, meta = load_split(processed, "test"), load_meta(processed, "test")
    partial_X = partial_evidence(
        test.X,
        meta["initial_evidence"],
        column_evidence(features),
        cal_cfg.partial_evidence.extra_evidences,
        cfg.seed,
    )
    k = abst.differential_size
    metrics: dict[str, dict[str, float]] = {}
    abstention: dict[str, dict[str, Any]] = {}
    results = {}
    for label, X in (("full", test.X), ("partial", partial_X)):
        proba = clf.predict_proba(X)
        res = evaluate_predictions(test.y, proba, test.ddx, eval_cfg.ks, eval_cfg.ddx_ks)
        metrics[label] = res.metrics | calibration_metrics(test.y, proba, cal_cfg.ece_bins)
        abstention[label] = answered_at(test.y, proba, clf.threshold, k)
        results[label] = res

    figure = plot_confusion(
        results["full"].confusion,
        [lab["name"] for lab in labels],
        reports / "figures" / "final_confusion_matrix.png",
        "Final champion (test, full evidence)",
    )
    summary = {
        "registered": {"name": name, "version": int(version.version), "alias": alias},
        "champion": {
            "model": version.tags.get("model_type"),
            "method": version.tags.get("calibration"),
            "fit_on": version.tags.get("calibrated_on", "full"),
        },
        "n_test": test.n,
        "threshold": clf.threshold,
        "metrics": metrics,
        "abstention": abstention,
    }
    with mlflow.start_run(run_name="final-test-evaluation"):
        mlflow.set_tags({"model_version": version.version, "split": "test"})
        for label, values in metrics.items():
            mlflow.log_metrics({f"test_{label}_{key}": v for key, v in values.items()})
        report = reports / REPORT_FILE
        report.write_text(render_final(summary), encoding="utf-8")
        (reports / SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for path in (report, reports / SUMMARY_FILE, figure):
            mlflow.log_artifact(str(path))
    lock = {
        "model_version": int(version.version),
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    try:
        print(run_final_evaluation(load_config()))
    except FinalEvalError as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: the champion's `model_type` tag carries the base label (for example `logreg+dropout`). In Task 4, set `model_type` in `register_version` tags to the candidate label.

- [ ] **Step 3: Implement `final_report.py`**

```python
# src/beaconai/eval/final_report.py
"""Render reports/final_evaluation.md. Pure string building."""

from __future__ import annotations

from typing import Any

DISCLAIMER = (
    "BeaconAI is decision support, not diagnosis. This model is trained only on synthetic "
    "data (DDXPlus) and is not a medical device."
)
ROWS = (("Full evidence", "full"), ("Partial evidence", "partial"))


def _f(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_final(summary: dict[str, Any]) -> str:
    champ, reg, m, abst = (
        summary["champion"],
        summary["registered"],
        summary["metrics"],
        summary["abstention"],
    )
    out = [
        "# Final test evaluation",
        "",
        f"> {DISCLAIMER}",
        "",
        f"Model: `{reg['name']}` version {reg['version']} (alias `{reg['alias']}`): "
        f"{champ['model']} + {champ['method']} (calibrated on {champ['fit_on']}).",
        f"Test patients: {summary['n_test']:,}. The test split is scored once; "
        "`reports/final_test_lock.json` blocks repeat scoring.",
        "Partial evidence: each patient keeps the initial evidence plus 3 other random evidences.",
        "",
        "## Accuracy and calibration",
        "",
        "| Evidence | Top-1 | Top-3 | Top-5 | Macro F1 | DDx nR@3 | ECE | NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {label} | {_f(m[key]['top1_accuracy'])} | {_f(m[key]['top3_accuracy'])} | "
            f"{_f(m[key]['top5_accuracy'])} | {_f(m[key]['macro_f1'])} | "
            f"{_f(m[key].get('ddx_recall_norm_at3'))} | {_f(m[key]['ece'], 4)} | "
            f"{_f(m[key]['nll'])} |"
            for label, key in ROWS
        ],
        "",
        "## Abstention",
        "",
        f"Threshold: {summary['threshold']:.3f}.",
        "",
        "| Evidence | Answered | Top-1 on answered | Top-3 on answered |",
        "|---|---:|---:|---:|",
        *[
            f"| {label} | {abst[key]['coverage']:.1%} | {_f(abst[key]['top1_accuracy'])} | "
            f"{_f(abst[key]['topk_accuracy'])} |"
            for label, key in ROWS
        ],
        "",
        *[
            f"Note ({label.lower()}): {abst[key]['note']}"
            for label, key in ROWS
            if "note" in abst[key]
        ],
        "",
        "![Confusion matrix](figures/final_confusion_matrix.png)",
        "",
    ]
    return "\n".join(out)
```

- [ ] **Step 4: Retire the Phase 2 test path.** In `evaluate.py`: `run_evaluation` raises `TestGuardError("The test split is scored only for the calibrated champion: run `make eval`.")` when `split == "test"`. Remove `check_test_allowed`, `LOCK_FILE` and the lock writing; keep `--split validate`. The CLI choices become `("validate",)`. Makefile: `eval:  ## Score the calibrated champion on test, once (full data)` → `uv run python -m beaconai.eval.final`.

- [ ] **Step 5:** `make test`, `make lint`. Commit: `Add the one-time final test evaluation of the calibrated champion`.

---

### Task 6: Dev run

- [ ] `make train && make train-dropout && make calibrate`. Expected: `reports/calibration_dev.md` with 20 candidates and the `Calibrated on` column; no baseline comparison (the dev run and the full baseline are different scopes); registry alias `dev` moves. Record per variant: partial and full ECE and top-3; the champion; the suggested threshold; and run times. If the dropout models lose more than 0.5 points of full-evidence top-3, ledger it and consider `copies`/`rate` before the full run.
- [ ] `make eval` must refuse (dev data on disk). Ledger the message.

---

### Task 7: Full run, threshold, the one-time test evaluation, docs

- [ ] **Full pipeline:**

```bash
BEACONAI_DEV_SAMPLE=null uv run python -m beaconai.data.preprocess
BEACONAI_DEV_SAMPLE=null make train-final
BEACONAI_DEV_SAMPLE=null make train-dropout
BEACONAI_DEV_SAMPLE=null make calibrate
```

- [ ] **Threshold:** read `reports/calibration.md` (it now includes the baseline comparison). Apply the Phase 3 rule (smallest threshold reaching 95% top-3 on answered partial-evidence patients), weigh it against coverage, set `abstention.threshold`, and re-run `BEACONAI_DEV_SAMPLE=null make calibrate` if it changed.
- [ ] **One-time test evaluation:** `BEACONAI_DEV_SAMPLE=null make eval`. Then run it a second time and confirm it is refused.
- [ ] **Restore dev data:** `make data`.
- [ ] **Docs:** `docs/phase3b_dropout.md` (plain, no em dashes) covering: the 2 x 2 results (baseline vs dropout x full vs partial calibration rows) against the Phase 3 baseline; the champion and why; the threshold; the final test numbers, with the honest note that this is the first and only look at test; why the yes/no/not-asked encoding was deferred; and what Phase 8 still needs. CHANGELOG entry with the key numbers. README: `make train-dropout`, `make eval` (final, once). Local phase line.
- [ ] **Verify and commit:** `make lint && make test`; commit the config, docs, reports (`calibration.md`, `calibration_summary.json`, `train_summary.json`, `train_dropout_summary.json`, `final_evaluation.md`, `final_evaluation.json`, `final_test_lock.json`, figures) as `Phase 3b: evidence dropout, partial-evidence calibration, and the final test evaluation`.

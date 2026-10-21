# Phase 3: Calibration, Abstention, and Champion Re-check Implementation Plan

> Execute task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate XGBoost and logistic regression on validation (temperature scaling and isotonic), measure ECE and reliability before and after on full and partial evidence, re-pick the champion by calibrated ECE with a top-3 floor, add a configurable abstention threshold justified by a coverage vs accuracy curve, and expose one `CalibratedClassifier.predict_differential` interface. Everything is logged to MLflow.

**Architecture:** Pure metric functions (`eval/calibration.py`) and pure calibrators with JSON state (`models/calibrators.py`, no pickles). A feature-level partial-evidence transform (`eval/partial.py`). `models/calibrated.py` wraps a Phase 2 model plus a calibrator behind `predict_differential`, which returns a Pydantic `Differential` or `Abstain`. `models/calibrate.py` runs the experiment: split validation into a fit part and a held-out part, score every (model, method) pair as a nested MLflow run, select the champion, draw plots, register it, and write `reports/calibration.md`.

**Tech Stack:** Python 3.11, numpy, scipy (temperature fit), scikit-learn (isotonic), Pydantic v2, MLflow, matplotlib, pytest.

**Spec:** `ROADMAP.md` Phase 3 (local file, including the "Champion re-check" note) plus the Phase 3 kickoff request: ECE and reliability diagrams for XGBoost and logistic regression on validation; temperature scaling and isotonic fit on validation, ECE before and after; champion by post-calibration ECE with top-3 accuracy as a floor, decision documented; configurable abstention threshold justified by a coverage vs accuracy plot; `CalibratedClassifier.predict_differential` returning the top-3 with probabilities or an `Abstain` with a reason; ECE on a partial-evidence validation set (initial evidence plus 3 random other evidences); everything logged to MLflow.

## Global Constraints

- Python `==3.11.*`; dependencies only through `uv add` (none new are expected).
- All tunable values (fit fraction, bins, bounds, floor, threshold, target, k) live in `config/base.yaml`.
- Seed everything with `cfg.seed`.
- Calibrators are fit on one part of validation and every reported number comes from the other part. The test split is not used in Phase 3 (it was scored once in Phase 2).
- Run on the dev subsample while iterating; the final numbers come from the full dataset.
- Only full-data runs move the `champion` registry alias; dev runs move `dev` (Phase 2 rule).
- Generated reports never overwrite each other: dev outputs carry a `_dev` suffix.
- Docs and reports: plain and direct, no em dashes. Every report states BeaconAI is decision support, not diagnosis, trained on synthetic data.
- No tool attribution or co-author trailers in commits, code comments, PR text, or any tracked file.
- On macOS run Python through `make` (shared OpenMP runtime; see `src/beaconai/models/openmp.py`).
- Phase ends with tests passing, `ruff check` clean, a `reports/CHANGELOG.md` entry with key metrics, and a commit.

## Method decisions (confirm at review)

1. **Held-out calibration scoring.** Validation is split per class (`calibration.fit_fraction: 0.5`). Isotonic regression in particular looks perfect on the rows it was fit to, so measuring ECE there would be meaningless.
2. **ECE definition.** Top-label ECE (confidence = top-1 probability), 15 equal-width bins, weighted by bin size (Guo et al. 2017). NLL is reported alongside.
3. **Partial evidence is per evidence, not per column.** An evidence is one question (for example `E_55`, with all its selected values). Each patient keeps the initial evidence plus 3 other random evidences; age and sex are always kept. Caveat recorded in the report: the models read a missing evidence as "no", so partial evidence also tests how a model trained on complete answers behaves when most answers are missing.
4. **Selection.** Candidates are (model, method) pairs, including `none` (raw). A candidate is eligible if its held-out top-3 accuracy is at least `calibration.top3_floor` (isotonic can change the ranking, temperature cannot). The champion is the eligible candidate with the lowest full-evidence held-out ECE. Partial-evidence ECE is reported, not used for selection.
5. **Abstention.** `predict_differential` abstains when the calibrated top-1 probability is below `abstention.threshold`. A patient whose top-1 probability equals the threshold is answered. The threshold is chosen from the partial-evidence coverage vs accuracy curve: the tool reports the smallest threshold whose answered patients reach `abstention.target_accuracy` top-3 accuracy, and the chosen value is written to config and justified in `docs/phase3_calibration.md`.

## Review Focus

1. **A calibrator scored on its own fit rows:** the fit and held-out parts must be disjoint and together cover validation. Test in Task 9 (`test_split_indices_*`).
2. **Isotonic output that does not sum to 1, or a row where every class maps to 0:** rows must be renormalised, and an all-zero row falls back to the uncalibrated row. Test in Task 4 (`test_isotonic_all_zero_row_falls_back`).
3. **`predict_differential` with an unknown evidence code, or an encoder whose columns do not match the model:** a clear `ValueError`, never a silent wrong prediction. Test in Task 6.
4. **A top-1 probability exactly at the threshold:** answered, and the same rule in `to_prediction` and in the coverage curve. Test in Tasks 3 and 6.
5. **Processed data on disk not matching the trained models, or a dev run moving `champion`:** refused or routed to `dev`, reusing the Phase 2 guards. Test in Task 9.

---

## File map

| File | Responsibility |
|---|---|
| `config/base.yaml`, `src/beaconai/config.py` | `calibration` and `abstention` sections; optional experiment name override |
| `src/beaconai/models/dataset.py` | `subset`, `load_meta`, `load_features` |
| `src/beaconai/eval/metrics.py` | `_check` becomes public `check_inputs` |
| `src/beaconai/eval/calibration.py` | Reliability bins, ECE, NLL, coverage curve, suggested threshold |
| `src/beaconai/models/calibrators.py` | Identity, temperature, isotonic calibrators with JSON state |
| `src/beaconai/eval/partial.py` | Partial-evidence transform |
| `src/beaconai/models/calibrated.py` | `PatientEvidence`, `Differential`, `Abstain`, `CalibratedClassifier`, `build_feature_space` |
| `src/beaconai/models/pyfunc.py` | `CalibratedPyfunc`, shared `log_pyfunc`, `PIP_REQUIREMENTS` |
| `src/beaconai/models/train.py` | `register_version` (shared alias rule), `configure_mlflow` experiment override |
| `src/beaconai/eval/plots.py` | `plot_reliability`, `plot_coverage` |
| `src/beaconai/eval/calibration_report.py` | Render `reports/calibration.md` |
| `src/beaconai/models/calibrate.py` | Experiment runner, `split_indices`, `select_calibrated`, `load_calibrated` |
| `tests/...` | Mirrors the modules above; `tests/conftest.py` toy data gains meta and evidence codes |
| `Makefile`, `README.md`, `reports/CHANGELOG.md`, `docs/phase3_calibration.md` | Commands, docs, decision record |

---

### Task 1: Config sections

**Files:**
- Modify: `config/base.yaml`, `src/beaconai/config.py`
- Test: `tests/test_config_calibration.py`

**Interfaces:**
- Produces: `CalibrationConfig(models, methods, fit_fraction, ece_bins, temperature_bounds, top3_floor, partial_evidence: PartialEvidenceConfig(extra_evidences), experiment)`, `AbstentionConfig(threshold, target_accuracy, differential_size, curve_points)`, optional `Config.calibration` / `Config.abstention`, `require_calibration(cfg) -> tuple[CalibrationConfig, AbstentionConfig]`, `CalibrationMethod = Literal["none", "temperature", "isotonic"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_calibration.py
import pytest
import yaml
from pydantic import ValidationError

from beaconai.config import (
    DEFAULT_CONFIG_PATH,
    CalibrationConfig,
    Config,
    load_config,
    require_calibration,
)


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_has_calibration_sections() -> None:
    cal, abst = require_calibration(load_config())
    assert cal.models == ["xgboost", "logreg"]
    assert cal.methods == ["none", "temperature", "isotonic"]
    assert cal.partial_evidence.extra_evidences == 3
    assert abst.differential_size == 3
    assert 0.0 <= abst.threshold <= 1.0


def test_require_calibration_rejects_missing_sections() -> None:
    raw = _raw()
    del raw["abstention"]
    with pytest.raises(ValueError, match="abstention"):
        require_calibration(Config.model_validate(raw))


@pytest.mark.parametrize("bounds", [[0.0, 5.0], [5.0, 1.0], [-1.0, 2.0]])
def test_temperature_bounds_must_be_positive_and_increasing(bounds: list[float]) -> None:
    raw = _raw()["calibration"]
    raw["temperature_bounds"] = bounds
    with pytest.raises(ValidationError, match="temperature_bounds"):
        CalibrationConfig.model_validate(raw)


def test_unknown_method_rejected() -> None:
    raw = _raw()["calibration"]
    raw["methods"] = ["platt"]
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate(raw)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_fit_fraction_strictly_between_zero_and_one(fraction: float) -> None:
    raw = _raw()["calibration"]
    raw["fit_fraction"] = fraction
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate(raw)
```

Run: `make test` → FAIL, `ImportError: cannot import name 'CalibrationConfig'`.

- [ ] **Step 2: Add the YAML sections**

Append to `config/base.yaml`:

```yaml
calibration:
  models: [xgboost, logreg]
  # none = raw probabilities, kept as a candidate so calibration has to earn its place.
  methods: [none, temperature, isotonic]
  # Validation is split per class: calibrators are fit on this share and every number is
  # reported on the rest, so no calibrator is scored on the rows it was fit to.
  fit_fraction: 0.5
  ece_bins: 15
  temperature_bounds: [0.05, 20.0]
  # A (model, method) pair can be champion only if its held-out top-3 accuracy is at least
  # this. Isotonic calibration can change the ranking; temperature scaling cannot.
  top3_floor: 0.99
  experiment: beaconai-phase3
  partial_evidence:
    # Each patient keeps the initial evidence plus this many other random evidences.
    extra_evidences: 3

abstention:
  # Abstain when the calibrated top-1 probability is below this (equal is answered).
  # Chosen from the partial-evidence coverage vs accuracy curve; see docs/phase3_calibration.md.
  threshold: 0.5
  # The run suggests the smallest threshold whose answered patients reach this top-k accuracy.
  target_accuracy: 0.95
  # k: how many conditions a differential shows.
  differential_size: 3
  # Thresholds evaluated for the coverage vs accuracy curve (evenly spaced from 0 to 1).
  curve_points: 101
```

- [ ] **Step 3: Add the schemas**

In `src/beaconai/config.py`, above `class Config`:

```python
CalibrationMethod = Literal["none", "temperature", "isotonic"]


class PartialEvidenceConfig(_StrictModel):
    extra_evidences: int = Field(default=3, ge=0)


class CalibrationConfig(_StrictModel):
    models: list[ModelName] = Field(min_length=1)
    methods: list[CalibrationMethod] = Field(min_length=1)
    fit_fraction: float = Field(default=0.5, gt=0, lt=1)
    ece_bins: int = Field(default=15, ge=2)
    temperature_bounds: tuple[float, float] = (0.05, 20.0)
    top3_floor: float = Field(default=0.99, ge=0, le=1)
    experiment: str | None = None
    partial_evidence: PartialEvidenceConfig = PartialEvidenceConfig()

    @field_validator("temperature_bounds")
    @classmethod
    def _bounds_increasing(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        low, high = bounds
        if not 0 < low < high:
            raise ValueError("temperature_bounds must satisfy 0 < low < high")
        return bounds


class AbstentionConfig(_StrictModel):
    threshold: float = Field(ge=0, le=1)
    target_accuracy: float = Field(default=0.95, gt=0, le=1)
    differential_size: int = Field(default=3, ge=1)
    curve_points: int = Field(default=101, ge=2)
```

Add to `Config`:

```python
    calibration: CalibrationConfig | None = None
    abstention: AbstentionConfig | None = None
```

Below `require_training`:

```python
def require_calibration(cfg: Config) -> tuple[CalibrationConfig, AbstentionConfig]:
    """Return the calibration sections, failing clearly if the config lacks any."""
    missing = [n for n in ("calibration", "abstention") if getattr(cfg, n) is None]
    if missing:
        raise ValueError(f"config is missing sections needed for calibration: {missing}")
    assert cfg.calibration and cfg.abstention
    return cfg.calibration, cfg.abstention
```

- [ ] **Step 4: Run tests**

Run: `make test` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add config/base.yaml src/beaconai/config.py tests/test_config_calibration.py
git commit -m "Add calibration and abstention config sections"
```

---

### Task 2: Data helpers and toy fixtures

**Files:**
- Modify: `src/beaconai/models/dataset.py`, `tests/conftest.py`
- Test: `tests/models/test_dataset.py`

**Interfaces:**
- Produces: `subset(data: SplitData, idx: np.ndarray) -> SplitData`, `load_meta(processed_dir, split) -> dict[str, np.ndarray]`, `load_features(processed_dir) -> list[dict]` (sorted by index; `load_feature_names` now uses it). Toy data: features carry `"evidence": f"E_{i}"` and `"kind": "binary"`; `{split}_meta.npz` holds `age`, `sex`, `initial_evidence` (= `E_{2y}`), `n_evidences`. `make_config` gains toy `calibration` and `abstention` sections.

- [ ] **Step 1: Update the toy fixtures**

In `tests/conftest.py`, inside `write_toy_processed`, after saving `{split}_ddx.npz`:

```python
        np.savez_compressed(
            out / f"{split}_meta.npz",
            age=np.full(n, 40, dtype=np.int16),
            sex=np.array(["F", "M"] * (n // 2) + ["F"] * (n % 2), dtype="U1"),
            initial_evidence=np.array([f"E_{2 * c}" for c in y]),
            n_evidences=np.asarray((X > 0).sum(axis=1), dtype=np.int16),
        )
```

Replace the toy `features` list with:

```python
    features = [
        {"index": i, "name": f"F_{i}", "kind": "binary", "label": f"Feature {i}", "evidence": f"E_{i}"}
        for i in range(TOY_FEATURES)
    ]
```

In `make_config`, before `return`:

```python
    raw["calibration"].update(ece_bins=10, top3_floor=0.5, partial_evidence={"extra_evidences": 1})
    raw["abstention"].update(curve_points=11, target_accuracy=0.9)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/models/test_dataset.py`:

```python
def test_subset_keeps_rows_aligned(toy_processed: Path) -> None:
    from beaconai.models.dataset import subset

    data = load_split(toy_processed, "train")
    idx = np.array([4, 0, 7])
    part = subset(data, idx)
    assert part.n == 3
    np.testing.assert_array_equal(part.y, data.y[idx])
    np.testing.assert_array_equal(part.X.toarray(), data.X[idx].toarray())
    assert part.ddx is not None
    np.testing.assert_array_equal(part.ddx.toarray(), data.ddx[idx].toarray())


def test_load_meta_and_features(toy_processed: Path) -> None:
    from beaconai.models.dataset import load_features, load_meta

    meta = load_meta(toy_processed, "validate")
    assert len(meta["initial_evidence"]) == TOY_ROWS["validate"]
    assert meta["initial_evidence"][1] == "E_2"  # row 1 is class 1, which always has F_2
    assert [f["evidence"] for f in load_features(toy_processed)][:2] == ["E_0", "E_1"]


def test_load_meta_missing_says_run_make_data(tmp_path: Path) -> None:
    from beaconai.models.dataset import load_meta

    with pytest.raises(FileNotFoundError, match="make data"):
        load_meta(tmp_path, "validate")
```

Run: `make test` → FAIL, `ImportError: cannot import name 'subset'`.

- [ ] **Step 3: Implement**

Append to `src/beaconai/models/dataset.py` (and change `load_feature_names` to use `load_features`):

```python
def subset(data: SplitData, idx: np.ndarray) -> SplitData:
    """Rows `idx` of a split, with X, y and the reference differential kept aligned."""
    return SplitData(
        X=data.X[idx],
        y=data.y[idx],
        ddx=None if data.ddx is None else data.ddx[idx],
    )


def load_meta(processed_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Raw age, sex, initial evidence code and evidence count per patient."""
    path = processed_dir / f"{split}_meta.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    with np.load(path) as meta:
        return {key: meta[key] for key in meta.files}


def load_features(processed_dir: Path) -> list[dict[str, Any]]:
    """features.json entries in column order."""
    return sorted(_read_json(processed_dir / "features.json"), key=lambda f: f["index"])


def load_feature_names(processed_dir: Path) -> list[str]:
    """Column names in matrix order, e.g. `E_55_@_V_14`."""
    return [f["name"] for f in load_features(processed_dir)]
```

- [ ] **Step 4: Run tests and commit**

Run: `make test` → all PASS.

```bash
git add src/beaconai/models/dataset.py tests/conftest.py tests/models/test_dataset.py
git commit -m "Add split subsetting, meta and feature loaders; richer toy data"
```

---

### Task 3: Calibration metrics

**Files:**
- Modify: `src/beaconai/eval/metrics.py` (rename `_check` to `check_inputs`, update its two callers)
- Create: `src/beaconai/eval/calibration.py`
- Test: `tests/eval/test_calibration.py`

**Interfaces:**
- Produces: `ReliabilityBins(edges, confidence, accuracy, counts)`, `reliability_bins(y, proba, n_bins)`, `expected_calibration_error(y, proba, n_bins=15) -> float`, `negative_log_likelihood(y, proba) -> float`, `CoverageCurve(thresholds, coverage, top1_accuracy, topk_accuracy, k)`, `coverage_curve(y, proba, thresholds, k)`, `suggest_threshold(curve, target) -> float | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_calibration.py
import math

import numpy as np
import pytest

from beaconai.eval.calibration import (
    coverage_curve,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    suggest_threshold,
)


def test_ece_zero_when_confidence_matches_accuracy() -> None:
    proba = np.tile([0.8, 0.2], (5, 1))
    y = np.array([0, 0, 0, 0, 1])  # 4 of 5 correct at confidence 0.8
    assert expected_calibration_error(y, proba, 15) == pytest.approx(0.0, abs=1e-12)


def test_ece_overconfident() -> None:
    proba = np.tile([0.9, 0.1], (4, 1))
    y = np.array([0, 0, 1, 1])  # 50% correct at confidence 0.9
    assert expected_calibration_error(y, proba, 15) == pytest.approx(0.4)


def test_ece_weights_bins_by_size() -> None:
    proba = np.array([[0.95, 0.05], [0.95, 0.05], [0.55, 0.45], [0.55, 0.45]])
    y = np.array([0, 0, 1, 1])  # high bin all right, low bin all wrong
    assert expected_calibration_error(y, proba, 15) == pytest.approx(0.5 * 0.05 + 0.5 * 0.55)


def test_confidence_of_one_lands_in_last_bin() -> None:
    bins = reliability_bins(np.array([0]), np.array([[1.0, 0.0]]), 10)
    assert bins.counts[-1] == 1 and bins.counts.sum() == 1
    assert np.isnan(bins.accuracy[0])  # empty bins are nan, not 0


def test_nll() -> None:
    assert negative_log_likelihood(np.array([0]), np.array([[0.5, 0.5]])) == pytest.approx(
        math.log(2)
    )
    # A zero probability on the true class is clipped, not infinite.
    assert math.isfinite(negative_log_likelihood(np.array([1]), np.array([[1.0, 0.0]])))


PROBA = np.array([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.4, 0.35, 0.25]])
Y = np.array([0, 0, 1])  # top-1 right, right, wrong; top-2 right for all three


def test_coverage_curve() -> None:
    curve = coverage_curve(Y, PROBA, [0.0, 0.5, 0.8, 0.95], k=2)
    np.testing.assert_allclose(curve.coverage, [1, 2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(curve.top1_accuracy[:3], [2 / 3, 1, 1])
    np.testing.assert_allclose(curve.topk_accuracy[:3], [1, 1, 1])
    assert np.isnan(curve.top1_accuracy[3])  # nobody answered


def test_probability_equal_to_threshold_is_answered() -> None:
    curve = coverage_curve(Y, PROBA, [0.6], k=1)
    assert curve.coverage[0] == pytest.approx(2 / 3)


def test_suggest_threshold() -> None:
    assert suggest_threshold(coverage_curve(Y, PROBA, [0.0, 0.5, 0.8], k=1), 1.0) == 0.5
    assert suggest_threshold(coverage_curve(Y, PROBA, [0.0, 0.5, 0.8], k=2), 1.0) == 0.0
    assert suggest_threshold(coverage_curve(Y, PROBA, [0.95], k=1), 0.5) is None
```

Run: `make test` → FAIL, `ModuleNotFoundError: beaconai.eval.calibration`.

- [ ] **Step 2: Make the input check public**

In `src/beaconai/eval/metrics.py`, rename `_check` to `check_inputs` (definition and both calls). No behavior change.

- [ ] **Step 3: Implement**

```python
# src/beaconai/eval/calibration.py
"""Calibration metrics: reliability bins, ECE, NLL, and coverage vs accuracy for abstention.

Confidence is the top-1 probability (top-label calibration, as in Guo et al. 2017).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from beaconai.eval.metrics import check_inputs, top_k_indices


@dataclass(frozen=True)
class ReliabilityBins:
    edges: np.ndarray  # n_bins + 1 equal-width edges over [0, 1]
    confidence: np.ndarray  # mean top-1 probability per bin, nan if the bin is empty
    accuracy: np.ndarray  # share of top-1 predictions that were right, nan if empty
    counts: np.ndarray


def reliability_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int) -> ReliabilityBins:
    y, p = check_inputs(y_true, proba)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_correct = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
        accuracy = np.where(counts > 0, sum_correct / counts, np.nan)
    return ReliabilityBins(np.linspace(0.0, 1.0, n_bins + 1), mean_conf, accuracy, counts)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy, weighted by how many patients each bin holds."""
    bins = reliability_bins(y_true, proba, n_bins)
    used = bins.counts > 0
    weights = bins.counts[used] / bins.counts.sum()
    return float(np.sum(weights * np.abs(bins.accuracy[used] - bins.confidence[used])))


def negative_log_likelihood(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    y, p = check_inputs(y_true, proba)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))))


@dataclass(frozen=True)
class CoverageCurve:
    thresholds: np.ndarray
    coverage: np.ndarray  # share of patients answered (top-1 probability >= threshold)
    top1_accuracy: np.ndarray  # on answered patients, nan if none answered
    topk_accuracy: np.ndarray  # on answered patients, nan if none answered
    k: int


def coverage_curve(
    y_true: np.ndarray, proba: np.ndarray, thresholds: Sequence[float], k: int
) -> CoverageCurve:
    y, p = check_inputs(y_true, proba)
    conf = p.max(axis=1)
    top1_hit = p.argmax(axis=1) == y
    topk_hit = (top_k_indices(p, k) == y[:, None]).any(axis=1)
    t = np.asarray(thresholds, dtype=np.float64)
    coverage, acc1, acck = [], [], []
    for threshold in t:
        answered = conf >= threshold
        n = int(answered.sum())
        coverage.append(n / len(y))
        acc1.append(float(top1_hit[answered].mean()) if n else np.nan)
        acck.append(float(topk_hit[answered].mean()) if n else np.nan)
    return CoverageCurve(t, np.array(coverage), np.array(acc1), np.array(acck), k)


def suggest_threshold(curve: CoverageCurve, target: float) -> float | None:
    """Smallest threshold whose answered patients reach `target` top-k accuracy, if any."""
    accuracy = np.nan_to_num(curve.topk_accuracy, nan=-1.0)
    ok = (curve.coverage > 0) & (accuracy >= target)
    return float(curve.thresholds[ok].min()) if ok.any() else None
```

- [ ] **Step 4: Run tests and commit**

Run: `make test` → all PASS.

```bash
git add src/beaconai/eval/metrics.py src/beaconai/eval/calibration.py tests/eval/test_calibration.py
git commit -m "Add ECE, reliability bins, NLL, and coverage vs accuracy metrics"
```

---

### Task 4: Calibrators

**Files:**
- Create: `src/beaconai/models/calibrators.py`
- Test: `tests/models/test_calibrators.py`

**Interfaces:**
- Produces: `Calibrator` ABC (`method: str`, `fit(proba, y) -> Self`, `transform(proba) -> np.ndarray`, `to_dict() -> dict`), `IdentityCalibrator`, `TemperatureCalibrator(bounds)` with `.temperature`, `IsotonicCalibrator`, `make_calibrator(method, bounds) -> Calibrator`, `calibrator_from_dict(d) -> Calibrator`. State is JSON only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_calibrators.py
import json

import numpy as np
import pytest

from beaconai.eval.calibration import expected_calibration_error
from beaconai.models.calibrators import (
    IsotonicCalibrator,
    TemperatureCalibrator,
    calibrator_from_dict,
    make_calibrator,
)

BOUNDS = (0.05, 20.0)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def overconfident() -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from softmax(z); the 'model' reports softmax(3z), so it is overconfident."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(6000, 4)) * 1.5
    true = _softmax(z)
    y = (rng.random(len(z))[:, None] < true.cumsum(axis=1)).argmax(axis=1)
    return _softmax(3 * z), y


def test_identity_returns_input() -> None:
    p = np.array([[0.7, 0.3]])
    np.testing.assert_array_equal(make_calibrator("none", BOUNDS).fit(p, [0]).transform(p), p)


def test_temperature_recovers_scale_and_keeps_ranking(overconfident) -> None:
    p, y = overconfident
    cal = TemperatureCalibrator(BOUNDS).fit(p, y)
    assert cal.temperature == pytest.approx(3.0, abs=0.4)
    q = cal.transform(p)
    np.testing.assert_array_equal(q.argsort(axis=1), p.argsort(axis=1))
    assert expected_calibration_error(y, q) < expected_calibration_error(y, p) / 2


def test_isotonic_reduces_ece_and_rows_sum_to_one(overconfident) -> None:
    p, y = overconfident
    q = IsotonicCalibrator().fit(p, y).transform(p)
    np.testing.assert_allclose(q.sum(axis=1), 1.0, rtol=1e-9)
    assert q.min() >= 0.0 and q.max() <= 1.0
    assert expected_calibration_error(y, q) < expected_calibration_error(y, p) / 2


def test_isotonic_all_zero_row_falls_back() -> None:
    # Every class maps to 0 for this input, so the row keeps its uncalibrated values.
    cal = calibrator_from_dict(
        {"method": "isotonic", "x": [[0.0, 1.0], [0.0, 1.0]], "y": [[0.0, 0.0], [0.0, 0.0]]}
    )
    np.testing.assert_allclose(cal.transform(np.array([[0.7, 0.3]])), [[0.7, 0.3]])


@pytest.mark.parametrize("method", ["none", "temperature", "isotonic"])
def test_round_trip_through_json(method: str, overconfident) -> None:
    p, y = overconfident
    cal = make_calibrator(method, BOUNDS).fit(p, y)
    state = json.loads(json.dumps(cal.to_dict()))
    np.testing.assert_allclose(calibrator_from_dict(state).transform(p), cal.transform(p))


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TemperatureCalibrator(BOUNDS).transform(np.array([[0.5, 0.5]]))


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        make_calibrator("platt", BOUNDS)
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrator_from_dict({"method": "platt"})
```

Run: `make test` → FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/models/calibrators.py
"""Post-hoc probability calibrators. State is plain JSON, so nothing is unpickled at load time."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Self

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(proba, dtype=np.float64), EPS, 1.0))


def _nll(y: np.ndarray, proba: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], EPS, 1.0))))


class Calibrator(ABC):
    method: ClassVar[str]

    @abstractmethod
    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self: ...

    @abstractmethod
    def transform(self, proba: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class IdentityCalibrator(Calibrator):
    """Raw probabilities, kept as a candidate so calibration has to beat doing nothing."""

    method = "none"

    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self:
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=np.float64).copy()

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}


class TemperatureCalibrator(Calibrator):
    """Divide log-probabilities by one temperature T fit by NLL. Never changes the ranking."""

    method = "temperature"

    def __init__(self, bounds: tuple[float, float], temperature: float | None = None) -> None:
        self.bounds = bounds
        self.temperature = temperature

    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self:
        logits, labels = _log(proba), np.asarray(y, dtype=np.int64)
        result = minimize_scalar(
            lambda t: _nll(labels, _softmax(logits / t)), bounds=self.bounds, method="bounded"
        )
        self.temperature = float(result.x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("temperature calibrator is not fitted")
        return _softmax(_log(proba) / self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature, "bounds": list(self.bounds)}


class IsotonicCalibrator(Calibrator):
    """One increasing step function per class (one-vs-rest), then rows renormalised.

    Can change the ranking, which is why selection applies a top-3 accuracy floor.
    """

    method = "isotonic"

    def __init__(
        self, x: list[list[float]] | None = None, y: list[list[float]] | None = None
    ) -> None:
        self.x, self.y = x, y

    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self:
        p, labels = np.asarray(proba, dtype=np.float64), np.asarray(y, dtype=np.int64)
        self.x, self.y = [], []
        for c in range(p.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p[:, c], (labels == c).astype(np.float64))
            self.x.append(iso.X_thresholds_.tolist())
            self.y.append(iso.y_thresholds_.tolist())
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None:
            raise RuntimeError("isotonic calibrator is not fitted")
        p = np.asarray(proba, dtype=np.float64)
        q = np.column_stack([np.interp(p[:, c], self.x[c], self.y[c]) for c in range(p.shape[1])])
        totals = q.sum(axis=1, keepdims=True)
        # A row where every class maps to 0 keeps its uncalibrated probabilities.
        return np.where(totals > 0, q / np.where(totals > 0, totals, 1.0), p)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "x": self.x, "y": self.y}


def make_calibrator(method: str, bounds: tuple[float, float]) -> Calibrator:
    if method == "none":
        return IdentityCalibrator()
    if method == "temperature":
        return TemperatureCalibrator(bounds)
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method {method!r}")


def calibrator_from_dict(state: dict[str, Any]) -> Calibrator:
    method = state.get("method")
    if method == "none":
        return IdentityCalibrator()
    if method == "temperature":
        return TemperatureCalibrator(tuple(state["bounds"]), state["temperature"])
    if method == "isotonic":
        return IsotonicCalibrator(state["x"], state["y"])
    raise ValueError(f"Unknown calibration method {method!r}")
```

Note: the all-zero fallback test builds a calibrator via `calibrator_from_dict` with only `x`/`y`, so `IsotonicCalibrator` takes them directly.

- [ ] **Step 3: Run tests and commit**

Run: `make test` → all PASS.

```bash
git add src/beaconai/models/calibrators.py tests/models/test_calibrators.py
git commit -m "Add identity, temperature, and isotonic calibrators with JSON state"
```

---

### Task 5: Partial-evidence transform

**Files:**
- Create: `src/beaconai/eval/partial.py`
- Test: `tests/eval/test_partial.py`

**Interfaces:**
- Consumes: `load_features`, `load_meta` (Task 2).
- Produces: `column_evidence(features: list[dict]) -> np.ndarray` (evidence code per column, `""` for age and sex), `partial_evidence(X, initial_evidence, col_evidence, extra, seed) -> csr_matrix`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_partial.py
import numpy as np
import pytest
from scipy import sparse

from beaconai.eval.partial import column_evidence, partial_evidence

FEATURES = [
    {"index": 0, "evidence": "E_1"},
    {"index": 1, "evidence": "E_2"},
    {"index": 2, "evidence": "E_3"},  # scale evidence, value 0.6
    {"index": 3, "evidence": "E_4"},  # multi-choice value V_1
    {"index": 4, "evidence": "E_4"},  # multi-choice value V_2
    {"index": 5, "evidence": "E_5"},
    {"index": 6, "evidence": "E_6"},
    {"index": 7, "evidence": None},  # age
    {"index": 8, "evidence": None},  # sex
]
COLS = column_evidence(FEATURES)
X = sparse.csr_matrix(
    np.array(
        [
            [1, 1, 0.6, 1, 1, 1, 1, 0.4, 1],  # initial E_1, five other evidences
            [1, 0, 0, 0, 0, 1, 0, 0.2, 0],  # initial E_5, one other evidence
        ],
        dtype=np.float32,
    )
)
INITIAL = np.array(["E_1", "E_5"])


def _kept_codes(row: sparse.csr_matrix) -> set[str]:
    return {c for c in COLS[row.indices] if c}


def test_column_evidence() -> None:
    assert COLS.tolist() == ["E_1", "E_2", "E_3", "E_4", "E_4", "E_5", "E_6", "", ""]


def test_keeps_initial_plus_extra_and_demographics() -> None:
    out = partial_evidence(X, INITIAL, COLS, extra=3, seed=0)
    kept = _kept_codes(out[0])
    assert "E_1" in kept and len(kept) == 4
    assert {7, 8} <= set(out[0].indices)  # age and sex always kept


def test_evidence_values_stay_together_and_unchanged() -> None:
    for seed in range(20):
        row = partial_evidence(X, INITIAL, COLS, extra=3, seed=seed)[0].toarray()[0]
        assert (row[3] == 0) == (row[4] == 0)  # both values of E_4 or neither
        assert row[2] in (0.0, pytest.approx(0.6))


def test_fewer_others_than_extra_keeps_everything() -> None:
    out = partial_evidence(X, INITIAL, COLS, extra=3, seed=0)
    np.testing.assert_array_equal(out[1].toarray(), X[1].toarray())


def test_extra_zero_keeps_only_initial() -> None:
    out = partial_evidence(X, INITIAL, COLS, extra=0, seed=0)
    assert _kept_codes(out[0]) == {"E_1"}


def test_seeded_and_varies_with_seed() -> None:
    a = partial_evidence(X, INITIAL, COLS, extra=2, seed=1).toarray()
    b = partial_evidence(X, INITIAL, COLS, extra=2, seed=1).toarray()
    np.testing.assert_array_equal(a, b)
    picks = {tuple(partial_evidence(X, INITIAL, COLS, 2, s)[0].indices) for s in range(20)}
    assert len(picks) > 1


def test_shape_mismatches_raise() -> None:
    with pytest.raises(ValueError):
        partial_evidence(X, INITIAL[:1], COLS, 3, 0)
    with pytest.raises(ValueError):
        partial_evidence(X, INITIAL, COLS[:5], 3, 0)
```

Run: `make test` → FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/eval/partial.py
"""A 'partial evidence' version of a split, closer to what users tell us.

Each patient keeps the initial evidence (what they came in with) plus a few other random
evidences. An evidence is one question with all its selected values (for example E_55 and every
E_55_@_V_* column), so a multi-choice answer is kept or dropped as a whole. Age and sex are
always kept. Dropped evidences become 0, which the models read the same as "no": this measures
a model trained on complete answers when most answers are missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def column_evidence(features: list[dict[str, Any]]) -> np.ndarray:
    """Evidence code per column, '' for columns that are not evidence (age, sex)."""
    ordered = sorted(features, key=lambda f: f["index"])
    return np.array([f.get("evidence") or "" for f in ordered], dtype=object)


def partial_evidence(
    X: sparse.csr_matrix,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    extra: int,
    seed: int,
) -> sparse.csr_matrix:
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
        n_pick = min(extra, len(others))
        picked = set(rng.choice(others, size=n_pick, replace=False)) if n_pick else set()
        keep = picked | {str(initial_evidence[i])}
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
```

- [ ] **Step 3: Run tests and commit**

Run: `make test` → all PASS.

```bash
git add src/beaconai/eval/partial.py tests/eval/test_partial.py
git commit -m "Add partial-evidence transform: initial evidence plus random others"
```

---

### Task 6: CalibratedClassifier and the prediction interface

**Files:**
- Create: `src/beaconai/models/calibrated.py`
- Test: `tests/models/test_calibrated.py`

**Interfaces:**
- Consumes: `ModelWrapper` (Phase 2), `Calibrator`, `calibrator_from_dict` (Task 4), `top_k_indices`, Phase 1 `FeatureSpace`, `load_catalog`.
- Produces:
  - Pydantic: `PatientEvidence(evidence: list[str], age: int, sex: "M" | "F")`, `DifferentialItem(condition, icd10, severity, probability)`, `Differential(kind="differential", items)`, `Abstain(kind="abstain", reason, top_probability)`, `Prediction = Differential | Abstain`
  - `to_prediction(proba_row, labels, k, threshold) -> Prediction` (abstain iff top-1 < threshold)
  - `CalibratedClassifier(model, calibrator, labels, feature_names, threshold, k, encoder=None)` with `predict_proba(X)`, `predictions(X) -> list[Prediction]`, `predict_differential(evidence: PatientEvidence) -> Prediction`, `with_encoder(encoder)`, `save(path)`, `CalibratedClassifier.load(path, encoder=None)`
  - `build_feature_space(cfg) -> FeatureSpace`

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_calibrated.py
from pathlib import Path

import numpy as np
import pytest

from beaconai.data.encode import FeatureSpace
from beaconai.data.schema import Catalog, parse_conditions, parse_evidences
from beaconai.models.calibrated import (
    Abstain,
    CalibratedClassifier,
    Differential,
    PatientEvidence,
    to_prediction,
)
from beaconai.models.calibrators import IdentityCalibrator, TemperatureCalibrator
from beaconai.models.registry import build_model
from tests.data.conftest import AGE_BINS, RAW_CONDITIONS, RAW_EVIDENCES

LABELS = [
    {"name": "Influenza", "icd10": "J10", "severity": 3},
    {"name": "Unstable angina", "icd10": "I20.0", "severity": 1},
    {"name": "Croup", "icd10": "J05.0", "severity": 2},
]


def _space() -> FeatureSpace:
    catalog = Catalog(
        evidences=parse_evidences(RAW_EVIDENCES), conditions=parse_conditions(RAW_CONDITIONS)
    )
    return FeatureSpace(catalog, AGE_BINS, age_scale=110, scale_max=10)


def _classifier(threshold: float = 0.0, k: int = 2) -> CalibratedClassifier:
    """Logistic regression on the Phase 1 toy catalog: fever + pain vs smoker + chest pain."""
    space = _space()
    flu, angina = ["E_1", "E_10_@_7"], ["E_2", "E_30_@_V_6"]
    tokens = [flu] * 30 + [angina] * 30
    X = space.encode_many(tokens, [30] * 60, ["F", "M"] * 30)
    y = np.array([0] * 30 + [1] * 30)
    model = build_model("logreg", {"max_iter": 200}, 2, 0).fit(X, y)
    return CalibratedClassifier(
        model, IdentityCalibrator(), LABELS[:2], space.names, threshold, k, encoder=space
    )


def test_to_prediction_returns_top_k_in_order() -> None:
    pred = to_prediction(np.array([0.2, 0.5, 0.3]), LABELS, k=2, threshold=0.4)
    assert isinstance(pred, Differential)
    assert [i.condition for i in pred.items] == ["Unstable angina", "Croup"]
    assert pred.items[0].probability == pytest.approx(0.5)
    assert pred.items[0].icd10 == "I20.0" and pred.items[0].severity == 1


def test_to_prediction_abstains_below_threshold_with_reason() -> None:
    pred = to_prediction(np.array([0.4, 0.35, 0.25]), LABELS, k=3, threshold=0.5)
    assert isinstance(pred, Abstain)
    assert pred.top_probability == pytest.approx(0.4)
    assert "0.40" in pred.reason and "0.50" in pred.reason


def test_probability_equal_to_threshold_is_answered() -> None:
    assert isinstance(to_prediction(np.array([0.5, 0.3, 0.2]), LABELS, 3, 0.5), Differential)


def test_predict_differential_from_evidence_codes() -> None:
    pred = _classifier().predict_differential(
        PatientEvidence(evidence=["E_1", "E_10_@_6"], age=34, sex="F")
    )
    assert isinstance(pred, Differential)
    assert pred.items[0].condition == "Influenza"
    assert sum(i.probability for i in pred.items) == pytest.approx(1.0)


def test_unknown_evidence_code_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="E_999"):
        _classifier().predict_differential(PatientEvidence(evidence=["E_999"], age=30, sex="M"))


def test_patient_evidence_validation() -> None:
    with pytest.raises(ValueError):
        PatientEvidence(evidence=[], age=-1, sex="M")
    with pytest.raises(ValueError):
        PatientEvidence(evidence=[], age=30, sex="X")


def test_encoder_column_mismatch_rejected() -> None:
    clf = _classifier()
    with pytest.raises(ValueError, match="columns"):
        CalibratedClassifier(
            clf.model, clf.calibrator, clf.labels, clf.feature_names[::-1], 0.5, 2, _space()
        )


def test_no_encoder_raises() -> None:
    clf = _classifier()
    bare = CalibratedClassifier(clf.model, clf.calibrator, clf.labels, clf.feature_names, 0.0, 2)
    with pytest.raises(RuntimeError, match="encoder"):
        bare.predict_differential(PatientEvidence(evidence=["E_1"], age=30, sex="M"))


def test_bad_k_or_label_count_rejected() -> None:
    clf = _classifier()
    with pytest.raises(ValueError):
        CalibratedClassifier(clf.model, clf.calibrator, clf.labels, clf.feature_names, 0.5, 3)
    with pytest.raises(ValueError):
        CalibratedClassifier(clf.model, clf.calibrator, LABELS, clf.feature_names, 0.5, 2)


def test_save_load_round_trip(tmp_path: Path) -> None:
    clf = _classifier(threshold=0.3)
    clf.calibrator = TemperatureCalibrator((0.05, 20.0), temperature=2.0)
    clf.save(tmp_path / "c")
    loaded = CalibratedClassifier.load(tmp_path / "c", encoder=_space())
    evidence = PatientEvidence(evidence=["E_2"], age=60, sex="M")
    assert loaded.predict_differential(evidence) == clf.predict_differential(evidence)
    assert loaded.threshold == 0.3 and loaded.k == 2
```

Run: `make test` → FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/models/calibrated.py
"""One prediction interface for everything downstream: CalibratedClassifier.predict_differential.

It returns the top-k conditions with calibrated probabilities, or an Abstain result with a reason
when the most likely condition is below the confidence threshold. BeaconAI is decision support,
not diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from beaconai.config import Config
from beaconai.data.download import raw_dir_from_config
from beaconai.data.encode import FeatureSpace
from beaconai.data.preprocess import CONDITIONS_FILE, EVIDENCES_FILE
from beaconai.data.schema import load_catalog
from beaconai.eval.metrics import top_k_indices
from beaconai.models.base import Matrix, ModelWrapper
from beaconai.models.calibrators import Calibrator, calibrator_from_dict

META_FILE = "calibrated.json"
BASE_DIR = "base"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PatientEvidence(_Frozen):
    """What we know about a patient: DDXPlus evidence tokens plus age and sex."""

    evidence: list[str]  # e.g. "E_91", "E_55_@_V_14", "E_56_@_7"
    age: int = Field(ge=0, le=130)
    sex: Literal["M", "F"]


class DifferentialItem(_Frozen):
    condition: str
    icd10: str = ""
    severity: int | None = None
    probability: float = Field(ge=0.0, le=1.0)


class Differential(_Frozen):
    kind: Literal["differential"] = "differential"
    items: list[DifferentialItem]


class Abstain(_Frozen):
    kind: Literal["abstain"] = "abstain"
    reason: str
    top_probability: float = Field(ge=0.0, le=1.0)


Prediction = Differential | Abstain


def to_prediction(
    proba_row: np.ndarray, labels: list[dict[str, Any]], k: int, threshold: float
) -> Prediction:
    """Top-k differential, or Abstain when the top-1 probability is below the threshold."""
    row = np.asarray(proba_row, dtype=np.float64)
    top = top_k_indices(row[None, :], k)[0]
    best = float(np.clip(row[top[0]], 0.0, 1.0))
    if best < threshold:
        return Abstain(
            reason=(
                "Not confident enough to rank conditions: the most likely one has probability "
                f"{best:.2f}, below the threshold of {threshold:.2f}. More information is needed."
            ),
            top_probability=best,
        )
    return Differential(
        items=[
            DifferentialItem(
                condition=labels[i]["name"],
                icd10=labels[i].get("icd10") or "",
                severity=labels[i].get("severity"),
                probability=float(np.clip(row[i], 0.0, 1.0)),
            )
            for i in top
        ]
    )


class CalibratedClassifier:
    def __init__(
        self,
        model: ModelWrapper,
        calibrator: Calibrator,
        labels: list[dict[str, Any]],
        feature_names: list[str],
        threshold: float,
        k: int,
        encoder: FeatureSpace | None = None,
    ) -> None:
        if model.n_features is None:
            raise ValueError("base model is not fitted")
        if len(labels) != model.n_classes:
            raise ValueError(f"{len(labels)} labels for a {model.n_classes}-class model")
        if len(feature_names) != model.n_features:
            raise ValueError(f"{len(feature_names)} feature names for {model.n_features} columns")
        if not 1 <= k <= model.n_classes:
            raise ValueError(f"k must be in [1, {model.n_classes}], got {k}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if encoder is not None and encoder.names != list(feature_names):
            raise ValueError("encoder columns do not match the model's feature columns")
        self.model = model
        self.calibrator = calibrator
        self.labels = list(labels)
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.k = int(k)
        self.encoder = encoder

    def predict_proba(self, X: Matrix) -> np.ndarray:
        """Calibrated probabilities, shape (n_patients, n_classes)."""
        return self.calibrator.transform(self.model.predict_proba(X))

    def predictions(self, X: Matrix) -> list[Prediction]:
        return [
            to_prediction(r, self.labels, self.k, self.threshold) for r in self.predict_proba(X)
        ]

    def encode(self, evidence: PatientEvidence) -> sparse.csr_matrix:
        if self.encoder is None:
            raise RuntimeError(
                "no encoder attached; use with_encoder(build_feature_space(cfg)) first"
            )
        try:
            values = self.encoder.encode(evidence.evidence, evidence.age, evidence.sex)
        except KeyError as err:
            raise ValueError(f"Unknown evidence code: {err.args[0]}") from err
        cols = sorted(values)
        return sparse.csr_matrix(
            (
                np.array([values[c] for c in cols], dtype=np.float32),
                (np.zeros(len(cols), dtype=np.int64), np.array(cols, dtype=np.int64)),
            ),
            shape=(1, len(self.feature_names)),
        )

    def predict_differential(self, evidence: PatientEvidence) -> Prediction:
        """Top-k conditions with calibrated probabilities, or Abstain with a reason."""
        return self.predictions(self.encode(evidence))[0]

    def with_encoder(self, encoder: FeatureSpace) -> CalibratedClassifier:
        return CalibratedClassifier(
            self.model,
            self.calibrator,
            self.labels,
            self.feature_names,
            self.threshold,
            self.k,
            encoder,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / BASE_DIR)
        meta = {
            "calibrator": self.calibrator.to_dict(),
            "labels": self.labels,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "k": self.k,
        }
        (path / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, encoder: FeatureSpace | None = None) -> CalibratedClassifier:
        path = Path(path)
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        return cls(
            ModelWrapper.load(path / BASE_DIR),
            calibrator_from_dict(meta["calibrator"]),
            meta["labels"],
            meta["feature_names"],
            meta["threshold"],
            meta["k"],
            encoder,
        )


def build_feature_space(cfg: Config) -> FeatureSpace:
    """The Phase 1 encoder (evidence tokens, age, sex -> feature columns) from config."""
    raw = raw_dir_from_config(cfg.data.paths.raw)
    enc = cfg.data.encoding
    catalog = load_catalog(raw / EVIDENCES_FILE, raw / CONDITIONS_FILE)
    return FeatureSpace(catalog, enc.age_bins, age_scale=enc.age_scale, scale_max=enc.scale_max)
```

- [ ] **Step 3: Run tests and commit**

Run: `make test` → all PASS. If `test_unknown_evidence_code_is_a_clear_error` fails because the catalog raises something other than `KeyError` for an unknown code, check `Catalog.evidence` in `src/beaconai/data/schema.py` and catch that type instead (ledger it).

```bash
git add src/beaconai/models/calibrated.py tests/models/test_calibrated.py
git commit -m "Add CalibratedClassifier with predict_differential and Abstain results"
```

---

### Task 7: Shared MLflow helpers (pyfunc logging, registration, experiment override)

**Files:**
- Modify: `src/beaconai/models/pyfunc.py`, `src/beaconai/models/train.py`
- Test: `tests/models/test_pyfunc.py`, existing `tests/models/test_train.py`

**Interfaces:**
- Produces:
  - `pyfunc.PIP_REQUIREMENTS` (moved from train), `pyfunc.to_csr(model_input) -> csr_matrix`
  - `log_pyfunc(python_model, model_dir: Path, example: pd.DataFrame, output: np.ndarray) -> str` (model URI)
  - `CalibratedPyfunc` (`.model: CalibratedClassifier` after load; `predict` returns calibrated probabilities)
  - `train.register_version(model_uri, cfg: MlflowConfig, dev_sample, tags: dict[str, str]) -> tuple[int, str]` (alias rule lives here; `register_champion` calls it)
  - `train.configure_mlflow(cfg: MlflowConfig, experiment: str | None = None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_pyfunc.py
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from beaconai.models.calibrated import CalibratedClassifier
from beaconai.models.calibrators import TemperatureCalibrator
from beaconai.models.dataset import load_feature_names, load_labels, load_split
from beaconai.models.pyfunc import CalibratedPyfunc, log_pyfunc
from beaconai.models.registry import build_model
from beaconai.models.train import configure_mlflow, register_version
from tests.conftest import make_config


def test_calibrated_model_logs_and_serves(tmp_path: Path, toy_processed: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    configure_mlflow(cfg.mlflow, experiment="pyfunc-test")
    train, test = load_split(toy_processed, "train"), load_split(toy_processed, "test")
    names = load_feature_names(toy_processed)
    model = build_model("logreg", {"max_iter": 200}, 3, 0).fit(train.X, train.y)
    clf = CalibratedClassifier(
        model,
        TemperatureCalibrator((0.05, 20.0), temperature=2.0),
        load_labels(toy_processed),
        names,
        0.5,
        3,
    )
    clf.save(tmp_path / "c")
    example = pd.DataFrame(test.X[:5].toarray().astype(np.float32), columns=names)
    with mlflow.start_run():
        uri = log_pyfunc(CalibratedPyfunc(), tmp_path / "c", example, clf.predict_proba(test.X[:5]))
    served = mlflow.pyfunc.load_model(uri)
    frame = pd.DataFrame(test.X.toarray(), columns=names)
    np.testing.assert_allclose(np.asarray(served.predict(frame)), clf.predict_proba(test.X))
    assert mlflow.get_experiment_by_name("pyfunc-test") is not None

    version, alias = register_version(uri, cfg.mlflow, None, {"phase": "3"})
    assert alias == "champion" and version == 1
    assert register_version(uri, cfg.mlflow, 50000, {})[1] == "dev"
```

Run: `make test` → FAIL, `ImportError: cannot import name 'CalibratedPyfunc'`.

- [ ] **Step 2: Implement in `pyfunc.py`**

Add imports (`from pathlib import Path`, `from mlflow.models import infer_signature`, and `from beaconai.models.calibrated import CalibratedClassifier`), move `PIP_REQUIREMENTS` here, and replace the body with:

```python
ARTIFACT_KEY = "model_dir"
PIP_REQUIREMENTS = ["numpy", "scipy", "pandas", "scikit-learn", "xgboost", "torch", "pydantic"]


def to_csr(model_input: Any) -> sparse.csr_matrix:
    """DataFrame, array, or sparse matrix in; float32 CSR out."""
    if isinstance(model_input, pd.DataFrame):
        model_input = model_input.to_numpy(dtype=np.float32)
    if sparse.issparse(model_input):
        return sparse.csr_matrix(model_input)
    return sparse.csr_matrix(np.asarray(model_input, dtype=np.float32))


class BeaconPyfunc(mlflow.pyfunc.PythonModel):
    """A raw Phase 2 model: features in, uncalibrated probabilities out."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = ModelWrapper.load(context.artifacts[ARTIFACT_KEY])

    # TypeFromExample: MLflow takes the input schema from the logged input example.
    def predict(self, context: Any, model_input: TypeFromExample, params: Any = None) -> np.ndarray:
        return self.model.predict_proba(to_csr(model_input))


class CalibratedPyfunc(mlflow.pyfunc.PythonModel):
    """A CalibratedClassifier: features in, calibrated probabilities out."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = CalibratedClassifier.load(Path(context.artifacts[ARTIFACT_KEY]))

    def predict(self, context: Any, model_input: TypeFromExample, params: Any = None) -> np.ndarray:
        return self.model.predict_proba(to_csr(model_input))


def log_pyfunc(
    python_model: mlflow.pyfunc.PythonModel,
    model_dir: Path,
    example: pd.DataFrame,
    output: np.ndarray,
) -> str:
    """Log a saved model directory as an MLflow pyfunc in the active run; return its URI."""
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=python_model,
        artifacts={ARTIFACT_KEY: str(model_dir)},
        pip_requirements=PIP_REQUIREMENTS,
        signature=infer_signature(example, output),
        input_example=example,
    )
    return info.model_uri
```

- [ ] **Step 3: Refactor `train.py` to use the helpers**

Replace `configure_mlflow` with:

```python
def configure_mlflow(cfg: MlflowConfig, experiment: str | None = None) -> None:
    """Point MLflow at the local store and select an experiment (default: cfg.experiment)."""
    name = experiment or cfg.experiment
    db = project_path(cfg.tracking_db)
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    if mlflow.get_experiment_by_name(name) is None:
        root = project_path(cfg.artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=root.as_uri())
    mlflow.set_experiment(name)
```

Replace `register_champion` with:

```python
def register_version(
    model_uri: str, cfg: MlflowConfig, dev_sample: int | None, tags: dict[str, str]
) -> tuple[int, str]:
    """Register a model version. Only full-data runs move the champion alias."""
    mv = mlflow.register_model(model_uri, cfg.registered_model)
    alias = cfg.champion_alias if dev_sample is None else cfg.dev_alias
    client = MlflowClient()
    client.set_registered_model_alias(cfg.registered_model, alias, mv.version)
    for key, value in {**tags, "dev_sample": str(dev_sample)}.items():
        client.set_model_version_tag(cfg.registered_model, mv.version, key, value)
    return int(mv.version), alias


def register_champion(
    champion: TrialResult, cfg: MlflowConfig, metric: str, dev_sample: int | None, mode: str
) -> tuple[int, str]:
    tags = {
        "model_type": champion.model,
        f"val_{metric}": str(champion.val_metrics[metric]),
        "mode": mode,
    }
    return register_version(champion.model_uri, cfg, dev_sample, tags)
```

In `run_trial`, replace the `mlflow.pyfunc.log_model(...)` block with:

```python
            model_uri = log_pyfunc(BeaconPyfunc(), tmp_path / "wrapper", example, val_proba[:5])
```

and return `model_uri` in the `TrialResult` in place of `info.model_uri`. Update imports (`from beaconai.models.pyfunc import BeaconPyfunc, log_pyfunc`; drop the now-unused `infer_signature`, `ARTIFACT_KEY`, and `PIP_REQUIREMENTS`).

- [ ] **Step 4: Run tests and commit**

Run: `make test` → all PASS (existing train tests prove the refactor kept behavior).

```bash
git add src/beaconai/models/pyfunc.py src/beaconai/models/train.py tests/models/test_pyfunc.py
git commit -m "Share pyfunc logging and registration; add calibrated pyfunc"
```

---

### Task 8: Plots and the calibration report

**Files:**
- Modify: `src/beaconai/eval/plots.py`
- Create: `src/beaconai/eval/calibration_report.py`
- Test: `tests/eval/test_plots.py`, `tests/eval/test_calibration_report.py`

**Interfaces:**
- Consumes: `ReliabilityBins`, `CoverageCurve` (Task 3).
- Produces: `plot_reliability(panels: dict[str, dict[str, ReliabilityBins]], path, title) -> Path`, `plot_coverage(curves: dict[str, CoverageCurve], threshold, path, title) -> Path`, `render_calibration(summary: dict, *, ece_bins: int, reliability_images: dict[str, str], coverage_image: str) -> str`. The `summary` dict layout is the one `run_calibration` writes (Task 9).

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_plots.py`:

```python
def test_calibration_plots_write_png(tmp_path: Path) -> None:
    from beaconai.eval.calibration import coverage_curve, reliability_bins
    from beaconai.eval.plots import plot_coverage, plot_reliability

    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1, 1])
    bins = reliability_bins(y, proba, 5)
    p1 = plot_reliability(
        {"Full evidence": {"none": bins}, "Partial evidence": {"none": bins}},
        tmp_path / "rel.png",
        "Reliability",
    )
    curve = coverage_curve(y, proba, np.linspace(0, 1, 11), k=1)
    p2 = plot_coverage({"Full evidence": curve}, 0.5, tmp_path / "cov.png", "Coverage")
    for p in (p1, p2):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
```

```python
# tests/eval/test_calibration_report.py
import unicodedata

from beaconai.eval.calibration_report import render_calibration


def _metrics(ece: float, top3: float, partial_ece: float) -> dict[str, float]:
    return {
        "ece": ece,
        "nll": 0.1,
        "top1_accuracy": 0.99,
        "top3_accuracy": top3,
        "partial_ece": partial_ece,
        "partial_nll": 1.5,
        "partial_top1_accuracy": 0.4,
        "partial_top3_accuracy": 0.7,
    }


SUMMARY = {
    "dev_sample": None,
    "n_fit": 66224,
    "n_held": 66224,
    "top3_floor": 0.99,
    "candidates": [
        {
            "model": "xgboost",
            "method": "none",
            "temperature": None,
            "metrics": _metrics(0.004, 1.0, 0.31),
        },
        {
            "model": "xgboost",
            "method": "temperature",
            "temperature": 1.4,
            "metrics": _metrics(0.002, 1.0, 0.25),
        },
        {
            "model": "logreg",
            "method": "isotonic",
            "temperature": None,
            "metrics": _metrics(0.001, 0.98, 0.2),
        },
    ],
    "champion": {"model": "xgboost", "method": "temperature"},
    "registered": {"name": "beaconai-classifier", "version": 4, "alias": "champion"},
    "abstention": {
        "threshold": 0.5,
        "suggested_threshold": 0.62,
        "target_accuracy": 0.95,
        "k": 3,
        "full": {"coverage": 0.999, "top1_accuracy": 0.998, "topk_accuracy": 1.0},
        "partial": {"coverage": 0.41, "top1_accuracy": 0.81, "topk_accuracy": 0.93},
    },
}


def _render() -> str:
    return render_calibration(
        SUMMARY,
        ece_bins=15,
        reliability_images={"xgboost": "figures/reliability_xgboost.png"},
        coverage_image="figures/coverage_accuracy.png",
    )


def test_report_tables_and_champion() -> None:
    md = _render()
    assert "decision support, not diagnosis" in md
    assert "| xgboost | temperature | 0.0020 |" in md
    assert "Champion: **xgboost + temperature**" in md
    assert "below the top-3 floor" in md  # logreg isotonic excluded on the floor
    assert "![xgboost reliability](figures/reliability_xgboost.png)" in md
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}  # docs style: no em dashes


def test_report_abstention_section() -> None:
    md = _render()
    assert "Threshold: 0.50" in md and "Suggested threshold" in md and "0.62" in md
    assert "| Partial evidence | 41.0% | 0.810 | 0.930 |" in md


def test_report_when_no_threshold_meets_target() -> None:
    summary = {**SUMMARY, "abstention": {**SUMMARY["abstention"], "suggested_threshold": None}}
    md = render_calibration(
        summary, ece_bins=15, reliability_images={}, coverage_image="figures/c.png"
    )
    assert "No threshold reaches" in md
```

Run: `make test` → FAIL.

- [ ] **Step 2: Implement the plots**

Append to `src/beaconai/eval/plots.py` (add `from beaconai.eval.calibration import CoverageCurve, ReliabilityBins` after the matplotlib setup):

```python
def plot_reliability(panels: dict[str, dict[str, ReliabilityBins]], path: Path, title: str) -> Path:
    """One panel per evidence setting; one line per calibration method; diagonal = perfect."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
    for ax, (panel, curves) in zip(axes[0], panels.items(), strict=True):
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
        for label, bins in curves.items():
            used = bins.counts > 0
            ax.plot(bins.confidence[used], bins.accuracy[used], marker="o", label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Probability of the top condition")
        ax.set_ylabel("Share where the top condition was right")
        ax.set_title(panel)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_coverage(
    curves: dict[str, CoverageCurve], threshold: float, path: Path, title: str
) -> Path:
    """Accuracy on answered patients as the model answers fewer of them."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, curve in curves.items():
        line = ax.plot(curve.coverage, curve.topk_accuracy, label=f"{label}, top-{curve.k}")[0]
        ax.plot(
            curve.coverage,
            curve.top1_accuracy,
            linestyle=":",
            color=line.get_color(),
            label=f"{label}, top-1",
        )
        i = int(np.argmin(np.abs(curve.thresholds - threshold)))
        ax.scatter(curve.coverage[i], curve.topk_accuracy[i], color=line.get_color(), zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Coverage (share of patients answered)")
    ax.set_ylabel("Accuracy on answered patients")
    ax.set_title(f"{title} (dots: threshold {threshold:.2f})")
    ax.legend(fontsize=8)
    return _save(fig, path)
```

- [ ] **Step 3: Implement the report renderer**

```python
# src/beaconai/eval/calibration_report.py
"""Render reports/calibration.md from the calibration summary. Pure string building."""

from __future__ import annotations

from typing import Any

DISCLAIMER = (
    "BeaconAI is decision support, not diagnosis. These models are trained only on "
    "synthetic data (DDXPlus) and are not a medical device."
)


def _f(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_calibration(
    summary: dict[str, Any],
    *,
    ece_bins: int,
    reliability_images: dict[str, str],
    coverage_image: str,
) -> str:
    champ = summary["champion"]
    floor = summary["top3_floor"]
    abst = summary["abstention"]
    scope = "dev subsample" if summary["dev_sample"] else "full dataset"
    out = [
        "# Calibration and abstention",
        "",
        f"> {DISCLAIMER}",
        "",
        "Generated by `make calibrate`. Do not edit by hand; the decision record is "
        "`docs/phase3_calibration.md`.",
        "",
        f"Data: DDXPlus {scope}, validation split. Calibrators are fit on {summary['n_fit']:,} "
        f"patients and every number below comes from the other {summary['n_held']:,}. "
        "The test split is not used.",
        f"ECE: top-label, {ece_bins} equal-width bins. Lower is better.",
        "",
        f"Champion: **{champ['model']} + {champ['method']}** (lowest held-out ECE among pairs "
        f"with top-3 accuracy of at least {floor:.3f}). Registered as "
        f"`{summary['registered']['name']}` version {summary['registered']['version']}, alias "
        f"`{summary['registered']['alias']}`.",
        "",
        "## Full evidence (held-out validation)",
        "",
        "| Model | Method | ECE | NLL | Top-1 | Top-3 | Temperature | Eligible |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in summary["candidates"]:
        m = c["metrics"]
        eligible = "yes" if m["top3_accuracy"] >= floor else "no, below the top-3 floor"
        out.append(
            f"| {c['model']} | {c['method']} | {_f(m['ece'], 4)} | {_f(m['nll'])} | "
            f"{_f(m['top1_accuracy'])} | {_f(m['top3_accuracy'])} | "
            f"{_f(c['temperature'], 2)} | {eligible} |"
        )
    out += [
        "",
        "## Partial evidence (held-out validation)",
        "",
        "Each patient keeps the initial evidence plus 3 other random evidences; age and sex are "
        'kept. The models read a missing evidence as "no", so this also shows how a model '
        "trained on complete answers behaves when most answers are missing.",
        "",
        "| Model | Method | ECE | NLL | Top-1 | Top-3 |",
        "|---|---|---:|---:|---:|---:|",
        *[
            f"| {c['model']} | {c['method']} | {_f(c['metrics']['partial_ece'], 4)} | "
            f"{_f(c['metrics']['partial_nll'])} | {_f(c['metrics']['partial_top1_accuracy'])} | "
            f"{_f(c['metrics']['partial_top3_accuracy'])} |"
            for c in summary["candidates"]
        ],
        "",
        "## Reliability diagrams",
        "",
        *[f"![{name} reliability]({img})" for name, img in reliability_images.items()],
        "",
        "## Abstention",
        "",
        f"Threshold: {abst['threshold']:.2f}. BeaconAI abstains when the calibrated probability "
        "of the most likely condition is below this; equal is answered.",
        "",
    ]
    if abst["suggested_threshold"] is None:
        out.append(
            f"No threshold reaches top-{abst['k']} accuracy of {abst['target_accuracy']:.2f} "
            "on answered partial-evidence patients."
        )
    else:
        out.append(
            f"Suggested threshold: {abst['suggested_threshold']:.2f} (smallest threshold whose "
            f"answered partial-evidence patients reach top-{abst['k']} accuracy of "
            f"{abst['target_accuracy']:.2f})."
        )
    out += [
        "",
        f"| Evidence | Answered | Top-1 on answered | Top-{abst['k']} on answered |",
        "|---|---:|---:|---:|",
        *[
            f"| {label} | {abst[key]['coverage']:.1%} | {_f(abst[key]['top1_accuracy'])} | "
            f"{_f(abst[key]['topk_accuracy'])} |"
            for label, key in (("Full evidence", "full"), ("Partial evidence", "partial"))
        ],
        "",
        f"![Coverage vs accuracy]({coverage_image})",
        "",
    ]
    return "\n".join(out)
```

- [ ] **Step 4: Run tests and commit**

Run: `make test` → all PASS.

```bash
git add src/beaconai/eval/plots.py src/beaconai/eval/calibration_report.py tests/eval/test_plots.py tests/eval/test_calibration_report.py
git commit -m "Add reliability and coverage plots and the calibration report"
```

---

### Task 9: Calibration runner, selection, and `make calibrate`

**Files:**
- Create: `src/beaconai/models/calibrate.py`
- Modify: `Makefile`
- Test: `tests/models/test_calibrate.py`

**Interfaces:**
- Consumes: everything above; Phase 2 `train_summary.json`, `load_wrapper`, `data_scope`, `check_data_scope`, `DataScopeError`.
- Produces:
  - `split_indices(y, fit_fraction, seed) -> tuple[np.ndarray, np.ndarray]` (stratified, disjoint, sorted)
  - `calibration_metrics(y, proba, n_bins) -> dict[str, float]` (`ece`, `nll`, `top1_accuracy`, `top3_accuracy`)
  - `CalibrationResult(model, method, run_id, model_uri, metrics, temperature)`
  - `select_calibrated(results, top3_floor) -> CalibrationResult`
  - `load_calibrated(model_uri) -> CalibratedClassifier`
  - `run_calibration(cfg) -> dict` writing `reports/calibration_summary{sfx}.json`, `reports/calibration{sfx}.md`, `reports/figures/reliability_<model>{sfx}.png`, `reports/figures/coverage_accuracy{sfx}.png` (`sfx` is `""` for full data, `"_dev"` for the dev subsample)

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_calibrate.py
import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from beaconai.models.calibrate import (
    CalibrationResult,
    load_calibrated,
    run_calibration,
    select_calibrated,
    split_indices,
)
from beaconai.models.calibrated import Abstain, Differential
from beaconai.models.dataset import load_split
from beaconai.models.train import DataScopeError, configure_mlflow, run_training
from tests.conftest import make_config, write_toy_processed


def test_split_indices_disjoint_cover_and_stratified() -> None:
    y = np.repeat([0, 1, 2], [100, 40, 1])
    fit, held = split_indices(y, 0.5, seed=0)
    assert set(fit).isdisjoint(held)
    assert sorted([*fit, *held]) == list(range(len(y)))
    assert np.sum(y[fit] == 0) == 50 and np.sum(y[fit] == 1) == 20
    assert 140 in held  # a class with one patient goes to the held-out part


def test_split_indices_seeded() -> None:
    y = np.arange(200) % 4
    a, b = split_indices(y, 0.5, 3), split_indices(y, 0.5, 3)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], split_indices(y, 0.5, 4)[0])


def _r(model: str, method: str, ece: float, top3: float) -> CalibrationResult:
    return CalibrationResult(model, method, "run", "uri", {"ece": ece, "top3_accuracy": top3}, None)


def test_select_calibrated_respects_floor() -> None:
    results = [
        _r("xgboost", "none", 0.010, 1.0),
        _r("xgboost", "temperature", 0.004, 1.0),
        _r("logreg", "isotonic", 0.001, 0.97),  # best ECE but below the floor
    ]
    best = select_calibrated(results, top3_floor=0.99)
    assert (best.model, best.method) == ("xgboost", "temperature")


def test_select_calibrated_none_eligible() -> None:
    with pytest.raises(ValueError, match="floor"):
        select_calibrated([_r("logreg", "none", 0.01, 0.5)], top3_floor=0.99)


def _trained(tmp_path: Path, dev_sample: int | None):
    cfg = make_config(tmp_path, dev_sample=dev_sample)
    write_toy_processed(tmp_path / "processed", dev_sample=dev_sample)
    run_training(cfg)
    return cfg


def test_run_calibration_full_data(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, None)
    summary = run_calibration(cfg)

    assert len(summary["candidates"]) == 6  # 2 models x 3 methods
    assert summary["registered"]["alias"] == "champion"
    assert summary["n_fit"] + summary["n_held"] == 60
    reports = tmp_path / "reports"
    assert json.loads((reports / "calibration_summary.json").read_text()) == summary
    md = (reports / "calibration.md").read_text(encoding="utf-8")
    assert "Champion:" in md and "Partial evidence" in md
    assert (reports / "figures" / "reliability_xgboost.png").is_file()
    assert (reports / "figures" / "coverage_accuracy.png").is_file()

    configure_mlflow(cfg.mlflow, experiment=cfg.calibration.experiment)
    runs = mlflow.search_runs(experiment_names=[cfg.calibration.experiment])
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert len(children) == 6
    assert children["metrics.ece"].notna().all()
    assert children["metrics.partial_ece"].notna().all()
    mv = MlflowClient().get_model_version_by_alias("beaconai-classifier", "champion")
    assert int(mv.version) == summary["registered"]["version"]
    assert mv.tags["calibration"] == summary["champion"]["method"]

    clf = load_calibrated(summary["champion"]["model_uri"])
    preds = clf.predictions(load_split(tmp_path / "processed" / "ddxplus", "test").X)
    assert all(isinstance(p, Differential | Abstain) for p in preds)


def test_dev_run_writes_dev_outputs_and_dev_alias(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, 50000)
    summary = run_calibration(cfg)
    assert summary["registered"]["alias"] == "dev"
    assert (tmp_path / "reports" / "calibration_dev.md").is_file()
    assert not (tmp_path / "reports" / "calibration.md").exists()


def test_refuses_when_data_on_disk_differs(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, None)
    manifest_path = tmp_path / "processed" / "ddxplus" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["settings"]["dev_sample"] = 50000
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataScopeError):
        run_calibration(cfg)


def test_missing_train_summary_says_run_make_train(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    with pytest.raises(FileNotFoundError, match="make train"):
        run_calibration(cfg)
```

Run: `make test` → FAIL, `ModuleNotFoundError: beaconai.models.calibrate`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/models/calibrate.py
"""Phase 3: calibrate the tuned models, pick the champion by calibrated ECE, set up abstention.

Run: uv run python -m beaconai.models.calibrate

Validation is split per class: calibrators are fit on one part and every number is reported on
the other part, so no calibrator is scored on the rows it was fit to. The test split is not used.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from beaconai.config import (
    AbstentionConfig,
    Config,
    load_config,
    project_path,
    require_calibration,
    require_training,
)
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.calibration import (
    coverage_curve,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    suggest_threshold,
)
from beaconai.eval.calibration_report import render_calibration
from beaconai.eval.metrics import top_k_accuracy
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.eval.plots import plot_coverage, plot_reliability
from beaconai.models.base import ModelWrapper
from beaconai.models.calibrated import CalibratedClassifier
from beaconai.models.calibrators import Calibrator, TemperatureCalibrator, make_calibrator
from beaconai.models.dataset import (
    SplitData,
    load_features,
    load_labels,
    load_meta,
    load_split,
    subset,
)
from beaconai.models.pyfunc import CalibratedPyfunc, log_pyfunc
from beaconai.models.train import (
    SUMMARY_FILE as TRAIN_SUMMARY_FILE,
)
from beaconai.models.train import (
    check_data_scope,
    configure_mlflow,
    data_scope,
    load_wrapper,
    register_version,
)

TOP3 = 3


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    method: str
    run_id: str
    model_uri: str
    metrics: dict[str, float]
    temperature: float | None


def split_indices(y: np.ndarray, fit_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class split of row indices into (fit, held-out). A one-patient class is held out."""
    rng = np.random.default_rng(seed)
    fit, held = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_fit = int(round(fit_fraction * len(idx)))
        if len(idx) >= 2:
            n_fit = min(max(n_fit, 1), len(idx) - 1)
        else:
            n_fit = 0
        fit.append(idx[:n_fit])
        held.append(idx[n_fit:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(held))


def calibration_metrics(y: np.ndarray, proba: np.ndarray, n_bins: int) -> dict[str, float]:
    return {
        "ece": expected_calibration_error(y, proba, n_bins),
        "nll": negative_log_likelihood(y, proba),
        "top1_accuracy": top_k_accuracy(y, proba, 1),
        "top3_accuracy": top_k_accuracy(y, proba, min(TOP3, proba.shape[1])),
    }


def select_calibrated(results: list[CalibrationResult], top3_floor: float) -> CalibrationResult:
    """Lowest held-out ECE among pairs whose top-3 accuracy meets the floor; ties keep order."""
    eligible = [r for r in results if r.metrics["top3_accuracy"] >= top3_floor]
    if not eligible:
        best = max(r.metrics["top3_accuracy"] for r in results)
        raise ValueError(
            f"No (model, method) pair meets the top-3 floor of {top3_floor}; best was {best:.4f}"
        )
    best = eligible[0]
    for r in eligible[1:]:
        if r.metrics["ece"] < best.metrics["ece"]:
            best = r
    return best


def load_calibrated(model_uri: str) -> CalibratedClassifier:
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().model


def _log_candidate(
    name: str,
    method: str,
    base: ModelWrapper,
    calibrator: Calibrator,
    metrics: dict[str, float],
    labels: list[dict[str, Any]],
    feature_names: list[str],
    abst: AbstentionConfig,
    held: SplitData,
) -> CalibrationResult:
    temperature = calibrator.temperature if isinstance(calibrator, TemperatureCalibrator) else None
    with mlflow.start_run(run_name=f"{name}-{method}", nested=True) as run:
        mlflow.set_tags(
            {"model_type": name, "calibration": method, "fit_on": "validate (fit part)"}
        )
        mlflow.log_params({"method": method, "temperature": temperature})
        mlflow.log_metrics(metrics)
        clf = CalibratedClassifier(
            base, calibrator, labels, feature_names, abst.threshold, abst.differential_size
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "calibrated"
            clf.save(model_dir)
            rows = held.X[:5]
            example = pd.DataFrame(rows.toarray().astype(np.float32), columns=feature_names)
            uri = log_pyfunc(CalibratedPyfunc(), model_dir, example, clf.predict_proba(rows))
    return CalibrationResult(name, method, run.info.run_id, uri, metrics, temperature)


def _at_threshold(y: np.ndarray, proba: np.ndarray, threshold: float, k: int) -> dict[str, float]:
    curve = coverage_curve(y, proba, [threshold], k)
    return {
        "coverage": float(curve.coverage[0]),
        "top1_accuracy": float(curve.top1_accuracy[0]),
        "topk_accuracy": float(curve.topk_accuracy[0]),
    }


def run_calibration(cfg: Config) -> dict[str, Any]:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    cal_cfg, abst = require_calibration(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    train_summary_path = reports / TRAIN_SUMMARY_FILE
    if not train_summary_path.is_file():
        raise FileNotFoundError(f"{train_summary_path} not found. Run `make train` first.")
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    dev_sample = data_scope(processed)
    check_data_scope(train_summary["dev_sample"], dev_sample)
    missing = [m for m in cal_cfg.models if m not in train_summary["models"]]
    if missing:
        raise ValueError(f"{missing} not in {train_summary_path}; add them to train.models")

    labels = load_labels(processed)
    features = load_features(processed)
    feature_names = [f["name"] for f in features]
    val = load_split(processed, "validate")
    meta = load_meta(processed, "validate")
    fit_idx, held_idx = split_indices(val.y, cal_cfg.fit_fraction, cfg.seed)
    fit, held = subset(val, fit_idx), subset(val, held_idx)
    partial = SplitData(
        partial_evidence(
            held.X,
            meta["initial_evidence"][held_idx],
            column_evidence(features),
            cal_cfg.partial_evidence.extra_evidences,
            cfg.seed,
        ),
        held.y,
        held.ddx,
    )
    sfx = "" if dev_sample is None else "_dev"
    figures = reports / "figures"
    k = abst.differential_size
    configure_mlflow(mlflow_cfg, experiment=cal_cfg.experiment)

    results: list[CalibrationResult] = []
    reliability_images: dict[str, str] = {}
    with mlflow.start_run(run_name="phase3-calibrate") as parent:
        mlflow.set_tags(
            {
                "dev_sample": str(dev_sample),
                "train_parent_run_id": train_summary["parent_run_id"],
                "phase": "3",
            }
        )
        mlflow.log_params(
            {
                "fit_fraction": cal_cfg.fit_fraction,
                "ece_bins": cal_cfg.ece_bins,
                "top3_floor": cal_cfg.top3_floor,
                "abstention_threshold": abst.threshold,
                "extra_evidences": cal_cfg.partial_evidence.extra_evidences,
                "n_fit": len(fit_idx),
                "n_held": len(held_idx),
            }
        )
        for name in cal_cfg.models:
            base = load_wrapper(train_summary["models"][name]["model_uri"])
            raw_fit = base.predict_proba(fit.X)
            raw_held = base.predict_proba(held.X)
            raw_partial = base.predict_proba(partial.X)
            panels: dict[str, dict[str, Any]] = {"Full evidence": {}, "Partial evidence": {}}
            for method in cal_cfg.methods:
                calibrator = make_calibrator(method, cal_cfg.temperature_bounds)
                calibrator.fit(raw_fit, fit.y)
                q_held, q_partial = (
                    calibrator.transform(raw_held),
                    calibrator.transform(raw_partial),
                )
                metrics = calibration_metrics(held.y, q_held, cal_cfg.ece_bins)
                partial_metrics = calibration_metrics(partial.y, q_partial, cal_cfg.ece_bins)
                metrics |= {f"partial_{key}": v for key, v in partial_metrics.items()}
                panels["Full evidence"][method] = reliability_bins(held.y, q_held, cal_cfg.ece_bins)
                panels["Partial evidence"][method] = reliability_bins(
                    partial.y, q_partial, cal_cfg.ece_bins
                )
                results.append(
                    _log_candidate(
                        name, method, base, calibrator, metrics, labels, feature_names, abst, held
                    )
                )
            image = f"reliability_{name}{sfx}.png"
            path = plot_reliability(panels, figures / image, f"{name}: reliability (held-out)")
            mlflow.log_artifact(str(path))
            reliability_images[name] = f"figures/{image}"

        champion = select_calibrated(results, cal_cfg.top3_floor)
        clf = load_calibrated(champion.model_uri)
        p_held, p_partial = clf.predict_proba(held.X), clf.predict_proba(partial.X)
        thresholds = np.linspace(0.0, 1.0, abst.curve_points)
        curves = {
            "Full evidence": coverage_curve(held.y, p_held, thresholds, k),
            "Partial evidence": coverage_curve(partial.y, p_partial, thresholds, k),
        }
        suggested = suggest_threshold(curves["Partial evidence"], abst.target_accuracy)
        coverage_image = f"coverage_accuracy{sfx}.png"
        cov_path = plot_coverage(
            curves,
            abst.threshold,
            figures / coverage_image,
            f"{champion.model} + {champion.method}",
        )
        mlflow.log_artifact(str(cov_path))
        full_at = _at_threshold(held.y, p_held, abst.threshold, k)
        partial_at = _at_threshold(partial.y, p_partial, abst.threshold, k)
        mlflow.log_metrics(
            {
                "champion_ece": champion.metrics["ece"],
                "abstain_rate_full": 1 - full_at["coverage"],
                "abstain_rate_partial": 1 - partial_at["coverage"],
                **({"suggested_threshold": suggested} if suggested is not None else {}),
            }
        )
        mlflow.set_tags({"champion_model": champion.model, "champion_method": champion.method})

    version, alias = register_version(
        champion.model_uri,
        mlflow_cfg,
        dev_sample,
        {
            "model_type": champion.model,
            "calibration": champion.method,
            "held_out_ece": f"{champion.metrics['ece']:.5f}",
            "phase": "3",
        },
    )
    summary = {
        "parent_run_id": parent.info.run_id,
        "train_parent_run_id": train_summary["parent_run_id"],
        "dev_sample": dev_sample,
        "n_fit": len(fit_idx),
        "n_held": len(held_idx),
        "top3_floor": cal_cfg.top3_floor,
        "champion": {
            "model": champion.model,
            "method": champion.method,
            "run_id": champion.run_id,
            "model_uri": champion.model_uri,
            "metrics": champion.metrics,
        },
        "registered": {"name": mlflow_cfg.registered_model, "version": version, "alias": alias},
        "candidates": [
            {
                "model": r.model,
                "method": r.method,
                "temperature": r.temperature,
                "metrics": r.metrics,
            }
            for r in results
        ],
        "abstention": {
            "threshold": abst.threshold,
            "suggested_threshold": suggested,
            "target_accuracy": abst.target_accuracy,
            "k": k,
            "full": full_at,
            "partial": partial_at,
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"calibration_summary{sfx}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    md = render_calibration(
        summary,
        ece_bins=cal_cfg.ece_bins,
        reliability_images=reliability_images,
        coverage_image=f"figures/{coverage_image}",
    )
    (reports / f"calibration{sfx}.md").write_text(md, encoding="utf-8")
    return summary


def main() -> int:
    summary = run_calibration(load_config())
    print(json.dumps({k: summary[k] for k in ("champion", "registered")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `json.dumps(summary)` fails on numpy floats. Every metric above is already a Python `float` (the metric functions return `float(...)`); if the test `json.loads(...) == summary` fails on a type, convert at the source and ledger it.

- [ ] **Step 3: Wire the Makefile**

```make
calibrate:  ## Calibrate models on validation, pick the champion by ECE, write reports/calibration.md
	uv run python -m beaconai.models.calibrate
```

Add `calibrate` to `.PHONY`.

- [ ] **Step 4: Run tests and commit**

Run: `make test` → all PASS; `make lint` clean.

```bash
git add src/beaconai/models/calibrate.py tests/models/test_calibrate.py Makefile
git commit -m "Add calibration runner with champion selection by calibrated ECE"
```

- [ ] **Step 5: Dev run on real data**

The current `reports/train_summary.json` is from the full-data run while the disk holds the dev subsample, so calibration refuses (by design). For the dev iteration:

Run: `make train && make calibrate`
Expected: `reports/calibration_dev.md`, `reports/calibration_summary_dev.json`, `reports/figures/reliability_{xgboost,logreg}_dev.png`, `reports/figures/coverage_accuracy_dev.png`; 6 child runs under the `beaconai-phase3` experiment; a new registry version with alias `dev` (not `champion`). Read the report: note raw vs calibrated ECE per model, partial-evidence ECE and accuracy, and the suggested threshold.

Note: `make train` rewrites `reports/train_summary.json` with the dev run. That file is restored by the full run in Task 10.

---

### Task 10: Full-data run, threshold choice, and docs

**Files:**
- Modify: `config/base.yaml` (`abstention.threshold`), `README.md`, `reports/CHANGELOG.md`, local phase line
- Create: `docs/phase3_calibration.md`

- [ ] **Step 1: Full data and models**

```bash
BEACONAI_DEV_SAMPLE=null uv run python -m beaconai.data.preprocess
BEACONAI_DEV_SAMPLE=null make train-final
BEACONAI_DEV_SAMPLE=null make calibrate
```

Expected: `reports/calibration.md` and `reports/calibration_summary.json` (no `_dev` suffix); registry alias `champion` moves to the calibrated champion. Note `train-final` re-registers the raw Phase 2 winner first (moving `champion` to it), and `make calibrate` then moves `champion` to the calibrated model.

- [ ] **Step 2: Choose the threshold**

From `reports/calibration.md` and the coverage plot, take the suggested threshold (smallest threshold whose answered partial-evidence patients reach the target top-3 accuracy). Weigh it against the abstain rate on partial evidence: a threshold that abstains on most partial-evidence patients is safe but unhelpful until Phase 8 asks follow-up questions. Set `abstention.threshold` in `config/base.yaml` with a one-line comment, then re-run `BEACONAI_DEV_SAMPLE=null make calibrate` so the registered model and report carry the chosen threshold.

- [ ] **Step 3: Restore dev data**

Run: `make data`

- [ ] **Step 4: Write `docs/phase3_calibration.md`** (plain, no em dashes): ECE before and after per model and method (full and partial), the champion decision under the rule (lowest calibrated ECE with the top-3 floor) and whether it changed from Phase 2, the threshold choice with the coverage and accuracy numbers that justify it, the partial-evidence caveat, and what Phase 8 needs from this (spread probabilities when evidence is missing).

- [ ] **Step 5: CHANGELOG, README, phase line**

Add a `## Phase 3: Calibration and abstention (date)` entry with: ECE before and after for both models (full and partial evidence), champion and method, threshold and abstain rates. README: `make calibrate`, the `predict_differential` example, and where the report lives. Set the local phase line to Phase 4.

- [ ] **Step 6: Verify and commit**

Run: `make lint && make test` → clean and all PASS.

```bash
git add config/base.yaml README.md reports/CHANGELOG.md docs/phase3_calibration.md reports/calibration.md reports/calibration_summary.json reports/figures reports/train_summary.json
git commit -m "Phase 3: calibration report, champion decision, and abstention threshold"
```

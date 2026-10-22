# Phase 4: Explainability for the Calibrated Logistic Regression Champion

> Execute task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every prediction can say why, in plain words. The explanations are exact for the linear champion: per-condition contributions (top 5 supporting, top 3 opposing), a contrastive "why A rather than B" for the top two conditions, readable feature labels, a SHAP cross-check, a global per-condition view with a sanity check, and a per-call latency under 50 ms.

**Architecture:** The champion is `CalibratedClassifier(LogRegModel, IsotonicCalibrator)`. Its raw probabilities are exactly `softmax(b + W x)`: this was checked on real data, so contributions need no approximation. `explain/linear.py` holds pure math on `(W, b, x)`. `explain/labels.py` turns a feature column and value into words. `explain/explainer.py` wraps the champion: `explain_evidence(PatientEvidence)` returns the prediction (a `Differential` or `Abstain`), an explanation per differential item, and a top-1 vs top-2 contrast. `explain/global_view.py` computes per-condition top features, automatic sanity flags, and how often calibration changes the ranking. `explain/run.py` (`make explain`) writes `reports/explainability.md`, plots, and an MLflow run.

**Tech Stack:** Python 3.11, numpy, scipy, scikit-learn, Pydantic v2, matplotlib, MLflow; `shap` as a dev dependency for the cross-check test only.

**Spec:** the Phase 4 request (2026-10-22), which adapts ROADMAP Phase 4 to the logistic regression champion:
- contribution of present feature j to condition c is `coef[c, j] * x[j]` in logit space, relative to an all-absent baseline
- readable labels from `features.json`
- top 5 positive and top 3 negative contributors
- contrastive explanation `(coef[A] - coef[B]) * x` for the top-1 vs top-2 conditions
- state that explanations are in logit space before isotonic calibration, and note cases where calibration changes the ranking
- a test that manual contributions match `shap.LinearExplainer` with the same zero baseline
- global top 10 features for 6 to 8 conditions as plots, plus a sanity-check section flagging clinically odd features
- per-call latency under 50 ms
- MLflow tag fixes, already done on this branch in `0ee7294`

## Global Constraints

- All tunables (k values, condition list, latency budget, flag thresholds) live in `config/base.yaml`.
- Explanations never change predictions. They read the same champion that `predict_differential` uses.
- Wording in code and docs: contributions are in the base model's logit space, before isotonic calibration.
- No test-split use: global statistics come from train (prevalence) and validation (ranking changes, latency).
- Docs and reports: plain and direct, no em dashes or escapes standing in for them. Every report states BeaconAI is decision support, not diagnosis, on synthetic data.
- No tool attribution or co-author trailers in commits, code comments, PR text, or any tracked file.
- On macOS run Python through `make`.
- Phase ends with tests passing, lint clean, a CHANGELOG entry, and a commit.

## Method decisions (confirm at review)

1. **What a single-condition contribution means.** The multinomial coefficients are only defined up to a per-feature constant shared by all conditions; adding the same number to every condition's coefficient changes no probability. The fitted L2 solution is centered, so `coef[c, j] * x[j]` reads as "support for condition c relative to the average condition". The contrast `(coef[A] - coef[B]) * x` has no such ambiguity, which is one more reason it is the headline explanation. Both are stated in docstrings and the report.
2. **Baseline.** The all-absent baseline is x = 0, so the baseline logit is the intercept. Age buckets, scaled age and sex are features too, so they appear as contributors. A female patient has `SEX_M = 0` and gets no sex contribution: female is part of the baseline. This is documented.
3. **Calibration and ranking.** The explanation uses the calibrated differential's order (what the user sees) for which conditions to explain, and the base model's logits for the reasons. When the calibrated top-1 has a lower logit than the calibrated top-2, the contrast is flagged `ranking_changed_by_calibration=True` with a note. The report quantifies how often calibration changes the top-1 and the top-3 set on held-out validation, with full and partial evidence.
4. **Global view.** For each configured condition: the 10 features with the largest positive coefficient (its contribution when fully present, since every feature is scaled to [0, 1]), alongside the feature's prevalence among training patients with that condition and among all training patients.
5. **Sanity check = automatic flags plus a human review.** Automatic flags per top feature:
   - (a) seen in fewer than `explain.min_prevalence` (1%) of patients with the condition;
   - (b) less common among patients with the condition than overall, yet pushing toward it (a reversed association, usually a sign of collinearity);
   - (c) a demographic feature (age or sex) in the top 10.
   A clinician-style review of each listed condition's top features is written in `docs/phase4_sanity_review.md`, and the report embeds it verbatim, so the sanity-check section sits inside the report.
6. **Latency.** It is measured end to end (encode `PatientEvidence`, calibrated predict, explain) on the first `explain.latency_calls` real validation patients from the raw CSV. The report shows median and p95 against the 50 ms budget. A unit test checks the same budget on a realistically sized model (49 conditions x 922 features).
7. **Non-linear champions are refused.** `ChampionExplainer` raises a clear error if the champion's base model is not logistic regression, rather than returning wrong explanations.

## Review Focus

1. **Manual contributions disagreeing with SHAP or with the model's own logits:** `b + sum(contributions)` must equal the logit, and each contribution must match `shap.LinearExplainer` with a zero background. Test in Task 2.
2. **A contrast whose sign contradicts the displayed order** (calibration flipped the top two): flagged, never silently presented as support. Test in Task 4.
3. **A patient with fewer than 5 supporting or 3 opposing features:** shorter lists, never padding with zero-contribution features. Test in Task 2.
4. **Abstain predictions:** the explanation is still returned (useful for "what would help"), with the `Abstain` prediction intact. Test in Task 4.
5. **A configured condition name that does not exist:** a clear config or run error, not a silent skip. Test in Tasks 1 and 5.

---

## File map

| File | Responsibility |
|---|---|
| `config/base.yaml`, `src/beaconai/config.py` | `explain` section |
| `pyproject.toml` | `shap` dev dependency |
| `src/beaconai/models/logreg.py` | `linear_parameters()` returns `(W, b)` aligned to all classes |
| `src/beaconai/explain/linear.py` | Pure contribution and contrast math |
| `src/beaconai/explain/labels.py` | Readable value display per feature kind |
| `src/beaconai/explain/explainer.py` | Pydantic outputs, `ChampionExplainer` |
| `src/beaconai/explain/global_view.py` | Top features, flags, ranking-change rates |
| `src/beaconai/explain/report.py` | Render `reports/explainability.md` |
| `src/beaconai/explain/run.py` | `make explain`: global plots, latency, report, MLflow |
| `src/beaconai/eval/plots.py` | `plot_top_features` |
| `docs/phase4_sanity_review.md` | Human review embedded in the report |
| `tests/explain/*` | Tests mirroring the modules |

---

### Task 1: Config and dependency

**Files:** `config/base.yaml`, `src/beaconai/config.py`, `pyproject.toml`; Test: `tests/test_config_explain.py`

**Interfaces:** `ExplainConfig(top_positive: int = 5, top_negative: int = 3, global_top: int = 10, global_conditions: list[str] (6 to 8 entries, unique), latency_budget_ms: float = 50, latency_calls: int = 300, min_prevalence: float = 0.01, experiment: str = "beaconai-phase4")`; `Config.explain: ExplainConfig | None`; `require_explain(cfg)`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_config_explain.py
import pytest
import yaml
from pydantic import ValidationError

from beaconai.config import DEFAULT_CONFIG_PATH, Config, ExplainConfig, load_config, require_explain


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_explain_section() -> None:
    ex = require_explain(load_config())
    assert (ex.top_positive, ex.top_negative, ex.global_top) == (5, 3, 10)
    assert 6 <= len(ex.global_conditions) <= 8
    assert ex.latency_budget_ms == 50


@pytest.mark.parametrize("n", [5, 9])
def test_global_condition_count(n: int) -> None:
    raw = _raw()["explain"]
    raw["global_conditions"] = [f"C{i}" for i in range(n)]
    with pytest.raises(ValidationError, match="6 to 8"):
        ExplainConfig.model_validate(raw)


def test_global_conditions_unique() -> None:
    raw = _raw()["explain"]
    raw["global_conditions"] = ["A", "A", "B", "C", "D", "E"]
    with pytest.raises(ValidationError, match="unique"):
        ExplainConfig.model_validate(raw)


def test_require_explain_missing() -> None:
    raw = _raw()
    del raw["explain"]
    with pytest.raises(ValueError, match="explain"):
        require_explain(Config.model_validate(raw))
```

- [ ] **Step 2: YAML**

```yaml
explain:
  # Contributors returned per explained condition.
  top_positive: 5
  top_negative: 3
  # Global view: top features per listed condition (names as in labels.json).
  global_top: 10
  global_conditions:
    - Pneumonia
    - Possible NSTEMI / STEMI
    - Unstable angina
    - Pulmonary embolism
    - Anaphylaxis
    - Influenza
    - Acute rhinosinusitis
    - Croup
  # Flag a top feature seen in fewer than this share of patients with the condition.
  min_prevalence: 0.01
  # End-to-end explain call (encode, calibrated predict, explain) on real validation patients.
  latency_budget_ms: 50
  latency_calls: 300
  experiment: beaconai-phase4
```

- [ ] **Step 3: Schema** in `config.py`:

```python
class ExplainConfig(_StrictModel):
    top_positive: int = Field(default=5, ge=1)
    top_negative: int = Field(default=3, ge=0)
    global_top: int = Field(default=10, ge=1)
    global_conditions: list[str]
    min_prevalence: float = Field(default=0.01, ge=0, le=1)
    latency_budget_ms: float = Field(default=50, gt=0)
    latency_calls: int = Field(default=300, ge=1)
    experiment: str = "beaconai-phase4"

    @field_validator("global_conditions")
    @classmethod
    def _six_to_eight_unique(cls, names: list[str]) -> list[str]:
        if not 6 <= len(names) <= 8:
            raise ValueError("global_conditions must list 6 to 8 conditions")
        if len(set(names)) != len(names):
            raise ValueError("global_conditions must be unique")
        return names


def require_explain(cfg: Config) -> ExplainConfig:
    if cfg.explain is None:
        raise ValueError("config is missing the explain section")
    return cfg.explain
```

Add `explain: ExplainConfig | None = None` to `Config`. Run `uv add --dev "shap>=0.46"` and confirm `make test` still passes (shap pulls numba). If shap does not install cleanly on Python 3.11, stop and report.

- [ ] **Step 4:** `make test`, `make lint`. Commit: `Add explain config and shap dev dependency`.

---

### Task 2: Linear parameters, contribution math, SHAP cross-check

**Files:** `src/beaconai/models/logreg.py`; create `src/beaconai/explain/linear.py`; Test: `tests/explain/test_linear.py`, `tests/models/test_logreg.py`

**Interfaces:**
- `LogRegModel.linear_parameters() -> tuple[np.ndarray, np.ndarray]`: `W` is (n_classes, n_features) and `b` is (n_classes,), in label order. Raises `ValueError` if a class was missing in training.
- `logits(W, b, x) -> np.ndarray`
- `contributions(W, x_row, c) -> list[tuple[int, float]]`: (feature index, coef * value) for present features only
- `top_contributors(pairs, n_pos, n_neg) -> tuple[list, list]`: positives sorted descending, negatives ascending, zeros excluded, ties broken by feature index
- `contrast_contributions(W, x_row, a, b) -> list[tuple[int, float]]`

- [ ] **Step 1: Failing tests**

```python
# tests/explain/test_linear.py
import numpy as np
import pytest
import shap
from scipy import sparse

from beaconai.explain.linear import (
    contrast_contributions,
    contributions,
    logits,
    top_contributors,
)

RNG = np.random.default_rng(0)
W = RNG.normal(size=(4, 7))
B = RNG.normal(size=4)
X = sparse.csr_matrix(
    np.array(
        [
            [1, 0, 0.6, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
)


def test_contributions_sum_to_logit_minus_intercept() -> None:
    z = logits(W, B, X[0])
    for c in range(4):
        total = sum(v for _, v in contributions(W, X[0], c))
        assert B[c] + total == pytest.approx(z[c])


def test_only_present_features_contribute() -> None:
    assert {j for j, _ in contributions(W, X[0], 2)} == {0, 2, 4, 6}
    assert contributions(W, X[0], 1)[1][1] == pytest.approx(W[1, 2] * 0.6, rel=1e-6)


def test_matches_shap_linear_explainer_with_zero_baseline() -> None:
    background = np.zeros((1, W.shape[1]))
    dense = X.toarray()
    for c in range(4):
        explainer = shap.LinearExplainer(
            (W[c], B[c]), shap.maskers.Independent(background, max_samples=1)
        )
        expected = explainer.shap_values(dense)
        for row in range(dense.shape[0]):
            ours = np.zeros(W.shape[1])
            for j, v in contributions(W, X[row], c):
                ours[j] = v
            np.testing.assert_allclose(ours, expected[row], atol=1e-6)


def test_top_contributors_order_and_no_padding() -> None:
    pairs = [(0, 2.0), (1, -0.5), (2, 0.0), (3, 3.0), (4, -1.5)]
    pos, neg = top_contributors(pairs, n_pos=5, n_neg=3)
    assert pos == [(3, 3.0), (0, 2.0)]  # zero excluded, only 2 positives exist
    assert neg == [(4, -1.5), (1, -0.5)]


def test_top_contributors_ties_by_feature_index() -> None:
    pos, _ = top_contributors([(5, 1.0), (2, 1.0)], 1, 0)
    assert pos == [(2, 1.0)]


def test_contrast_is_difference_of_contributions() -> None:
    a, b = 0, 3
    by_a = dict(contributions(W, X[0], a))
    by_b = dict(contributions(W, X[0], b))
    for j, v in contrast_contributions(W, X[0], a, b):
        assert v == pytest.approx(by_a[j] - by_b[j])
    gap = logits(W, B, X[0])[a] - logits(W, B, X[0])[b]
    total = sum(v for _, v in contrast_contributions(W, X[0], a, b))
    assert (B[a] - B[b]) + total == pytest.approx(gap)


def test_empty_row_has_no_contributions() -> None:
    empty = sparse.csr_matrix((1, 7), dtype=np.float32)
    assert contributions(W, empty, 0) == []
```

Append to `tests/models/test_logreg.py`:

```python
def test_linear_parameters_reproduce_predict_proba(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    W, b = model.linear_parameters()
    X = load_split(toy_processed, "test").X
    z = np.asarray(X @ W.T) + b
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p, model.predict_proba(X), rtol=1e-8)


def test_linear_parameters_need_every_class(toy_processed: Path) -> None:
    import pytest

    train = load_split(toy_processed, "train")
    keep = train.y != 2
    model = build_model("logreg", {}, 3, 42).fit(train.X[keep], train.y[keep])
    with pytest.raises(ValueError, match="missing"):
        model.linear_parameters()
```

Run `make test` and expect `ModuleNotFoundError: beaconai.explain.linear` and `AttributeError: linear_parameters`.

- [ ] **Step 2: Implement**

In `logreg.py`:

```python
def linear_parameters(self) -> tuple[np.ndarray, np.ndarray]:
    """(W, b) with logits = W @ x + b and probabilities = softmax(logits), in label order."""
    classes = np.asarray(self._clf.classes_)
    if len(classes) != self.n_classes:
        missing = sorted(set(range(self.n_classes)) - set(classes.tolist()))
        raise ValueError(f"classes {missing} were missing in training; no logits for them")
    order = np.argsort(classes)
    return self._clf.coef_[order].astype(np.float64), self._clf.intercept_[order].astype(np.float64)
```

```python
# src/beaconai/explain/linear.py
"""Exact explanations for a multinomial linear model, in logit space.

Logits are z = W x + b and the base model's probabilities are softmax(z). Relative to the
all-absent baseline (x = 0, logit = b), feature j adds W[c, j] * x[j] to condition c's logit.
The multinomial coefficients are only defined up to a per-feature constant shared by all
conditions; the fitted (L2) solution is centered, so a single-condition contribution reads as
support relative to the average condition. A contrast W[a] - W[b] has no such ambiguity.

These are the base model's logits, before isotonic calibration. Calibration changes the
probabilities shown to the user, and occasionally their order, but not these scores.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

Pairs = list[tuple[int, float]]


def _row(x_row: sparse.csr_matrix | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(column indices, values) of the non-zero features of one row."""
    if sparse.issparse(x_row):
        row = sparse.csr_matrix(x_row)
        if row.shape[0] != 1:
            raise ValueError("expected one row")
        return row.indices.astype(np.int64), row.data.astype(np.float64)
    dense = np.asarray(x_row, dtype=np.float64).ravel()
    cols = np.flatnonzero(dense)
    return cols, dense[cols]


def logits(W: np.ndarray, b: np.ndarray, x_row: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    cols, vals = _row(x_row)
    return b + W[:, cols] @ vals


def contributions(W: np.ndarray, x_row: sparse.csr_matrix | np.ndarray, c: int) -> Pairs:
    """(feature index, W[c, j] * x[j]) for every present feature j."""
    cols, vals = _row(x_row)
    return [(int(j), float(W[c, j] * v)) for j, v in zip(cols, vals, strict=True)]


def contrast_contributions(
    W: np.ndarray, x_row: sparse.csr_matrix | np.ndarray, a: int, b: int
) -> Pairs:
    """(feature index, (W[a, j] - W[b, j]) * x[j]): positive favors a, negative favors b."""
    cols, vals = _row(x_row)
    diff = W[a, cols] - W[b, cols]
    return [(int(j), float(d * v)) for j, d, v in zip(cols, diff, vals, strict=True)]


def top_contributors(pairs: Pairs, n_pos: int, n_neg: int) -> tuple[Pairs, Pairs]:
    """Largest positive and most negative contributions; zeros excluded; ties by feature index."""
    pos = sorted((p for p in pairs if p[1] > 0), key=lambda p: (-p[1], p[0]))[:n_pos]
    neg = sorted((p for p in pairs if p[1] < 0), key=lambda p: (p[1], p[0]))[:n_neg]
    return pos, neg
```

- [ ] **Step 3:** `make test` passes, including the SHAP cross-check. If `shap.LinearExplainer` returns a different shape in the installed version, adapt the test's indexing only (not the expected values) and ledger it. Commit: `Add exact linear contributions with a SHAP cross-check`.

---

### Task 3: Readable labels

**Files:** create `src/beaconai/explain/labels.py`; Test: `tests/explain/test_labels.py`

**Interfaces:** `FeatureLabeler(features: list[dict], scale_max: float, age_scale: float)`; `.label(j) -> str`; `.value_display(j, value) -> str`; `.describe(j, value) -> str` ("Fever above 38C: yes", "How intense is the pain? (0-10 scale): 6/10", "Age 30-44", "Age (scaled): 52", "Sex: male").

- [ ] **Step 1: Failing tests**

```python
# tests/explain/test_labels.py
import pytest

from beaconai.explain.labels import FeatureLabeler

FEATURES = [
    {"index": 0, "name": "E_91", "kind": "binary", "label": "Do you have a fever?"},
    {"index": 1, "name": "E_56", "kind": "scale", "label": "How intense is the pain? (0-10 scale)"},
    {"index": 2, "name": "E_55_@_V_14", "kind": "multi", "label": "Where is the pain? = chest"},
    {"index": 3, "name": "E_4_@_V_2", "kind": "categorical", "label": "Rash color = pink"},
    {"index": 4, "name": "AGE_BIN_30", "kind": "age_bucket", "label": "Age 30-44"},
    {"index": 5, "name": "AGE_SCALED", "kind": "age_scaled", "label": "Age (scaled)"},
    {"index": 6, "name": "SEX_M", "kind": "sex", "label": "Sex: male"},
]
L = FeatureLabeler(FEATURES, scale_max=10, age_scale=110)


@pytest.mark.parametrize(
    ("j", "value", "expected"),
    [
        (0, 1.0, "Do you have a fever?: yes"),
        (1, 0.6, "How intense is the pain? (0-10 scale): 6/10"),
        (2, 1.0, "Where is the pain? = chest"),
        (3, 1.0, "Rash color = pink"),
        (4, 1.0, "Age 30-44"),
        (5, 52 / 110, "Age: 52"),
        (6, 1.0, "Sex: male"),
    ],
)
def test_describe(j: int, value: float, expected: str) -> None:
    assert L.describe(j, value) == expected


def test_unknown_kind_falls_back_to_label() -> None:
    odd = FeatureLabeler([{"index": 0, "name": "X", "kind": "new", "label": "Thing"}], 10, 110)
    assert odd.describe(0, 1.0) == "Thing"
```

- [ ] **Step 2: Implement**

```python
# src/beaconai/explain/labels.py
"""Plain-English wording for a feature column and its value (from features.json)."""

from __future__ import annotations

from typing import Any


class FeatureLabeler:
    def __init__(self, features: list[dict[str, Any]], scale_max: float, age_scale: float) -> None:
        self._features = sorted(features, key=lambda f: f["index"])
        self.scale_max = scale_max
        self.age_scale = age_scale

    def name(self, j: int) -> str:
        return self._features[j]["name"]

    def label(self, j: int) -> str:
        return self._features[j]["label"]

    def describe(self, j: int, value: float) -> str:
        f = self._features[j]
        kind, label = f.get("kind"), f["label"]
        if kind == "binary":
            return f"{label}: yes"
        if kind == "scale":
            return f"{label}: {round(value * self.scale_max)}/{round(self.scale_max)}"
        if kind == "age_scaled":
            return f"Age: {round(value * self.age_scale)}"
        return label  # categorical / multi values, age buckets and sex already read as text
```

- [ ] **Step 3:** `make test`. Commit: `Add readable feature descriptions for explanations`.

---

### Task 4: `ChampionExplainer`

**Files:** create `src/beaconai/explain/explainer.py`; Test: `tests/explain/test_explainer.py`

**Interfaces:**
- Pydantic (frozen): `Contribution(feature, description, contribution)`, `ConditionExplanation(condition, logit, baseline_logit, supporting, opposing)`, `ContrastExplanation(condition, versus, logit_gap, baseline_gap, favors_condition, favors_versus, ranking_changed_by_calibration, note)`, `ExplainedPrediction(prediction: Differential | Abstain, explanations, contrast, space: str)`.
- `ChampionExplainer(clf: CalibratedClassifier, labeler: FeatureLabeler, top_positive=5, top_negative=3)`: raises `TypeError` unless `clf.model` is `LogRegModel`.
- Methods: `.explain_row(x_row) -> ExplainedPrediction`, `.explain_evidence(PatientEvidence) -> ExplainedPrediction` (needs `clf.encoder`).
- `SPACE_NOTE` constant: the logit-space wording.

Which conditions are explained: the top-k of the calibrated probabilities (the same items `to_prediction` shows, even when it abstains). The contrast is calibrated top-1 vs top-2.

- [ ] **Step 1: Failing tests**

```python
# tests/explain/test_explainer.py
import time

import numpy as np
import pytest
from scipy import sparse

from beaconai.explain.explainer import SPACE_NOTE, ChampionExplainer
from beaconai.explain.labels import FeatureLabeler
from beaconai.models.calibrated import Abstain, CalibratedClassifier, Differential
from beaconai.models.calibrators import IdentityCalibrator, calibrator_from_dict
from beaconai.models.registry import build_model

N_CLASSES, N_FEATURES = 4, 12
LABELS = [{"name": f"Cond {i}", "icd10": "", "severity": 3} for i in range(N_CLASSES)]
FEATURES = [
    {"index": j, "name": f"E_{j}", "kind": "binary", "label": f"Symptom {j}?"}
    for j in range(N_FEATURES)
]


def _model(n_classes=N_CLASSES, n_features=N_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(400) % n_classes
    X = (rng.random((400, n_features)) < 0.15).astype(np.float32)
    X[np.arange(400), y % n_features] = 1.0
    return build_model("logreg", {"max_iter": 300}, n_classes, 0).fit(sparse.csr_matrix(X), y)


def _explainer(threshold=0.0, calibrator=None):
    clf = CalibratedClassifier(
        _model(),
        calibrator or IdentityCalibrator(),
        LABELS,
        [f["name"] for f in FEATURES],
        threshold,
        3,
    )
    return ChampionExplainer(clf, FeatureLabeler(FEATURES, 10, 110), 5, 3)


ROW = sparse.csr_matrix(np.eye(1, N_FEATURES, 1, dtype=np.float32) + np.eye(1, N_FEATURES, 5))


def test_explains_each_differential_item_in_logit_space() -> None:
    out = _explainer().explain_row(ROW)
    assert isinstance(out.prediction, Differential)
    assert [e.condition for e in out.explanations] == [i.condition for i in out.prediction.items]
    first = out.explanations[0]
    W, b = _explainer().clf.model.linear_parameters()
    c = [lab["name"] for lab in LABELS].index(first.condition)
    assert first.baseline_logit == pytest.approx(b[c])
    total = sum(x.contribution for x in first.supporting + first.opposing)
    assert first.baseline_logit + total == pytest.approx(first.logit)  # 2 features <= 5 + 3
    assert out.space == SPACE_NOTE and "before" in SPACE_NOTE and "calibration" in SPACE_NOTE
    assert first.supporting[0].description.endswith(": yes")


def test_contrast_top1_vs_top2() -> None:
    out = _explainer().explain_row(ROW)
    con = out.contrast
    assert con is not None
    assert (con.condition, con.versus) == (
        out.prediction.items[0].condition,
        out.prediction.items[1].condition,
    )
    assert con.logit_gap >= 0 and not con.ranking_changed_by_calibration
    total = sum(x.contribution for x in con.favors_condition + con.favors_versus)
    assert con.baseline_gap + total == pytest.approx(con.logit_gap)


def test_flags_when_calibration_flips_the_top_two() -> None:
    # Isotonic maps that shrink class 0 to 1% and classes 2-3 to 0.01%: when the model's own
    # top-1 is class 0 (row with feature 0), calibration shows class 1 first and class 0 second.
    flip = {
        "method": "isotonic",
        "x": [[0.0, 1.0]] * N_CLASSES,
        "y": [[0.0, 0.01], [0.0, 1.0], [0.0, 0.0001], [0.0, 0.0001]],
    }
    exp = _explainer(calibrator=calibrator_from_dict(flip))
    row = sparse.csr_matrix(np.eye(1, N_FEATURES, 0, dtype=np.float32))
    assert int(np.argmax(exp.clf.model.predict_proba(row))) == 0  # precondition
    out = exp.explain_row(row)
    con = out.contrast
    assert (con.condition, con.versus) == ("Cond 1", "Cond 0")
    assert con.logit_gap < 0
    assert con.ranking_changed_by_calibration
    assert "calibration" in con.note.lower()


def test_abstain_still_explained() -> None:
    out = _explainer(threshold=1.0).explain_row(ROW)
    assert isinstance(out.prediction, Abstain)
    assert len(out.explanations) == 3 and out.contrast is not None


def test_refuses_non_linear_base() -> None:
    xgb = build_model("xgboost", {"n_estimators": 5}, N_CLASSES, 0).fit(
        ROW.toarray().repeat(8, 0), np.arange(8) % N_CLASSES
    )
    clf = CalibratedClassifier(
        xgb, IdentityCalibrator(), LABELS, [f["name"] for f in FEATURES], 0.0, 3
    )
    with pytest.raises(TypeError, match="logistic regression"):
        ChampionExplainer(clf, FeatureLabeler(FEATURES, 10, 110))


def test_latency_on_a_realistically_sized_model() -> None:
    n_classes, n_features = 49, 922
    feats = [
        {"index": j, "name": f"E_{j}", "kind": "binary", "label": f"S{j}"}
        for j in range(n_features)
    ]
    labels = [{"name": f"C{i}", "icd10": "", "severity": 3} for i in range(n_classes)]
    model = _model(n_classes, n_features, seed=1)
    clf = CalibratedClassifier(
        model, IdentityCalibrator(), labels, [f["name"] for f in feats], 0.5, 3
    )
    exp = ChampionExplainer(clf, FeatureLabeler(feats, 10, 110))
    rng = np.random.default_rng(2)
    rows = [
        sparse.csr_matrix((rng.random((1, n_features)) < 0.025).astype(np.float32))
        for _ in range(100)
    ]
    times = []
    for r in rows:
        start = time.perf_counter()
        exp.explain_row(r)
        times.append((time.perf_counter() - start) * 1000)
    assert np.percentile(times, 95) < 50
```

- [ ] **Step 2: Implement**

```python
# src/beaconai/explain/explainer.py
"""Why the champion ranks conditions the way it does, in plain words.

Explanations are exact for the logistic regression base model, in its logit space, before
isotonic calibration. Calibration changes the probabilities shown to the user (and occasionally
their order), not these scores. The contrast between the top two conditions is the most useful
explanation for a triage user, and the only one free of the multinomial baseline ambiguity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import sparse

from beaconai.eval.metrics import top_k_indices
from beaconai.explain.labels import FeatureLabeler
from beaconai.explain.linear import (
    Pairs,
    contrast_contributions,
    contributions,
    logits,
    top_contributors,
)
from beaconai.models.calibrated import (
    Abstain,
    CalibratedClassifier,
    Differential,
    PatientEvidence,
    to_prediction,
)
from beaconai.models.logreg import LogRegModel

SPACE_NOTE = (
    "Contributions are in the model's logit space before isotonic calibration, relative to a "
    "patient with no evidence (baseline = intercept). Calibration changes the probabilities "
    "shown, not these scores."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Contribution(_Frozen):
    feature: str
    description: str
    contribution: float


class ConditionExplanation(_Frozen):
    """Support for one condition relative to the average condition (centered coefficients)."""

    condition: str
    logit: float
    baseline_logit: float
    supporting: list[Contribution]
    opposing: list[Contribution]


class ContrastExplanation(_Frozen):
    """Why `condition` rather than `versus`: positive contributions favor `condition`."""

    condition: str
    versus: str
    logit_gap: float
    baseline_gap: float
    favors_condition: list[Contribution]
    favors_versus: list[Contribution]
    ranking_changed_by_calibration: bool
    note: str | None = None


class ExplainedPrediction(_Frozen):
    prediction: Differential | Abstain
    explanations: list[ConditionExplanation]
    contrast: ContrastExplanation | None
    space: str = SPACE_NOTE


class ChampionExplainer:
    def __init__(
        self,
        clf: CalibratedClassifier,
        labeler: FeatureLabeler,
        top_positive: int = 5,
        top_negative: int = 3,
    ) -> None:
        if not isinstance(clf.model, LogRegModel):
            raise TypeError(
                "Exact explanations are implemented for a logistic regression champion only; "
                f"got {type(clf.model).__name__}."
            )
        self.clf = clf
        self.labeler = labeler
        self.top_positive = top_positive
        self.top_negative = top_negative
        self.W, self.b = clf.model.linear_parameters()
        self.names = [lab["name"] for lab in clf.labels]

    def _contribs(self, pairs: Pairs, values: dict[int, float]) -> tuple[list, list]:
        pos, neg = top_contributors(pairs, self.top_positive, self.top_negative)

        def wrap(items: Pairs) -> list[Contribution]:
            return [
                Contribution(
                    feature=self.labeler.name(j),
                    description=self.labeler.describe(j, values[j]),
                    contribution=v,
                )
                for j, v in items
            ]

        return wrap(pos), wrap(neg)

    def explain_row(self, x_row: sparse.csr_matrix) -> ExplainedPrediction:
        row = sparse.csr_matrix(x_row)
        values = dict(zip(row.indices.tolist(), row.data.astype(float).tolist(), strict=True))
        proba = self.clf.predict_proba(row)[0]
        prediction = to_prediction(proba, self.clf.labels, self.clf.k, self.clf.threshold)
        shown = top_k_indices(proba[None, :], self.clf.k)[0]
        z = logits(self.W, self.b, row)

        explanations = []
        for c in shown:
            supporting, opposing = self._contribs(contributions(self.W, row, int(c)), values)
            explanations.append(
                ConditionExplanation(
                    condition=self.names[c],
                    logit=float(z[c]),
                    baseline_logit=float(self.b[c]),
                    supporting=supporting,
                    opposing=opposing,
                )
            )

        contrast = None
        if len(shown) >= 2:
            a, b = int(shown[0]), int(shown[1])
            favors_a, favors_b = self._contribs(contrast_contributions(self.W, row, a, b), values)
            gap = float(z[a] - z[b])
            flipped = gap < 0
            contrast = ContrastExplanation(
                condition=self.names[a],
                versus=self.names[b],
                logit_gap=gap,
                baseline_gap=float(self.b[a] - self.b[b]),
                favors_condition=favors_a,
                favors_versus=favors_b,
                ranking_changed_by_calibration=flipped,
                note=(
                    f"Calibration ranks {self.names[a]} above {self.names[b]}, but the model's "
                    "own scores rank them the other way; the reasons below explain the scores."
                    if flipped
                    else None
                ),
            )
        return ExplainedPrediction(
            prediction=prediction, explanations=explanations, contrast=contrast
        )

    def explain_evidence(self, evidence: PatientEvidence) -> ExplainedPrediction:
        return self.explain_row(self.clf.encode(evidence))


def to_dict(explained: ExplainedPrediction) -> dict[str, Any]:
    return explained.model_dump()
```

- [ ] **Step 3:** `make test`. Commit: `Add ChampionExplainer with per-condition and contrastive explanations`.

---

### Task 5: Global view, sanity flags, ranking-change rates

**Files:** create `src/beaconai/explain/global_view.py`; modify `src/beaconai/eval/plots.py`; Test: `tests/explain/test_global_view.py`

**Interfaces:**
- `TopFeature(index, name, label, coef, prevalence_condition, prevalence_all, flags: list[str])`
- `global_top_features(W, y, X, labels, labeler, conditions, n, min_prevalence) -> dict[str, list[TopFeature]]`: raises `ValueError` naming unknown conditions
- `feature_kind_is_demographic(feature) -> bool`
- `ranking_change_rates(raw_proba, calibrated_proba, k) -> dict[str, float]` with keys `top1_changed`, `topk_set_changed`
- `plot_top_features(features: list[TopFeature], condition, path) -> Path`

Flags: `"rarely seen with this condition"` (prevalence_condition < min_prevalence), `"less common with this condition than overall"` (prevalence_condition < prevalence_all), `"demographic"`.

- [ ] **Step 1: Failing tests**

```python
# tests/explain/test_global_view.py
import numpy as np
import pytest
from scipy import sparse

from beaconai.explain.global_view import global_top_features, ranking_change_rates
from beaconai.explain.labels import FeatureLabeler

FEATURES = [
    {"index": 0, "name": "E_0", "kind": "binary", "label": "Cough?"},
    {"index": 1, "name": "E_1", "kind": "binary", "label": "Fever?"},
    {"index": 2, "name": "E_2", "kind": "binary", "label": "Rare thing?"},
    {"index": 3, "name": "SEX_M", "kind": "sex", "label": "Sex: male"},
]
LABELS = [{"name": "Flu"}, {"name": "Cold"}]
W = np.array([[2.0, 1.0, 3.0, 0.5], [-1.0, 0.5, -3.0, -0.5]])
X = sparse.csr_matrix(
    np.array([[1, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
)
Y = np.array([0, 0, 1, 1])
LAB = FeatureLabeler(FEATURES, 10, 110)


def test_top_features_sorted_with_prevalence_and_flags() -> None:
    out = global_top_features(W, Y, X, LABELS, LAB, ["Flu"], n=3, min_prevalence=0.01)
    top = out["Flu"]
    assert [t.name for t in top] == ["E_2", "E_0", "E_1"]
    rare = top[0]
    assert rare.prevalence_condition == 0.0 and rare.prevalence_all == 0.25
    assert "rarely seen with this condition" in rare.flags
    assert "less common with this condition than overall" in rare.flags
    assert top[1].flags == []  # cough: 100% of flu vs 50% overall


def test_demographic_flag() -> None:
    out = global_top_features(W, Y, X, LABELS, LAB, ["Flu"], n=4, min_prevalence=0.01)
    sex = next(t for t in out["Flu"] if t.name == "SEX_M")
    assert "demographic" in sex.flags


def test_unknown_condition_is_an_error() -> None:
    with pytest.raises(ValueError, match="Measles"):
        global_top_features(W, Y, X, LABELS, LAB, ["Flu", "Measles"], 3, 0.01)


def test_ranking_change_rates() -> None:
    raw = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])
    cal = np.array([[0.6, 0.3, 0.1], [0.4, 0.5, 0.1], [0.3, 0.1, 0.6]])
    rates = ranking_change_rates(raw, cal, k=2)
    # Row 2: top-1 changes (0 -> 1), top-2 set stays {0, 1}.
    # Row 3: top-1 stays 2, top-2 set changes ({2, 1} -> {2, 0}).
    assert rates["top1_changed"] == pytest.approx(1 / 3)
    assert rates["topk_set_changed"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Implement**

```python
# src/beaconai/explain/global_view.py
"""Global view of the linear champion: top features per condition, sanity flags, and how often
calibration changes the ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from beaconai.eval.metrics import top_k_indices
from beaconai.explain.labels import FeatureLabeler

DEMOGRAPHIC_KINDS = {"age_bucket", "age_scaled", "sex"}


@dataclass(frozen=True)
class TopFeature:
    index: int
    name: str
    label: str
    coef: float
    prevalence_condition: float
    prevalence_all: float
    flags: list[str] = field(default_factory=list)


def global_top_features(
    W: np.ndarray,
    y: np.ndarray,
    X: sparse.csr_matrix,
    labels: list[dict[str, Any]],
    labeler: FeatureLabeler,
    conditions: list[str],
    n: int,
    min_prevalence: float,
) -> dict[str, list[TopFeature]]:
    """For each condition, the n features with the largest positive coefficient."""
    names = [lab["name"] for lab in labels]
    unknown = [c for c in conditions if c not in names]
    if unknown:
        raise ValueError(f"unknown conditions in explain.global_conditions: {unknown}")
    present = sparse.csr_matrix(X).copy()
    present.data = (present.data != 0).astype(np.float64)
    prevalence_all = np.asarray(present.mean(axis=0)).ravel()
    out: dict[str, list[TopFeature]] = {}
    for cond in conditions:
        c = names.index(cond)
        rows = y == c
        prev_c = (
            np.asarray(present[rows].mean(axis=0)).ravel() if rows.any() else np.zeros(X.shape[1])
        )
        order = sorted(range(W.shape[1]), key=lambda j: (-W[c, j], j))[:n]
        items = []
        for j in order:
            if W[c, j] <= 0:
                break
            flags = []
            if prev_c[j] < min_prevalence:
                flags.append("rarely seen with this condition")
            if prev_c[j] < prevalence_all[j]:
                flags.append("less common with this condition than overall")
            if labeler.kind(j) in DEMOGRAPHIC_KINDS:
                flags.append("demographic")
            items.append(
                TopFeature(
                    j,
                    labeler.name(j),
                    labeler.label(j),
                    float(W[c, j]),
                    float(prev_c[j]),
                    float(prevalence_all[j]),
                    flags,
                )
            )
        out[cond] = items
    return out


def ranking_change_rates(
    raw_proba: np.ndarray, calibrated_proba: np.ndarray, k: int
) -> dict[str, float]:
    """How often calibration changes the top-1 condition, or the set of top-k conditions."""
    raw_top = top_k_indices(raw_proba, k)
    cal_top = top_k_indices(calibrated_proba, k)
    top1 = raw_top[:, 0] != cal_top[:, 0]
    sets = np.array(
        [set(r) != set(c) for r, c in zip(raw_top.tolist(), cal_top.tolist(), strict=True)]
    )
    return {"top1_changed": float(top1.mean()), "topk_set_changed": float(sets.mean())}
```

Add `def kind(self, j: int) -> str: return self._features[j].get("kind", "")` to `FeatureLabeler`. Add `plot_top_features` to `eval/plots.py` as a horizontal bar chart of `coef`, with flagged bars hatched and a legend explaining the hatching.

- [ ] **Step 3:** `make test`. Commit: `Add global top features, sanity flags, and ranking-change rates`.

---

### Task 6: Report and `make explain`

**Files:** create `src/beaconai/explain/report.py`, `src/beaconai/explain/run.py`, `docs/phase4_sanity_review.md` (placeholder heading only at this step); modify `Makefile`; Test: `tests/explain/test_report.py`, `tests/explain/test_run.py`

**Interfaces:**
- `render_explainability(summary: dict, *, review_md: str | None) -> str`
- `run_explain(cfg) -> dict`, which writes `reports/explainability{sfx}.md`, `reports/explainability_summary{sfx}.json` and `reports/figures/top_features_<slug>{sfx}.png`
- Summary keys: `dev_sample`, `champion` (version, model, method, calibrated_on), `centering_max_abs_mean`, `ranking` (full and partial `ranking_change_rates` on the held-out half), `latency_ms` (median, p95, budget, n), `global` (condition to list of TopFeature dicts), `examples` (3 explained validation patients, as text)

The runner:
1. Loads `models:/beaconai-classifier@champion` with `build_feature_space(cfg)` attached, and builds the explainer (`TypeError` if not logistic regression).
2. Takes training prevalence from the processed train split.
3. Computes ranking-change rates on the held-out half of validation (the same `split_indices` as calibration), on full and on partial evidence (initial plus `calibration.partial_evidence.extra_evidences`).
4. Times `explain_evidence` end to end on the first `latency_calls` rows of `release_validate_patients.csv` (raw tokens, age, sex).
5. Explains 3 held-out patients with partial evidence as worked examples, written as plain sentences ("Why X rather than Y: ...").
6. Embeds `docs/phase4_sanity_review.md` if it exists.
7. Logs metrics, the report and the figures to MLflow (`explain.experiment`).

Report sections, in order:
- disclaimer
- "How to read these explanations" (logit space before calibration; baseline is a patient with no evidence, female, age contributions explicit; single-condition contributions are relative to the average condition, with the centering check; the contrast is the headline)
- calibration and ranking (the rates table)
- latency vs budget
- worked examples
- global view (one subsection per condition: plot plus a table with label, coefficient, prevalence with the condition, prevalence overall, flags)
- the sanity check (automatic flags summary, then the embedded human review)

- [ ] **Step 1: Failing tests**

```python
# tests/explain/test_report.py
import unicodedata

from beaconai.explain.report import render_explainability

SUMMARY = {
    "dev_sample": None,
    "champion": {
        "version": 13,
        "model": "logreg+dropout",
        "method": "isotonic",
        "calibrated_on": "partial",
    },
    "centering_max_abs_mean": 0.015,
    "ranking": {
        "full": {"top1_changed": 0.001, "topk_set_changed": 0.004},
        "partial": {"top1_changed": 0.012, "topk_set_changed": 0.05},
    },
    "latency_ms": {"median": 2.1, "p95": 3.4, "budget": 50, "n": 300},
    "examples": ["Why Pneumonia rather than Bronchitis: fever (+1.2), ..."],
    "global": {
        "Pneumonia": [
            {
                "index": 3,
                "name": "E_91",
                "label": "Fever?",
                "coef": 2.5,
                "prevalence_condition": 0.9,
                "prevalence_all": 0.3,
                "flags": [],
            },
            {
                "index": 9,
                "name": "SEX_M",
                "label": "Sex: male",
                "coef": 0.4,
                "prevalence_condition": 0.5,
                "prevalence_all": 0.5,
                "flags": ["demographic"],
            },
        ]
    },
    "images": {"Pneumonia": "figures/top_features_pneumonia.png"},
}


def test_report_sections_and_wording() -> None:
    md = render_explainability(SUMMARY, review_md="## Human review\n\nLooks sensible.")
    assert "decision support, not diagnosis" in md
    assert "logit space before isotonic calibration" in md
    assert "relative to the average condition" in md
    assert "| Full evidence | 0.1% | 0.4% |" in md
    assert "2.1" in md and "3.4" in md and "50" in md
    assert "![Pneumonia](figures/top_features_pneumonia.png)" in md
    assert "| Fever? | 2.500 | 90.0% | 30.0% |  |" in md
    assert "demographic" in md and "## Human review" in md
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}


def test_report_without_review_says_pending() -> None:
    assert "Human review pending" in render_explainability(SUMMARY, review_md=None)


def test_latency_over_budget_is_called_out() -> None:
    slow = {**SUMMARY, "latency_ms": {**SUMMARY["latency_ms"], "p95": 80}}
    assert "over the 50 ms budget" in render_explainability(slow, review_md=None)
```

`run_explain(cfg, *, patients: list[PatientEvidence] | None = None)`: when `patients` is None, the runner reads the first `latency_calls` rows of the raw validation CSV. The test passes its own patients and patches `build_feature_space`.

```python
# tests/explain/test_run.py
import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from scipy import sparse

from beaconai.config import ExplainConfig
from beaconai.data.encode import FeatureSpace
from beaconai.data.schema import Catalog, parse_conditions, parse_evidences
from beaconai.models.calibrate import run_calibration
from beaconai.models.calibrated import PatientEvidence
from beaconai.models.train import configure_mlflow, run_training
from tests.conftest import make_config
from tests.data.conftest import AGE_BINS, RAW_CONDITIONS, RAW_EVIDENCES

TOKENS = {
    0: ["E_1", "E_10_@_7", "E_30_@_V_5"],
    1: ["E_2", "E_30_@_V_6", "E_20_@_V_2"],
    2: ["E_1", "E_2", "E_10_@_3"],
}
CONDITIONS = ["Influenza", "Unstable angina", "Croup"]


def _space() -> FeatureSpace:
    catalog = Catalog(
        evidences=parse_evidences(RAW_EVIDENCES), conditions=parse_conditions(RAW_CONDITIONS)
    )
    return FeatureSpace(catalog, AGE_BINS, age_scale=110, scale_max=10)


def write_catalog_processed(root: Path) -> Path:
    """Processed data whose columns are the Phase 1 toy catalog's, so the encoder lines up."""
    space = _space()
    out = root / "ddxplus"
    out.mkdir(parents=True)
    for split, n in {"train": 150, "validate": 60, "test": 30}.items():
        y = np.arange(n) % 3
        tokens = [TOKENS[c] for c in y]
        ages = [20 + (i % 50) for i in range(n)]
        sexes = ["F", "M"] * (n // 2)
        X = space.encode_many(tokens, ages, sexes)
        ddx = np.zeros((n, 3), dtype=np.float32)
        ddx[np.arange(n), y] = 1.0
        sparse.save_npz(out / f"{split}_X.npz", X)
        np.save(out / f"{split}_y.npy", y.astype(np.int16))
        sparse.save_npz(out / f"{split}_ddx.npz", sparse.csr_matrix(ddx))
        np.savez_compressed(
            out / f"{split}_meta.npz",
            age=np.array(ages, dtype=np.int16),
            sex=np.array(sexes, dtype="U1"),
            initial_evidence=np.array([t[0] for t in tokens]),
            n_evidences=np.array([len(t) for t in tokens], dtype=np.int16),
        )
    labels = [
        {"index": i, "name": n, "name_fr": "", "icd10": "", "severity": 3}
        for i, n in enumerate(CONDITIONS)
    ]
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "features.json").write_text(json.dumps([f.to_dict() for f in space.features]))
    manifest = {
        "splits": {},
        "n_features": space.n_features,
        "n_classes": 3,
        "settings": {"dev_sample": None},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def _cfg(tmp_path: Path, conditions: list[str]):
    cfg = make_config(tmp_path, dev_sample=None)
    write_catalog_processed(tmp_path / "processed")
    run_training(cfg)
    run_calibration(cfg)
    # model_construct: the toy has 3 conditions, below the 6-to-8 rule meant for real data.
    explain = ExplainConfig.model_construct(
        top_positive=5,
        top_negative=3,
        global_top=4,
        global_conditions=conditions,
        min_prevalence=0.01,
        latency_budget_ms=50,
        latency_calls=5,
        experiment="explain-test",
    )
    return cfg.model_copy(update={"explain": explain})


PATIENTS = [PatientEvidence(evidence=TOKENS[c], age=40, sex="F") for c in (0, 1, 2, 0, 1)]


def test_run_explain_writes_report_figures_and_mlflow(tmp_path: Path, monkeypatch) -> None:
    import beaconai.explain.run as run

    cfg = _cfg(tmp_path, CONDITIONS)
    monkeypatch.setattr(run, "build_feature_space", lambda _cfg: _space())
    summary = run.run_explain(cfg, patients=PATIENTS)
    reports = tmp_path / "reports"
    assert (reports / "explainability.md").is_file()
    assert json.loads((reports / "explainability_summary.json").read_text()) == summary
    assert summary["latency_ms"]["n"] == len(PATIENTS)
    assert set(summary["global"]) == set(CONDITIONS)
    for name in CONDITIONS:
        assert (reports / summary["images"][name]).is_file()
    assert summary["examples"]
    configure_mlflow(cfg.mlflow, experiment="explain-test")
    assert len(mlflow.search_runs(experiment_names=["explain-test"])) == 1


def test_unknown_condition_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    import beaconai.explain.run as run

    cfg = _cfg(tmp_path, ["Influenza", "Measles"])
    monkeypatch.setattr(run, "build_feature_space", lambda _cfg: _space())
    with pytest.raises(ValueError, match="Measles"):
        run.run_explain(cfg, patients=PATIENTS)
```

- [ ] **Step 2: Implement** `report.py` (pure string building, following the sections above) and `run.py`, following the interfaces above and the existing runner patterns (`calibrate.py`: data scope, `_dev` suffix, MLflow run, `allow_nan=False`). Makefile: `explain:  ## Explain the champion: global view, latency, sanity check` runs `uv run python -m beaconai.explain.run`.

- [ ] **Step 3:** `make test`, `make lint`. Commit: `Add the explainability report and make explain`.

---

### Task 7: Real run, human sanity review, docs

- [ ] `make explain` on the dev data. That is enough: global coefficients belong to the champion itself, prevalence from 50,000 training patients is stable, and the report labels the data scope. Read the global tables and flags for all 8 conditions.
- [ ] Write `docs/phase4_sanity_review.md`: for each listed condition, whether the top features are the expected presenting symptoms and risk factors; each flagged feature with a short clinical judgment (expected, plausible, or odd), and for odd ones a likely cause (collinearity, dropout artifact, synthetic-data quirk). Plain, no em dashes. Re-run `make explain` so the report embeds it.
- [ ] Check latency: p95 under 50 ms end to end. If not, profile (encode vs predict vs explain) before changing anything.
- [ ] CHANGELOG entry (key numbers: latency, ranking-change rates, count of flags); README: `make explain` and an example `ChampionExplainer` call; ROADMAP Phase 4 (local) updated to describe the logistic regression version; local phase line set to Phase 5.
- [ ] `make lint && make test`. Commit: `Phase 4: explanations for the calibrated logistic regression champion`.

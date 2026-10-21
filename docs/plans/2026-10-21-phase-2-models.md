# Phase 2: Baseline Models and Evaluation Implementation Plan

> Execute task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train logistic regression, XGBoost, and a small PyTorch MLP on the Phase 1 DDXPlus features, tune each on validation, log every run to MLflow, register the best (by validation top-3 accuracy) as `beaconai-classifier`, evaluate on test exactly once, and auto-generate `reports/model_comparison.md`.

**Architecture:** One abstract `ModelWrapper` (fit, predict_proba, save, load) with three subclasses and a name-to-class registry. A pure grid expander turns `config/base.yaml` model specs into trials. `models/train.py` runs every trial as a nested MLflow run, picks winners, and registers the champion. `eval/metrics.py` holds pure metric functions; `eval/evaluate.py` scores the winners on a split, draws plots, and renders the report. A guard refuses test evaluation on the dev subsample or a second time.

**Tech Stack:** Python 3.11, uv, scikit-learn, XGBoost (native `xgb.train` API), PyTorch (CPU), MLflow (SQLite backend in `mlruns/`), matplotlib, pytest, ruff.

**Spec:** `ROADMAP.md` Phase 2 (local file) plus the Phase 2 kickoff request. Inputs are the Phase 1 outputs described in the docstring of `src/beaconai/data/preprocess.py`.

## Global Constraints

- Python `==3.11.*`; dependencies only through `uv add`.
- All hyperparameters, grids, k values, MLflow names, and report settings live in `config/base.yaml`. No tunable numbers in code.
- Seed everything with `cfg.seed` (default 42).
- Tune on `validate` only. `test` is scored once, at the end, on the full dataset.
- Selection metric: validation `top3_accuracy`. Registered model name: `beaconai-classifier`.
- Use the dev subsample (`data.dev_sample: 50000`) while iterating.
- Synthetic data only. Tests use tiny hand-made arrays, never the real files.
- Docs and report text: plain and direct, no em dashes. Every generated report states that BeaconAI is decision support, not diagnosis, trained on synthetic data.
- No tool attribution or co-author trailers in commits, code comments, or any tracked file.
- Each phase ends with tests passing, `ruff check` clean, a `reports/CHANGELOG.md` entry with key metrics, and a commit.

## Review Focus

1. **A class absent from the training rows** (possible with small subsamples or folds): `predict_proba` must still return all 49 columns in `labels.json` order, with zeros for the missing class. Test in Task 4 (`test_logreg_handles_missing_class`) and Task 5 (`num_class` fixed).
2. **Tied probabilities** (for example a near-uniform output): top-k must be deterministic, lower class id first, so metrics are reproducible. Test in Task 3 (`test_top_k_ties_prefer_lower_class_id`).
3. **A patient with an empty ground-truth differential:** excluded from differential-overlap averages instead of counting as 0 or dividing by zero. Test in Task 3 (`test_ddx_overlap_skips_empty_rows`).
4. **Running test evaluation on the dev subsample, or a second time:** refused with a clear message unless `--force`. Test in Task 8 (`test_guard_*`).
5. **A saved model reloaded in a fresh object:** identical probabilities, including XGBoost's early-stopped iteration and the MLP's architecture. Test in Tasks 4, 5, 6 (`test_*_save_load_round_trip`).

---

## File map

| File | Responsibility |
|---|---|
| `config/base.yaml` | New `mlflow`, `train`, `models`, `eval` sections |
| `src/beaconai/config.py` | Pydantic schemas for the new sections, `project_path` helper |
| `src/beaconai/models/dataset.py` | Load processed split arrays, labels, manifest |
| `src/beaconai/models/seed.py` | `set_global_seed` |
| `src/beaconai/models/base.py` | `ModelWrapper` ABC: fit, predict_proba, predict, save, load |
| `src/beaconai/models/logreg.py` | Multinomial logistic regression wrapper |
| `src/beaconai/models/xgb.py` | XGBoost wrapper |
| `src/beaconai/models/mlp.py` | PyTorch MLP wrapper |
| `src/beaconai/models/registry.py` | `MODEL_REGISTRY`, `build_model` |
| `src/beaconai/models/tuning.py` | `expand_grid`, `flatten_params`, `select_best` |
| `src/beaconai/models/pyfunc.py` | MLflow pyfunc adapter around `ModelWrapper` |
| `src/beaconai/models/train.py` | Grid search, MLflow logging, registration, `train_summary.json` |
| `src/beaconai/eval/metrics.py` | Top-k, macro F1, per-class recall, confusion, differential overlap |
| `src/beaconai/eval/plots.py` | Confusion matrix and per-class recall figures |
| `src/beaconai/eval/report.py` | Render `model_comparison.md` (pure string building) |
| `src/beaconai/eval/evaluate.py` | Score winners on a split, test-once guard, write report |
| `tests/conftest.py` | Toy processed dataset and config factory shared by tests |
| `tests/models/*`, `tests/eval/*` | Tests mirroring the modules above |
| `Makefile`, `pyproject.toml`, `README.md`, `reports/CHANGELOG.md` | Commands, deps, docs |
| `docs/phase2_analysis.md` | Hand-written model choice and confusion analysis (Task 10) |

---

### Task 0: Preconditions (branch, data, dependencies)

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Branch off main**

```bash
git switch -c phase-2-models
```

- [ ] **Step 2: Generate the Phase 1 outputs (dev subsample)**

`data/processed/ddxplus/` does not exist yet; only the raw CSVs are on disk.

Run: `make data`
Expected: `data/processed/ddxplus/` contains `{train,validate,test}_{X.npz,y.npy,ddx.npz,meta.npz}`, `features.json`, `labels.json`, `manifest.json`. `manifest.json` shows `"splits": {"train": 50000, "validate": 50000, "test": 50000}` and `"n_classes": 49`.

- [ ] **Step 3: Add dependencies, CPU-only torch on Linux so CI stays small**

Append to `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Run: `uv add "scikit-learn>=1.5" "xgboost>=2.1" "torch>=2.4" "mlflow>=3.1"`

- [ ] **Step 4: Smoke-check imports (macOS OpenMP)**

Run: `uv run python -c "import xgboost, torch, sklearn, mlflow; print(xgboost.__version__, torch.__version__, mlflow.__version__)"`
Expected: three versions print. If XGBoost fails with a `libomp` error, run `brew install libomp` and retry. If the process aborts with `OMP: Error #15`, stop and report it; do not set `KMP_DUPLICATE_LIB_OK` silently.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add scikit-learn, XGBoost, PyTorch, and MLflow dependencies"
```

---

### Task 1: Config sections for training, tuning, MLflow, and evaluation

**Files:**
- Modify: `config/base.yaml`, `src/beaconai/config.py`
- Test: `tests/test_config_training.py`

**Interfaces:**
- Produces: `ModelSpec(fixed: dict, grid: dict[str, list])`, `ModelsConfig(logreg, xgboost, mlp)`, `TrainConfig(models: list[str], selection_metric: str, latency_calls: int)`, `MlflowConfig(tracking_db: Path, artifact_root: Path, experiment: str, registered_model: str, champion_alias: str)`, `EvalConfig(ks, ddx_ks, output_dir, top_confusions, lowest_recall)`. New optional `Config` fields `mlflow`, `train`, `models`, `eval` (default `None`, so the existing minimal-YAML tests keep passing). `project_path(p: Path) -> Path` resolves relative paths against `PROJECT_ROOT`. `require_training(cfg) -> tuple[MlflowConfig, TrainConfig, ModelsConfig, EvalConfig]` raises `ValueError` if any section is missing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_training.py
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from beaconai.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    Config,
    ModelSpec,
    load_config,
    project_path,
    require_training,
)


def _base_raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_has_training_sections() -> None:
    cfg = load_config()
    mlflow_cfg, train, models, ev = require_training(cfg)
    assert mlflow_cfg.registered_model == "beaconai-classifier"
    assert train.selection_metric == "top3_accuracy"
    assert train.models == ["logreg", "xgboost", "mlp"]
    assert 3 in ev.ks
    assert models.xgboost.grid  # something to tune


def test_require_training_rejects_missing_sections() -> None:
    raw = _base_raw()
    del raw["models"]
    with pytest.raises(ValueError, match="models"):
        require_training(Config.model_validate(raw))


def test_grid_values_must_be_non_empty_lists() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(fixed={}, grid={"C": []})


def test_key_cannot_be_both_fixed_and_tuned() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(fixed={"C": 1.0}, grid={"C": [0.1, 1.0]})


def test_selection_metric_must_be_computed() -> None:
    raw = _base_raw()
    raw["train"]["selection_metric"] = "top7_accuracy"
    with pytest.raises(ValidationError, match="selection_metric"):
        Config.model_validate(raw)


def test_k_values_must_be_positive() -> None:
    raw = _base_raw()
    raw["eval"]["ks"] = [0, 3]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_project_path_resolves_relative_only(tmp_path: Path) -> None:
    assert project_path(Path("reports")) == PROJECT_ROOT / "reports"
    assert project_path(tmp_path) == tmp_path
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config_training.py -v`
Expected: FAIL, `ImportError: cannot import name 'ModelSpec'`.

- [ ] **Step 3: Add the YAML sections**

Append to `config/base.yaml`:

```yaml
mlflow:
  # Local SQLite backend (supports the model registry) and artifact store, both gitignored.
  tracking_db: mlruns/mlflow.db
  artifact_root: mlruns/artifacts
  experiment: beaconai-phase2
  registered_model: beaconai-classifier
  champion_alias: champion

train:
  models: [logreg, xgboost, mlp]
  # Picks the winner per model type and the registered champion. Validation split only.
  selection_metric: top3_accuracy
  # Single-patient predict_proba calls timed per model (median reported).
  latency_calls: 100

# Each model: `fixed` params are used as-is, `grid` params are searched (full product).
models:
  logreg:
    fixed: {max_iter: 500, solver: lbfgs, tol: 1.0e-4}
    grid:
      C: [0.1, 1.0, 10.0]
  xgboost:
    fixed:
      n_estimators: 400
      early_stopping_rounds: 20
      tree_method: hist
      subsample: 0.8
      colsample_bytree: 0.8
    grid:
      max_depth: [4, 6]
      learning_rate: [0.1, 0.3]
  mlp:
    fixed: {epochs: 30, batch_size: 512, patience: 3, weight_decay: 1.0e-5}
    grid:
      hidden_sizes: [[256], [512, 256]]
      dropout: [0.2, 0.4]
      lr: [1.0e-3]

eval:
  ks: [1, 3, 5]
  # k values for overlap between our top-k and the DDXPlus ground-truth differential.
  ddx_ks: [3, 5]
  output_dir: reports
  top_confusions: 15
  lowest_recall: 10
```

- [ ] **Step 4: Add the schemas**

In `src/beaconai/config.py`, change the pydantic import to
`from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator`, add `from typing import Any, Literal` (keep `Any`), and insert above `class Config`:

```python
ModelName = Literal["logreg", "xgboost", "mlp"]


class ModelSpec(_StrictModel):
    fixed: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_grid(self) -> ModelSpec:
        empty = [k for k, v in self.grid.items() if not v]
        if empty:
            raise ValueError(f"grid values must be non-empty lists: {empty}")
        both = sorted(set(self.fixed) & set(self.grid))
        if both:
            raise ValueError(f"keys are both fixed and tuned: {both}")
        return self


class ModelsConfig(_StrictModel):
    logreg: ModelSpec
    xgboost: ModelSpec
    mlp: ModelSpec


class TrainConfig(_StrictModel):
    models: list[ModelName] = Field(min_length=1)
    selection_metric: str = "top3_accuracy"
    latency_calls: int = Field(default=100, gt=0)


class MlflowConfig(_StrictModel):
    tracking_db: Path
    artifact_root: Path
    experiment: str
    registered_model: str
    champion_alias: str = "champion"


class EvalConfig(_StrictModel):
    ks: list[int] = Field(min_length=1)
    ddx_ks: list[int] = Field(default_factory=list)
    output_dir: Path = Path("reports")
    top_confusions: int = Field(default=15, gt=0)
    lowest_recall: int = Field(default=10, gt=0)

    @field_validator("ks", "ddx_ks")
    @classmethod
    def _positive(cls, ks: list[int]) -> list[int]:
        if any(k < 1 for k in ks):
            raise ValueError("k values must be >= 1")
        return ks
```

Replace `class Config` with:

```python
class Config(_StrictModel):
    seed: int = Field(default=42, ge=0)
    data: DataConfig
    eda: EdaConfig = EdaConfig()
    mlflow: MlflowConfig | None = None
    train: TrainConfig | None = None
    models: ModelsConfig | None = None
    eval: EvalConfig | None = None

    @model_validator(mode="after")
    def _selection_metric_is_computed(self) -> Config:
        if self.train is None or self.eval is None:
            return self
        known = {f"top{k}_accuracy" for k in self.eval.ks} | {"macro_f1"}
        if self.train.selection_metric not in known:
            raise ValueError(
                f"train.selection_metric {self.train.selection_metric!r} is not computed; "
                f"choose one of {sorted(known)}"
            )
        return self


def project_path(path: Path) -> Path:
    """Resolve a config path: relative paths are relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_training(
    cfg: Config,
) -> tuple[MlflowConfig, TrainConfig, ModelsConfig, EvalConfig]:
    """Return the training sections, failing clearly if the config lacks any."""
    missing = [n for n in ("mlflow", "train", "models", "eval") if getattr(cfg, n) is None]
    if missing:
        raise ValueError(f"config is missing sections needed for training: {missing}")
    assert cfg.mlflow and cfg.train and cfg.models and cfg.eval
    return cfg.mlflow, cfg.train, cfg.models, cfg.eval
```

- [ ] **Step 5: Run all config tests**

Run: `uv run pytest tests/test_config.py tests/test_config_env.py tests/test_config_training.py -v`
Expected: all PASS (the old minimal-YAML tests still pass because the new sections are optional).

- [ ] **Step 6: Commit**

```bash
git add config/base.yaml src/beaconai/config.py tests/test_config_training.py
git commit -m "Add training, tuning, MLflow, and eval config sections"
```

---

### Task 2: Shared test fixtures and the processed-data loader

**Files:**
- Create: `tests/conftest.py`, `src/beaconai/models/dataset.py`, `src/beaconai/models/seed.py`
- Test: `tests/models/test_dataset.py`

**Interfaces:**
- Produces: `SplitData(X: csr_matrix, y: np.ndarray[int64], ddx: csr_matrix | None)`; `load_split(processed_dir: Path, split: str) -> SplitData`; `load_labels(processed_dir) -> list[dict]`; `load_manifest(processed_dir) -> dict`; `set_global_seed(seed: int) -> None`. Test helpers `write_toy_processed(root: Path) -> Path` (3 classes, 6 features; returns `root / "ddxplus"`) and `make_config(tmp_path, *, dev_sample: int | None) -> Config`.

- [ ] **Step 1: Write the shared fixtures**

```python
# tests/conftest.py
"""Tiny synthetic processed dataset and config factory. No real data is used in tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from beaconai.config import DEFAULT_CONFIG_PATH, Config

TOY_CLASSES = 3
TOY_FEATURES = 6
TOY_ROWS = {"train": 150, "validate": 60, "test": 60}


def write_toy_processed(root: Path, seed: int = 0) -> Path:
    """Write Phase 1 shaped outputs. Class c switches on features 2c and (mostly) 2c+1."""
    out = root / "ddxplus"
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for split, n in TOY_ROWS.items():
        y = np.arange(n) % TOY_CLASSES
        X = (rng.random((n, TOY_FEATURES)) < 0.1).astype(np.float32)
        X[np.arange(n), 2 * y] = 1.0
        X[np.arange(n), 2 * y + 1] = (rng.random(n) < 0.8).astype(np.float32)
        ddx = np.zeros((n, TOY_CLASSES), dtype=np.float32)
        ddx[np.arange(n), y] = 0.6
        ddx[np.arange(n), (y + 1) % TOY_CLASSES] = 0.4
        sparse.save_npz(out / f"{split}_X.npz", sparse.csr_matrix(X))
        np.save(out / f"{split}_y.npy", y.astype(np.int16))
        sparse.save_npz(out / f"{split}_ddx.npz", sparse.csr_matrix(ddx))
    labels = [
        {"index": i, "name": f"Condition {i}", "name_fr": "", "icd10": "", "severity": i + 1}
        for i in range(TOY_CLASSES)
    ]
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    features = [
        {"index": i, "name": f"F_{i}", "label": f"Feature {i}"} for i in range(TOY_FEATURES)
    ]
    (out / "features.json").write_text(json.dumps(features), encoding="utf-8")
    manifest = {"splits": TOY_ROWS, "n_features": TOY_FEATURES, "n_classes": TOY_CLASSES}
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


def make_config(tmp_path: Path, *, dev_sample: int | None) -> Config:
    """Base config pointed at tmp_path, with tiny grids so tests run in seconds."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["dev_sample"] = dev_sample
    raw["data"]["paths"]["processed"] = str(tmp_path / "processed")
    raw["mlflow"]["tracking_db"] = str(tmp_path / "mlruns" / "mlflow.db")
    raw["mlflow"]["artifact_root"] = str(tmp_path / "mlruns" / "artifacts")
    raw["train"]["latency_calls"] = 5
    raw["eval"].update(output_dir=str(tmp_path / "reports"), ks=[1, 3], ddx_ks=[1, 3])
    raw["models"] = {
        "logreg": {"fixed": {"max_iter": 200}, "grid": {"C": [0.1, 1.0]}},
        "xgboost": {
            "fixed": {"n_estimators": 10, "early_stopping_rounds": 3, "tree_method": "hist"},
            "grid": {"max_depth": [2]},
        },
        "mlp": {
            "fixed": {"epochs": 3, "batch_size": 32, "patience": 2, "weight_decay": 0.0},
            "grid": {"hidden_sizes": [[8]], "dropout": [0.0], "lr": [0.01]},
        },
    }
    return Config.model_validate(raw)


@pytest.fixture
def toy_processed(tmp_path: Path) -> Path:
    return write_toy_processed(tmp_path / "processed")
```

- [ ] **Step 2: Write the failing loader test**

```python
# tests/models/test_dataset.py
from pathlib import Path

import numpy as np

from beaconai.models.dataset import load_labels, load_manifest, load_split
from tests.conftest import TOY_CLASSES, TOY_FEATURES, TOY_ROWS


def test_load_split_shapes_and_types(toy_processed: Path) -> None:
    data = load_split(toy_processed, "train")
    assert data.X.shape == (TOY_ROWS["train"], TOY_FEATURES)
    assert data.y.dtype == np.int64
    assert data.ddx is not None and data.ddx.shape == (TOY_ROWS["train"], TOY_CLASSES)
    assert data.n == TOY_ROWS["train"]


def test_load_split_without_ddx(toy_processed: Path) -> None:
    (toy_processed / "test_ddx.npz").unlink()
    assert load_split(toy_processed, "test").ddx is None


def test_labels_and_manifest(toy_processed: Path) -> None:
    assert [lab["name"] for lab in load_labels(toy_processed)][0] == "Condition 0"
    assert load_manifest(toy_processed)["n_classes"] == TOY_CLASSES


def test_missing_split_says_run_make_data(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="make data"):
        load_split(tmp_path, "train")
```

Run: `uv run pytest tests/models/test_dataset.py -v`
Expected: FAIL, `ModuleNotFoundError: beaconai.models.dataset`.

- [ ] **Step 3: Implement**

```python
# src/beaconai/models/dataset.py
"""Load the Phase 1 processed arrays (see beaconai.data.preprocess for the format)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SplitData:
    X: sparse.csr_matrix
    y: np.ndarray
    ddx: sparse.csr_matrix | None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def load_split(processed_dir: Path, split: str) -> SplitData:
    x_path = processed_dir / f"{split}_X.npz"
    if not x_path.is_file():
        raise FileNotFoundError(f"{x_path} not found. Run `make data` first.")
    ddx_path = processed_dir / f"{split}_ddx.npz"
    return SplitData(
        X=sparse.load_npz(x_path).tocsr(),
        y=np.load(processed_dir / f"{split}_y.npy").astype(np.int64),
        ddx=sparse.load_npz(ddx_path).tocsr() if ddx_path.is_file() else None,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(processed_dir: Path) -> list[dict[str, Any]]:
    return _read_json(processed_dir / "labels.json")


def load_manifest(processed_dir: Path) -> dict[str, Any]:
    return _read_json(processed_dir / "manifest.json")
```

```python
# src/beaconai/models/seed.py
"""Seed every random number generator the training code touches."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/models/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py src/beaconai/models/dataset.py src/beaconai/models/seed.py tests/models/test_dataset.py
git commit -m "Add processed-data loader and shared toy dataset fixtures"
```

---

### Task 3: Evaluation metrics

**Files:**
- Create: `src/beaconai/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces:
  - `top_k_indices(proba, k) -> np.ndarray` shape (n, k), descending, ties to lower class id
  - `top_k_accuracy(y_true, proba, k) -> float`
  - `macro_f1(y_true, y_pred) -> float` (sklearn default: average over classes present in `y_true` or `y_pred`)
  - `confusion_matrix(y_true, y_pred, n_classes) -> np.ndarray[int64]` rows = true, cols = predicted
  - `per_class_recall(cm) -> np.ndarray[float]`, `nan` for classes with no support
  - `ddx_overlap_at_k(proba, ddx, k) -> dict[str, float]` keys `precision`, `recall`, `recall_norm`, `n_rows`
  - `top_confusions(cm, n) -> list[tuple[int, int, int]]` (true, pred, count), off-diagonal
  - `EvalResult(metrics: dict[str, float], confusion: np.ndarray, recall: np.ndarray)`
  - `evaluate_predictions(y_true, proba, ddx, ks, ddx_ks) -> EvalResult`. Metric keys: `top{k}_accuracy`, `macro_f1`, `ddx_precision_at{k}`, `ddx_recall_at{k}`, `ddx_recall_norm_at{k}`.

**Differential overlap definition.** For a patient, `D` = classes with non-zero probability in the DDXPlus `DIFFERENTIAL_DIAGNOSIS`, `P` = our top-k classes. `precision@k = |P ∩ D| / k` (how much of what we show is in the reference differential). Raw `recall@k = |P ∩ D| / |D|` (how much of the reference differential we cover; below 1 whenever `|D| > k`). Normalized `recall_norm@k = |P ∩ D| / min(k, |D|)` (1.0 means our k slots are as full of reference conditions as they can be). All three are averaged over patients with non-empty `D`. Normalized recall@3 is the headline differential metric in the report.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_metrics.py
import numpy as np
import pytest
from scipy import sparse

from beaconai.eval.metrics import (
    confusion_matrix,
    ddx_overlap_at_k,
    evaluate_predictions,
    macro_f1,
    per_class_recall,
    top_confusions,
    top_k_accuracy,
    top_k_indices,
)

# 4 patients, 4 classes.
PROBA = np.array(
    [
        [0.70, 0.20, 0.05, 0.05],  # true 0 -> top-1 hit
        [0.10, 0.20, 0.60, 0.10],  # true 1 -> top-2 hit
        [0.40, 0.30, 0.20, 0.10],  # true 3 -> rank 4
        [0.05, 0.05, 0.10, 0.80],  # true 3 -> top-1 hit
    ]
)
Y = np.array([0, 1, 3, 3])


def test_top_k_accuracy() -> None:
    assert top_k_accuracy(Y, PROBA, 1) == pytest.approx(0.5)
    assert top_k_accuracy(Y, PROBA, 2) == pytest.approx(0.75)
    assert top_k_accuracy(Y, PROBA, 4) == pytest.approx(1.0)


def test_top_k_ties_prefer_lower_class_id() -> None:
    uniform = np.full((1, 4), 0.25)
    assert top_k_indices(uniform, 2).tolist() == [[0, 1]]


def test_top_k_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        top_k_accuracy(Y, PROBA, 5)  # k > n_classes
    with pytest.raises(ValueError):
        top_k_accuracy(Y[:3], PROBA, 1)  # length mismatch
    with pytest.raises(ValueError):
        top_k_accuracy(np.array([0, 1, 3, 9]), PROBA, 1)  # label out of range


def test_confusion_and_recall() -> None:
    y_pred = PROBA.argmax(axis=1)  # [0, 2, 0, 3]
    cm = confusion_matrix(Y, y_pred, 4)
    assert cm.shape == (4, 4)
    assert cm[1, 2] == 1 and cm[3, 0] == 1 and cm[3, 3] == 1
    recall = per_class_recall(cm)
    assert recall[0] == 1.0 and recall[1] == 0.0 and recall[3] == 0.5
    assert np.isnan(recall[2])  # no true examples of class 2


def test_macro_f1_ignores_classes_absent_everywhere() -> None:
    # Class 2 and 3 never appear; perfect predictions on 0 and 1 give F1 1.0.
    assert macro_f1(np.array([0, 1, 1]), np.array([0, 1, 1])) == pytest.approx(1.0)


def test_top_confusions_ranks_off_diagonal() -> None:
    cm = np.array([[5, 3, 0], [1, 4, 0], [0, 7, 2]])
    assert top_confusions(cm, 2) == [(2, 1, 7), (0, 1, 3)]
    assert len(top_confusions(cm, 10)) == 3  # zero cells are skipped


def test_ddx_overlap() -> None:
    # Reference differential: patient 0 -> {0, 1}; patient 1 -> {1, 2, 3}.
    ddx = sparse.csr_matrix(
        np.array([[0.5, 0.5, 0, 0], [0, 0.4, 0.3, 0.3], [0, 0, 0, 0], [0, 0, 0, 1.0]])
    )
    out = ddx_overlap_at_k(PROBA[:2], ddx[:2], 2)
    # Top-2: p0 = {0, 1} -> 2 hits; p1 = {2, 1} -> 2 hits.
    assert out["precision"] == pytest.approx((2 / 2 + 2 / 2) / 2)
    assert out["recall"] == pytest.approx((2 / 2 + 2 / 3) / 2)
    assert out["recall_norm"] == pytest.approx((2 / 2 + 2 / 2) / 2)
    assert out["n_rows"] == 2


def test_ddx_overlap_small_reference_differential() -> None:
    # |D| = 1 < k = 3: one hit fills the only slot that can be filled.
    ddx = sparse.csr_matrix(np.array([[1.0, 0, 0, 0]]))
    out = ddx_overlap_at_k(PROBA[:1], ddx, 3)
    assert out["precision"] == pytest.approx(1 / 3)
    assert out["recall"] == pytest.approx(1.0)
    assert out["recall_norm"] == pytest.approx(1.0)


def test_ddx_overlap_ignores_stored_zeros() -> None:
    # An explicitly stored 0.0 is not part of the reference differential.
    ddx = sparse.csr_matrix(
        (np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 2])), shape=(1, 4)
    )
    out = ddx_overlap_at_k(PROBA[:1], ddx, 2)
    assert out["recall"] == pytest.approx(1.0)  # D = {0}, not {0, 1}
    assert out["precision"] == pytest.approx(1 / 2)


def test_ddx_overlap_skips_empty_rows() -> None:
    ddx = sparse.csr_matrix(np.array([[1.0, 0, 0, 0], [0, 0, 0, 0]]))
    out = ddx_overlap_at_k(PROBA[:2], ddx, 1)
    assert out["n_rows"] == 1
    assert out["precision"] == pytest.approx(1.0)


def test_ddx_overlap_all_empty_is_nan() -> None:
    out = ddx_overlap_at_k(PROBA[:1], sparse.csr_matrix((1, 4)), 1)
    assert out["n_rows"] == 0 and np.isnan(out["precision"])


def test_evaluate_predictions_keys() -> None:
    ddx = sparse.csr_matrix(np.eye(4)[Y])
    res = evaluate_predictions(Y, PROBA, ddx, ks=[1, 3], ddx_ks=[3])
    assert set(res.metrics) == {
        "top1_accuracy",
        "top3_accuracy",
        "macro_f1",
        "ddx_precision_at3",
        "ddx_recall_at3",
        "ddx_recall_norm_at3",
    }
    assert res.confusion.shape == (4, 4)


def test_evaluate_predictions_without_ddx() -> None:
    res = evaluate_predictions(Y, PROBA, None, ks=[1], ddx_ks=[3])
    assert "ddx_precision_at3" not in res.metrics
```

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement everything except `ddx_overlap_at_k`**

```python
# src/beaconai/eval/metrics.py
"""Classifier metrics. Inputs: integer class ids and an (n_patients, n_classes) probability matrix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import f1_score


def _check(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {p.shape}")
    if len(y) != p.shape[0]:
        raise ValueError(f"{len(y)} labels but {p.shape[0]} probability rows")
    if len(y) == 0:
        raise ValueError("no rows to evaluate")
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError(f"labels must be in [0, {p.shape[1]})")
    return y, p


def top_k_indices(proba: np.ndarray, k: int) -> np.ndarray:
    """Top-k class ids per row, highest first. Stable sort: ties go to the lower class id."""
    p = np.asarray(proba, dtype=np.float64)
    if not 1 <= k <= p.shape[1]:
        raise ValueError(f"k must be in [1, {p.shape[1]}], got {k}")
    return np.argsort(-p, axis=1, kind="stable")[:, :k]


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    y, p = _check(y_true, proba)
    return float((top_k_indices(p, k) == y[:, None]).any(axis=1).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over classes present in y_true or y_pred."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return _sk_confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(np.int64)


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)


def top_confusions(cm: np.ndarray, n: int) -> list[tuple[int, int, int]]:
    """The n largest off-diagonal cells as (true, predicted, count). Zero cells skipped."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows, cols = np.nonzero(off)
    cells = sorted(zip(rows, cols, off[rows, cols], strict=True), key=lambda c: (-c[2], c[0], c[1]))
    return [(int(r), int(c), int(v)) for r, c, v in cells[:n]]


def ddx_overlap_at_k(proba: np.ndarray, ddx: sparse.csr_matrix, k: int) -> dict[str, float]:
    """Overlap between our top-k and the DDXPlus ground-truth differential.

    D = classes with non-zero reference probability, P = our top-k.
    precision = |P & D| / k, recall = |P & D| / |D|, averaged over rows with non-empty D.
    Returns {"precision", "recall", "n_rows"}; precision and recall are nan if no row has a D.
    """
    raise NotImplementedError  # written in Step 3


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    confusion: np.ndarray
    recall: np.ndarray


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    ddx: sparse.csr_matrix | None,
    ks: Sequence[int],
    ddx_ks: Sequence[int],
) -> EvalResult:
    y, p = _check(y_true, proba)
    y_pred = p.argmax(axis=1)
    metrics = {f"top{k}_accuracy": top_k_accuracy(y, p, k) for k in ks}
    metrics["macro_f1"] = macro_f1(y, y_pred)
    if ddx is not None:
        for k in ddx_ks:
            overlap = ddx_overlap_at_k(p, ddx, k)
            metrics[f"ddx_precision_at{k}"] = overlap["precision"]
            metrics[f"ddx_recall_at{k}"] = overlap["recall"]
            metrics[f"ddx_recall_norm_at{k}"] = overlap["recall_norm"]
    cm = confusion_matrix(y, y_pred, p.shape[1])
    return EvalResult(metrics=metrics, confusion=cm, recall=per_class_recall(cm))
```

- [ ] **Step 3: Implement `ddx_overlap_at_k`**

It must satisfy all `test_ddx_overlap*` tests. Hints: `top_k_indices(proba, k)` gives P; `ddx[i].indices` gives D for row i on a CSR matrix (check `ddx.data > 0` if explicit zeros could be stored; call `ddx.eliminate_zeros()` on a copy first); skip rows where D is empty.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/beaconai/eval/metrics.py tests/eval/test_metrics.py
git commit -m "Add top-k, macro F1, per-class recall, confusion, and differential overlap metrics"
```

---

### Task 4: Shared model wrapper and logistic regression

**Files:**
- Create: `src/beaconai/models/base.py`, `src/beaconai/models/logreg.py`, `src/beaconai/models/registry.py`
- Test: `tests/models/test_logreg.py`, `tests/models/test_base.py`

**Interfaces:**
- Consumes: `SplitData`, `load_split` (Task 2).
- Produces:
  - `ModelWrapper(params: dict, n_classes: int, seed: int)`; class attribute `name: str`
  - `.fit(X, y, X_val=None, y_val=None) -> self`
  - `.predict_proba(X) -> np.ndarray` shape (n, n_classes), float64
  - `.predict(X) -> np.ndarray`
  - `.save(path: Path) -> None` (directory with `meta.json` plus model files)
  - `ModelWrapper.load(path: Path) -> ModelWrapper` (static; dispatches on `meta.json["name"]`)
  - `MODEL_REGISTRY: dict[str, type[ModelWrapper]]`, `build_model(name, params, n_classes, seed) -> ModelWrapper`
  - Subclasses implement `_fit`, `_predict_proba`, `_save`, `_load`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_base.py
from pathlib import Path

import numpy as np
import pytest

from beaconai.models.dataset import load_split
from beaconai.models.registry import MODEL_REGISTRY, build_model


def test_registry_names() -> None:
    assert set(MODEL_REGISTRY) >= {"logreg"}


def test_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("svm", {}, 3, 0)


def test_predict_before_fit_raises(toy_processed: Path) -> None:
    model = build_model("logreg", {}, 3, 0)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(load_split(toy_processed, "test").X)


def test_feature_count_mismatch_raises(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("logreg", {}, 3, 0).fit(train.X, train.y)
    with pytest.raises(ValueError, match="features"):
        model.predict_proba(train.X[:, :4])


def test_label_out_of_range_raises(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    with pytest.raises(ValueError, match="labels"):
        build_model("logreg", {}, 2, 0).fit(train.X, train.y)  # toy has 3 classes


def test_val_x_without_y_raises(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    with pytest.raises(ValueError, match="together"):
        build_model("logreg", {}, 3, 0).fit(train.X, train.y, X_val=train.X)


def test_load_unknown_name_raises(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text('{"name": "svm"}', encoding="utf-8")
    from beaconai.models.base import ModelWrapper

    with pytest.raises(ValueError, match="Unknown model"):
        ModelWrapper.load(tmp_path)


def test_predict_is_argmax(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("logreg", {}, 3, 0).fit(train.X, train.y)
    np.testing.assert_array_equal(model.predict(train.X), model.predict_proba(train.X).argmax(1))
```

```python
# tests/models/test_logreg.py
from pathlib import Path

import numpy as np

from beaconai.eval.metrics import top_k_accuracy
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import build_model


def _fit(toy_processed: Path):
    train = load_split(toy_processed, "train")
    return build_model("logreg", {"C": 1.0, "max_iter": 200}, 3, 42).fit(train.X, train.y)


def test_logreg_learns_toy_task(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    test = load_split(toy_processed, "test")
    proba = model.predict_proba(test.X)
    assert proba.shape == (test.n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert top_k_accuracy(test.y, proba, 1) > 0.9


def test_logreg_handles_missing_class(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    keep = train.y != 2
    model = build_model("logreg", {}, 3, 42).fit(train.X[keep], train.y[keep])
    proba = model.predict_proba(train.X)
    assert proba.shape[1] == 3
    assert np.all(proba[:, 2] == 0.0)


def test_logreg_save_load_round_trip(toy_processed: Path, tmp_path: Path) -> None:
    model = _fit(toy_processed)
    model.save(tmp_path / "m")
    loaded = ModelWrapper.load(tmp_path / "m")
    X = load_split(toy_processed, "test").X
    assert loaded.name == "logreg" and loaded.params == model.params
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
```

Run: `uv run pytest tests/models/test_base.py tests/models/test_logreg.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement the base class**

```python
# src/beaconai/models/base.py
"""One interface for every classifier: fit, predict_proba, save, load."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
from scipy import sparse

Matrix = sparse.csr_matrix | np.ndarray
META_FILE = "meta.json"


def _as_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(np.int64).ravel()
    if len(y) and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes})")
    return y


class ModelWrapper(ABC):
    name: ClassVar[str]

    def __init__(self, params: dict[str, Any], n_classes: int, seed: int) -> None:
        self.params = dict(params)
        self.n_classes = n_classes
        self.seed = seed
        self.n_features: int | None = None

    def fit(
        self,
        X: Matrix,
        y: np.ndarray,
        X_val: Matrix | None = None,
        y_val: np.ndarray | None = None,
    ) -> Self:
        """Train. X_val/y_val are used only for early stopping, never for the fit itself."""
        y = _as_labels(y, self.n_classes)
        if X.shape[0] != len(y):
            raise ValueError(f"{X.shape[0]} rows but {len(y)} labels")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be given together")
        yv = None if y_val is None else _as_labels(y_val, self.n_classes)
        self.n_features = int(X.shape[1])
        self._fit(X, y, X_val, yv)
        return self

    def predict_proba(self, X: Matrix) -> np.ndarray:
        if self.n_features is None:
            raise RuntimeError(f"{self.name} model is not fitted")
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return np.asarray(self._predict_proba(X), dtype=np.float64)

    def predict(self, X: Matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        if self.n_features is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "params": self.params,
            "n_classes": self.n_classes,
            "seed": self.seed,
            "n_features": self.n_features,
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._save(path)

    @staticmethod
    def load(path: Path) -> ModelWrapper:
        """Load a model saved by `save`. Files may be pickles: only load trusted directories."""
        from beaconai.models.registry import MODEL_REGISTRY

        path = Path(path)
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        if meta.get("name") not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name in {path / META_FILE}: {meta.get('name')!r}")
        model = MODEL_REGISTRY[meta["name"]](meta["params"], meta["n_classes"], meta["seed"])
        model.n_features = meta["n_features"]
        model._load(path)
        return model

    @abstractmethod
    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: Matrix) -> np.ndarray: ...

    @abstractmethod
    def _save(self, path: Path) -> None: ...

    @abstractmethod
    def _load(self, path: Path) -> None: ...
```

- [ ] **Step 3: Implement logistic regression and the registry**

```python
# src/beaconai/models/logreg.py
"""Multinomial logistic regression baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from beaconai.models.base import Matrix, ModelWrapper

MODEL_FILE = "model.joblib"


class LogRegModel(ModelWrapper):
    name = "logreg"

    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None:
        # No early stopping, so the validation split is unused here; it is used by the grid search.
        self._clf = LogisticRegression(random_state=self.seed, **self.params).fit(X, y)

    def _predict_proba(self, X: Matrix) -> np.ndarray:
        # Classes missing from the training rows get probability 0 in their fixed column.
        out = np.zeros((X.shape[0], self.n_classes))
        out[:, self._clf.classes_] = self._clf.predict_proba(X)
        return out

    def _save(self, path: Path) -> None:
        joblib.dump(self._clf, path / MODEL_FILE)

    def _load(self, path: Path) -> None:
        # joblib is pickle: only files this project wrote (local MLflow store) are loaded.
        self._clf = joblib.load(path / MODEL_FILE)
```

```python
# src/beaconai/models/registry.py
"""Model name -> wrapper class."""

from __future__ import annotations

from typing import Any

from beaconai.models.base import ModelWrapper
from beaconai.models.logreg import LogRegModel

MODEL_REGISTRY: dict[str, type[ModelWrapper]] = {
    LogRegModel.name: LogRegModel,
}


def build_model(name: str, params: dict[str, Any], n_classes: int, seed: int) -> ModelWrapper:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](params, n_classes, seed)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/models -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/beaconai/models/base.py src/beaconai/models/logreg.py src/beaconai/models/registry.py tests/models/test_base.py tests/models/test_logreg.py
git commit -m "Add shared model wrapper and logistic regression baseline"
```

---

### Task 5: XGBoost wrapper

**Files:**
- Create: `src/beaconai/models/xgb.py`
- Modify: `src/beaconai/models/registry.py`
- Test: `tests/models/test_xgb.py`

**Interfaces:**
- Consumes: `ModelWrapper` (Task 4).
- Produces: `XGBoostModel` with `name = "xgboost"`. Params: `n_estimators` and optional `early_stopping_rounds` are popped; everything else goes to `xgb.train` as booster params. Attribute `best_iteration: int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_xgb.py
from pathlib import Path

import numpy as np

from beaconai.eval.metrics import top_k_accuracy
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import build_model

PARAMS = {"n_estimators": 50, "early_stopping_rounds": 3, "max_depth": 2, "tree_method": "hist"}


def _fit(toy_processed: Path):
    train, val = load_split(toy_processed, "train"), load_split(toy_processed, "validate")
    return build_model("xgboost", PARAMS, 3, 42).fit(train.X, train.y, val.X, val.y)


def test_xgb_learns_and_early_stops(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    test = load_split(toy_processed, "test")
    proba = model.predict_proba(test.X)
    assert proba.shape == (test.n, 3)
    assert top_k_accuracy(test.y, proba, 1) > 0.9
    # A very high learning rate makes validation loss plateau fast, so early stopping must fire.
    train, val = load_split(toy_processed, "train"), load_split(toy_processed, "validate")
    fast = {**PARAMS, "n_estimators": 200, "learning_rate": 1.0}
    stopped = build_model("xgboost", fast, 3, 42).fit(train.X, train.y, val.X, val.y)
    assert stopped.best_iteration < 199


def test_xgb_without_validation_uses_all_rounds(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("xgboost", PARAMS, 3, 42).fit(train.X, train.y)
    assert model.best_iteration == PARAMS["n_estimators"] - 1


def test_xgb_handles_missing_class(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    keep = train.y != 2
    model = build_model("xgboost", PARAMS, 3, 42).fit(train.X[keep], train.y[keep])
    assert model.predict_proba(train.X).shape[1] == 3


def test_xgb_save_load_round_trip(toy_processed: Path, tmp_path: Path) -> None:
    model = _fit(toy_processed)
    model.save(tmp_path / "m")
    loaded = ModelWrapper.load(tmp_path / "m")
    X = load_split(toy_processed, "test").X
    assert loaded.best_iteration == model.best_iteration
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)
```

Run: `uv run pytest tests/models/test_xgb.py -v`
Expected: FAIL, `ValueError: Unknown model 'xgboost'`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/models/xgb.py
"""XGBoost multi-class model (native API, so a class missing from training is fine)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from beaconai.models.base import Matrix, ModelWrapper

MODEL_FILE = "model.ubj"
EXTRA_FILE = "xgb.json"


class XGBoostModel(ModelWrapper):
    name = "xgboost"

    def _booster_params(self) -> tuple[dict, int, int | None]:
        p = dict(self.params)
        n_rounds = int(p.pop("n_estimators"))
        early = p.pop("early_stopping_rounds", None)
        booster = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "eval_metric": "mlogloss",
            "seed": self.seed,
            **p,
        }
        return booster, n_rounds, early

    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None:
        params, n_rounds, early = self._booster_params()
        evals = [] if X_val is None else [(xgb.DMatrix(X_val, label=y_val), "validate")]
        use_early = bool(evals) and early is not None
        self._booster = xgb.train(
            params,
            xgb.DMatrix(X, label=y),
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early if use_early else None,
            verbose_eval=False,
        )
        self.best_iteration = int(self._booster.best_iteration) if use_early else n_rounds - 1

    def _predict_proba(self, X: Matrix) -> np.ndarray:
        return self._booster.predict(xgb.DMatrix(X), iteration_range=(0, self.best_iteration + 1))

    def _save(self, path: Path) -> None:
        self._booster.save_model(path / MODEL_FILE)
        (path / EXTRA_FILE).write_text(
            json.dumps({"best_iteration": self.best_iteration}), encoding="utf-8"
        )

    def _load(self, path: Path) -> None:
        self._booster = xgb.Booster()
        self._booster.load_model(path / MODEL_FILE)
        extra = json.loads((path / EXTRA_FILE).read_text(encoding="utf-8"))
        self.best_iteration = int(extra["best_iteration"])
```

In `registry.py`, add `from beaconai.models.xgb import XGBoostModel` and the entry `XGBoostModel.name: XGBoostModel,`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/models -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/beaconai/models/xgb.py src/beaconai/models/registry.py tests/models/test_xgb.py
git commit -m "Add XGBoost model wrapper with early stopping on validation"
```

---

### Task 6: PyTorch MLP wrapper

**Files:**
- Create: `src/beaconai/models/mlp.py`
- Modify: `src/beaconai/models/registry.py`
- Test: `tests/models/test_mlp.py`

**Interfaces:**
- Consumes: `ModelWrapper` (Task 4).
- Produces: `MLPModel` with `name = "mlp"`. Params: `hidden_sizes: list[int]`, `dropout`, `lr`, `weight_decay`, `epochs`, `batch_size`, `patience`. Attribute `epochs_trained: int`. CPU only. Early stopping on validation loss restores the best weights.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_mlp.py
from pathlib import Path

import numpy as np

from beaconai.eval.metrics import top_k_accuracy
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import build_model

PARAMS = {
    "hidden_sizes": [16],
    "dropout": 0.0,
    "lr": 0.01,
    "weight_decay": 0.0,
    "epochs": 40,
    "batch_size": 32,
    "patience": 3,
}


def _fit(toy_processed: Path, seed: int = 42):
    train, val = load_split(toy_processed, "train"), load_split(toy_processed, "validate")
    return build_model("mlp", PARAMS, 3, seed).fit(train.X, train.y, val.X, val.y)


def test_mlp_learns_toy_task(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    test = load_split(toy_processed, "test")
    proba = model.predict_proba(test.X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert top_k_accuracy(test.y, proba, 1) > 0.9


def test_mlp_is_reproducible(toy_processed: Path) -> None:
    X = load_split(toy_processed, "test").X
    np.testing.assert_allclose(
        _fit(toy_processed).predict_proba(X), _fit(toy_processed).predict_proba(X), rtol=1e-6
    )


def test_mlp_early_stops(toy_processed: Path) -> None:
    assert _fit(toy_processed).epochs_trained <= PARAMS["epochs"]


def test_mlp_save_load_round_trip(toy_processed: Path, tmp_path: Path) -> None:
    model = _fit(toy_processed)
    model.save(tmp_path / "m")
    loaded = ModelWrapper.load(tmp_path / "m")
    X = load_split(toy_processed, "test").X
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)
```

Run: `uv run pytest tests/models/test_mlp.py -v`
Expected: FAIL, `ValueError: Unknown model 'mlp'`.

- [ ] **Step 2: Implement**

```python
# src/beaconai/models/mlp.py
"""Small feed-forward network (comparison model). Trains on CPU, mini-batches densified on the fly."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn

from beaconai.models.base import Matrix, ModelWrapper

MODEL_FILE = "model.pt"


def _build_net(
    n_features: int, hidden_sizes: list[int], dropout: float, n_classes: int
) -> nn.Module:
    layers: list[nn.Module] = []
    width = n_features
    for h in hidden_sizes:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


def _dense(X: Matrix) -> torch.Tensor:
    arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


class MLPModel(ModelWrapper):
    name = "mlp"

    def _new_net(self) -> nn.Module:
        assert self.n_features is not None
        return _build_net(
            self.n_features,
            list(self.params["hidden_sizes"]),
            float(self.params["dropout"]),
            self.n_classes,
        )

    def _logits(self, net: nn.Module, X: Matrix) -> torch.Tensor:
        bs = int(self.params["batch_size"])
        net.eval()
        with torch.no_grad():
            return torch.cat([net(_dense(X[i : i + bs])) for i in range(0, X.shape[0], bs)])

    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = self._new_net()
        opt = torch.optim.Adam(
            net.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_fn = nn.CrossEntropyLoss()
        bs, patience = int(self.params["batch_size"]), int(self.params["patience"])
        best_loss, best_state, bad = math.inf, None, 0
        y_t = torch.from_numpy(y)

        epoch = 0
        for epoch in range(1, int(self.params["epochs"]) + 1):
            net.train()
            order = rng.permutation(X.shape[0])
            for start in range(0, len(order), bs):
                idx = order[start : start + bs]
                opt.zero_grad()
                loss_fn(net(_dense(X[idx])), y_t[idx]).backward()
                opt.step()
            if X_val is None:
                continue
            val_loss = float(loss_fn(self._logits(net, X_val), torch.from_numpy(y_val)))
            if val_loss < best_loss:
                best_loss, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net.eval()
        self.epochs_trained = epoch

    def _predict_proba(self, X: Matrix) -> np.ndarray:
        return torch.softmax(self._logits(self._net, X), dim=1).numpy()

    def _save(self, path: Path) -> None:
        torch.save(self._net.state_dict(), path / MODEL_FILE)

    def _load(self, path: Path) -> None:
        net = self._new_net()
        net.load_state_dict(torch.load(path / MODEL_FILE, weights_only=True))
        self._net = net.eval()
```

In `registry.py`, add `from beaconai.models.mlp import MLPModel` and the entry `MLPModel.name: MLPModel,`. Update `test_registry_names` in `tests/models/test_base.py` to `assert set(MODEL_REGISTRY) == {"logreg", "xgboost", "mlp"}`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/models -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/beaconai/models/mlp.py src/beaconai/models/registry.py tests/models/test_mlp.py tests/models/test_base.py
git commit -m "Add PyTorch MLP model wrapper with early stopping"
```

---

### Task 7: Grid search, MLflow logging, and model registration

**Files:**
- Create: `src/beaconai/models/tuning.py`, `src/beaconai/models/pyfunc.py`, `src/beaconai/models/train.py`, `src/beaconai/eval/plots.py`
- Modify: `Makefile`
- Test: `tests/models/test_tuning.py`, `tests/models/test_train.py`, `tests/eval/test_plots.py`

**Interfaces:**
- Consumes: `require_training`, `project_path` (Task 1); `load_split`, `load_labels`, `load_manifest`, `set_global_seed` (Task 2); `evaluate_predictions` (Task 3); `build_model` (Tasks 4 to 6).
- Produces:
  - `expand_grid(fixed: dict, grid: dict[str, list]) -> list[dict]` (keys sorted, deterministic order)
  - `flatten_params(params: dict) -> dict[str, str | int | float]`
  - `TrialResult(model, params, run_id, model_uri, val_metrics, fit_seconds, latency_ms)`
  - `select_best(results: list[TrialResult], metric: str) -> TrialResult` (ties keep the earlier trial)
  - `BeaconPyfunc` (mlflow.pyfunc.PythonModel) with attribute `.model: ModelWrapper` after load
  - `configure_mlflow(cfg: MlflowConfig) -> None`
  - `load_wrapper(model_uri: str) -> ModelWrapper`
  - `run_training(cfg: Config, *, winners_only: bool = False) -> dict` returning and writing `<eval.output_dir>/train_summary.json`:
    `{"parent_run_id", "dev_sample", "mode", "selection_metric", "champion", "registered": {"name", "version"}, "models": {name: {"run_id", "model_uri", "params", "val_metrics", "fit_seconds", "latency_ms", "n_trials"}}}`
  - `plot_confusion(cm, class_names, path, title) -> Path`, `plot_per_class_recall(recall, class_names, path, title) -> Path`

- [ ] **Step 1: Write the failing tuning tests**

```python
# tests/models/test_tuning.py
import pytest

from beaconai.models.tuning import TrialResult, expand_grid, flatten_params, select_best


def test_expand_grid_product_and_order() -> None:
    trials = expand_grid({"a": 1}, {"z": [1, 2], "b": ["x", "y"]})
    assert trials == [
        {"a": 1, "b": "x", "z": 1},
        {"a": 1, "b": "x", "z": 2},
        {"a": 1, "b": "y", "z": 1},
        {"a": 1, "b": "y", "z": 2},
    ]


def test_expand_grid_without_grid_is_one_trial() -> None:
    assert expand_grid({"a": 1}, {}) == [{"a": 1}]


def test_flatten_params_stringifies_lists() -> None:
    assert flatten_params({"h": [256, 128], "lr": 0.1}) == {"h": "[256, 128]", "lr": 0.1}


def _r(name: str, top3: float) -> TrialResult:
    return TrialResult(name, {}, f"run-{name}", "uri", {"top3_accuracy": top3}, 1.0, 1.0)


def test_select_best_and_ties() -> None:
    assert select_best([_r("a", 0.5), _r("b", 0.9), _r("c", 0.9)], "top3_accuracy").model == "b"


def test_select_best_rejects_empty_and_unknown_metric() -> None:
    with pytest.raises(ValueError):
        select_best([], "top3_accuracy")
    with pytest.raises(KeyError):
        select_best([_r("a", 0.5)], "auc")
```

Run: `uv run pytest tests/models/test_tuning.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement tuning helpers**

```python
# src/beaconai/models/tuning.py
"""Grid expansion and winner selection. Pure functions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class TrialResult:
    model: str
    params: dict[str, Any]
    run_id: str
    model_uri: str
    val_metrics: dict[str, float]
    fit_seconds: float
    latency_ms: float


def expand_grid(fixed: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of grid values, each merged with the fixed params."""
    keys = sorted(grid)
    return [
        {**fixed, **dict(zip(keys, combo, strict=True))}
        for combo in product(*(grid[k] for k in keys))
    ]


def flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """MLflow params must be scalars; lists and dicts become JSON strings."""
    return {k: json.dumps(v) if isinstance(v, list | dict) else v for k, v in params.items()}


def select_best(results: list[TrialResult], metric: str) -> TrialResult:
    """Highest validation metric wins; on a tie the earlier trial is kept."""
    if not results:
        raise ValueError("no trials to select from")
    if metric not in results[0].val_metrics:
        raise KeyError(f"metric {metric!r} not in {sorted(results[0].val_metrics)}")
    best = results[0]
    for r in results[1:]:
        if r.val_metrics[metric] > best.val_metrics[metric]:
            best = r
    return best
```

Run: `uv run pytest tests/models/test_tuning.py -v` → PASS.

- [ ] **Step 3: Write plot tests and implement plots**

```python
# tests/eval/test_plots.py
from pathlib import Path

import numpy as np

from beaconai.eval.plots import plot_confusion, plot_per_class_recall


def test_plots_write_png(tmp_path: Path) -> None:
    cm = np.array([[3, 1, 0], [0, 4, 0], [0, 0, 0]])
    names = ["A", "B", "C"]
    p1 = plot_confusion(cm, names, tmp_path / "cm.png", "Confusion")
    p2 = plot_per_class_recall(np.array([0.75, 1.0, np.nan]), names, tmp_path / "r.png", "Recall")
    assert p1.stat().st_size > 0 and p2.stat().st_size > 0
```

```python
# src/beaconai/eval/plots.py
"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> Path:
    """Row-normalised confusion matrix: each row shows where that true condition went."""
    support = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, support, out=np.zeros(cm.shape, dtype=float), where=support > 0)
    size = max(6.0, 0.28 * len(class_names))
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=90, fontsize=6)
    ax.set_yticks(ticks, class_names, fontsize=6)
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true condition")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_class_recall(
    recall: np.ndarray, class_names: list[str], path: Path, title: str
) -> Path:
    order = np.argsort(np.nan_to_num(recall, nan=-1.0))
    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.2 * len(class_names))))
    ax.barh(np.arange(len(order)), np.nan_to_num(recall[order]), color="#4c72b0")
    ax.set_yticks(np.arange(len(order)), [class_names[i] for i in order], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
```

Run: `uv run pytest tests/eval/test_plots.py -v` → PASS.

- [ ] **Step 4: Write the failing training integration test**

```python
# tests/models/test_train.py
import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from beaconai.models.dataset import load_split
from beaconai.models.train import configure_mlflow, load_wrapper, run_training
from tests.conftest import make_config, write_toy_processed


def test_run_training_logs_registers_and_summarises(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=50000)
    processed = write_toy_processed(tmp_path / "processed")
    summary = run_training(cfg)

    # 2 logreg + 1 xgboost + 1 mlp trials, each its own nested run.
    assert {m: v["n_trials"] for m, v in summary["models"].items()} == {
        "logreg": 2,
        "xgboost": 1,
        "mlp": 1,
    }
    assert summary["mode"] == "grid" and summary["dev_sample"] == 50000
    assert summary["champion"] in summary["models"]
    written = json.loads((tmp_path / "reports" / "train_summary.json").read_text())
    assert written == summary

    configure_mlflow(cfg.mlflow)
    client = MlflowClient()
    mv = client.get_model_version_by_alias("beaconai-classifier", "champion")
    assert mv.version == str(summary["registered"]["version"])
    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) == 5  # parent + 4 trials
    child = runs[runs["tags.model_type"] == "logreg"].iloc[0]
    assert "metrics.val_top3_accuracy" in child

    champ = summary["models"][summary["champion"]]
    model = load_wrapper(champ["model_uri"])
    test = load_split(processed, "test")
    assert model.predict_proba(test.X).shape == (test.n, 3)


def test_winners_only_reuses_best_params(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=50000)
    write_toy_processed(tmp_path / "processed")
    first = run_training(cfg)
    second = run_training(cfg, winners_only=True)
    assert second["mode"] == "winners"
    for name, info in second["models"].items():
        assert info["n_trials"] == 1
        assert info["params"] == first["models"][name]["params"]
    assert second["registered"]["version"] == first["registered"]["version"] + 1


def test_training_is_reproducible(tmp_path: Path) -> None:
    a = make_config(tmp_path / "a", dev_sample=None)
    b = make_config(tmp_path / "b", dev_sample=None)
    write_toy_processed(tmp_path / "a" / "processed")
    write_toy_processed(tmp_path / "b" / "processed")
    ma, mb = run_training(a)["models"], run_training(b)["models"]
    for name in ma:
        np.testing.assert_allclose(
            list(ma[name]["val_metrics"].values()), list(mb[name]["val_metrics"].values())
        )
```

Run: `uv run pytest tests/models/test_train.py -v`
Expected: FAIL, `ModuleNotFoundError: beaconai.models.train`.

- [ ] **Step 5: Implement the pyfunc adapter**

```python
# src/beaconai/models/pyfunc.py
"""Expose any ModelWrapper as an MLflow pyfunc model, so the registry can serve it later."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy import sparse

from beaconai.models.base import ModelWrapper

ARTIFACT_KEY = "model_dir"


class BeaconPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = ModelWrapper.load(context.artifacts[ARTIFACT_KEY])

    def predict(self, context: Any, model_input: Any, params: Any = None) -> np.ndarray:
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_numpy(dtype=np.float32)
        if not sparse.issparse(model_input):
            model_input = sparse.csr_matrix(np.asarray(model_input, dtype=np.float32))
        return self.model.predict_proba(model_input)
```

- [ ] **Step 6: Implement training**

```python
# src/beaconai/models/train.py
"""Tune every model on the validation split, log each trial to MLflow, register the champion.

Run: uv run python -m beaconai.models.train [--winners-only]
--winners-only retrains only the best params per model from the last train_summary.json
(used for the final full-data run, so the grid search itself stays on the dev subsample).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow import MlflowClient

from beaconai.config import (
    Config,
    MlflowConfig,
    TrainConfig,
    load_config,
    project_path,
    require_training,
)
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.metrics import evaluate_predictions
from beaconai.eval.plots import plot_confusion
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import SplitData, load_labels, load_manifest, load_split
from beaconai.models.pyfunc import ARTIFACT_KEY, BeaconPyfunc
from beaconai.models.registry import build_model
from beaconai.models.seed import set_global_seed
from beaconai.models.tuning import TrialResult, expand_grid, flatten_params, select_best

SUMMARY_FILE = "train_summary.json"
PIP_REQUIREMENTS = ["numpy", "scipy", "pandas", "scikit-learn", "xgboost", "torch"]


def configure_mlflow(cfg: MlflowConfig) -> None:
    db = project_path(cfg.tracking_db)
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    if mlflow.get_experiment_by_name(cfg.experiment) is None:
        root = project_path(cfg.artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(cfg.experiment, artifact_location=root.as_uri())
    mlflow.set_experiment(cfg.experiment)


def load_wrapper(model_uri: str) -> ModelWrapper:
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().model


def measure_latency_ms(model: ModelWrapper, data: SplitData, n_calls: int) -> float:
    """Median time of a single-patient predict_proba call, the shape the API will use."""
    times = []
    for i in range(min(n_calls, data.n)):
        row = data.X[i : i + 1]
        start = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def trial_plan(cfg: Config, winners_only: bool, summary_path: Path) -> list[tuple[str, dict]]:
    _, train_cfg, models_cfg, _ = require_training(cfg)
    if winners_only:
        if not summary_path.is_file():
            raise FileNotFoundError(f"{summary_path} not found; run a grid search first")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))["models"]
        return [(name, previous[name]["params"]) for name in train_cfg.models]
    plan = []
    for name in train_cfg.models:
        spec = getattr(models_cfg, name)
        plan += [(name, params) for params in expand_grid(spec.fixed, spec.grid)]
    return plan


def run_trial(
    name: str,
    params: dict[str, Any],
    index: int,
    train: SplitData,
    val: SplitData,
    class_names: list[str],
    cfg: Config,
    train_cfg: TrainConfig,
) -> TrialResult:
    _, _, _, eval_cfg = require_training(cfg)
    set_global_seed(cfg.seed)
    with mlflow.start_run(run_name=f"{name}-{index}", nested=True) as run:
        mlflow.set_tags({"model_type": name, "tuned_on": "validate"})
        mlflow.log_params(flatten_params(params))
        model = build_model(name, params, len(class_names), cfg.seed)
        start = time.perf_counter()
        model.fit(train.X, train.y, val.X, val.y)
        fit_seconds = time.perf_counter() - start
        result = evaluate_predictions(
            val.y, model.predict_proba(val.X), val.ddx, eval_cfg.ks, eval_cfg.ddx_ks
        )
        latency = measure_latency_ms(model, val, train_cfg.latency_calls)
        mlflow.log_metrics({f"val_{k}": v for k, v in result.metrics.items()})
        mlflow.log_metrics({"fit_seconds": fit_seconds, "latency_ms": latency})
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mlflow.log_artifact(
                str(
                    plot_confusion(
                        result.confusion,
                        class_names,
                        tmp_path / "val_confusion.png",
                        f"{name} (validation)",
                    )
                )
            )
            model.save(tmp_path / "wrapper")
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=BeaconPyfunc(),
                artifacts={ARTIFACT_KEY: str(tmp_path / "wrapper")},
                pip_requirements=PIP_REQUIREMENTS,
            )
    return TrialResult(
        name, params, run.info.run_id, info.model_uri, result.metrics, fit_seconds, latency
    )


def register_champion(champion: TrialResult, cfg: MlflowConfig, metric: str) -> int:
    mv = mlflow.register_model(champion.model_uri, cfg.registered_model)
    client = MlflowClient()
    client.set_registered_model_alias(cfg.registered_model, cfg.champion_alias, mv.version)
    client.set_model_version_tag(cfg.registered_model, mv.version, "model_type", champion.model)
    client.set_model_version_tag(
        cfg.registered_model, mv.version, f"val_{metric}", str(champion.val_metrics[metric])
    )
    return int(mv.version)


def run_training(cfg: Config, *, winners_only: bool = False) -> dict[str, Any]:
    mlflow_cfg, train_cfg, _, eval_cfg = require_training(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    summary_path = reports / SUMMARY_FILE
    plan = trial_plan(cfg, winners_only, summary_path)

    class_names = [lab["name"] for lab in load_labels(processed)]
    train, val = load_split(processed, "train"), load_split(processed, "validate")
    configure_mlflow(mlflow_cfg)
    metric = train_cfg.selection_metric

    with mlflow.start_run(run_name="phase2-train") as parent:
        mlflow.set_tags(
            {
                "dev_sample": str(cfg.data.dev_sample),
                "mode": "winners" if winners_only else "grid",
                "selection_metric": f"val_{metric}",
            }
        )
        mlflow.log_dict(load_manifest(processed), "data_manifest.json")
        results = [
            run_trial(name, params, i, train, val, class_names, cfg, train_cfg)
            for i, (name, params) in enumerate(plan)
        ]
        best = {
            name: select_best([r for r in results if r.model == name], metric)
            for name in train_cfg.models
        }
        champion = select_best(list(best.values()), metric)
        mlflow.log_metric(f"champion_val_{metric}", champion.val_metrics[metric])
        mlflow.set_tag("champion", champion.model)

    version = register_champion(champion, mlflow_cfg, metric)
    summary = {
        "parent_run_id": parent.info.run_id,
        "dev_sample": cfg.data.dev_sample,
        "mode": "winners" if winners_only else "grid",
        "selection_metric": metric,
        "champion": champion.model,
        "registered": {"name": mlflow_cfg.registered_model, "version": version},
        "models": {
            name: {
                "run_id": r.run_id,
                "model_uri": r.model_uri,
                "params": r.params,
                "val_metrics": r.val_metrics,
                "fit_seconds": r.fit_seconds,
                "latency_ms": r.latency_ms,
                "n_trials": sum(1 for t in results if t.model == name),
            }
            for name, r in best.items()
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winners-only", action="store_true")
    args = parser.parse_args()
    summary = run_training(load_config(), winners_only=args.winners_only)
    print(json.dumps({k: summary[k] for k in ("champion", "registered")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note on the MLflow API: this targets MLflow 3 (`log_model(name=...)` and `info.model_uri`). If the installed version rejects `name=`, check `uv run python -c "import mlflow; help(mlflow.pyfunc.log_model)"` and use the parameter it documents; do not downgrade MLflow silently.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/models tests/eval -v`
Expected: all PASS.

- [ ] **Step 8: Wire the Makefile**

Replace the `train` and `mlflow` targets:

```make
train:  ## Grid-search all models on validation, log to MLflow, register the best
	uv run python -m beaconai.models.train

train-final:  ## Retrain only the winning params from the last grid search (full data)
	uv run python -m beaconai.models.train --winners-only

mlflow:  ## MLflow UI on :5000 (backend path matches mlflow.tracking_db in config)
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Add `train-final` to `.PHONY`.

- [ ] **Step 9: Dev run on real data**

Run: `make train`
Expected: finishes without errors; `reports/train_summary.json` lists `logreg` (3 trials), `xgboost` (4), `mlp` (4); `make mlflow` shows 12 child runs plus a parent, each with metrics, a confusion PNG, and a model; the Models tab shows `beaconai-classifier` version 1 with alias `champion`. Record the run time per model.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/beaconai/models src/beaconai/eval/plots.py tests/models tests/eval/test_plots.py Makefile
git commit -m "Add grid search with MLflow logging and champion registration"
```

---

### Task 8: Evaluation, test-once guard, and the comparison report

**Files:**
- Create: `src/beaconai/eval/report.py`, `src/beaconai/eval/evaluate.py`
- Modify: `Makefile`
- Test: `tests/eval/test_report.py`, `tests/eval/test_evaluate.py`

**Interfaces:**
- Consumes: `train_summary.json` (Task 7), `load_wrapper`, `configure_mlflow` (Task 7), metrics and plots (Tasks 3, 7).
- Produces:
  - `ComparisonRow(model, params, val, test, fit_seconds, latency_ms)`
  - `ConfusionPair(true_name, pred_name, count, rate, true_severity, pred_severity)`
  - `render_comparison(rows, *, split, champion, selection_metric, dataset_note, confusions, lowest_recall, confusion_image, recall_image) -> str`
  - `TestGuardError(Exception)`; `check_test_allowed(summary: dict, lock_path: Path, force: bool) -> None`
  - `run_evaluation(cfg: Config, split: str, *, force: bool = False) -> Path` (path of the report)
  - Outputs: `reports/model_comparison.md`, `reports/figures/confusion_matrix.png`, `reports/figures/per_class_recall.png`, and for test, `reports/test_eval_lock.json`.

- [ ] **Step 1: Write the failing report test**

```python
# tests/eval/test_report.py
import unicodedata

from beaconai.eval.report import ComparisonRow, ConfusionPair, render_comparison


def _rows(with_test: bool) -> list[ComparisonRow]:
    val = {"top1_accuracy": 0.5, "top3_accuracy": 0.8, "macro_f1": 0.4}
    test = {"top1_accuracy": 0.49, "top3_accuracy": 0.79, "macro_f1": 0.39} if with_test else None
    return [ComparisonRow("xgboost", {"max_depth": 6}, val, test, 12.3, 0.8)]


def _render(with_test: bool) -> str:
    return render_comparison(
        _rows(with_test),
        split="test" if with_test else "validate",
        champion="xgboost",
        selection_metric="top3_accuracy",
        dataset_note="Full DDXPlus",
        confusions=[ConfusionPair("Bronchitis", "Pneumonia", 42, 0.12, 4, 3)],
        lowest_recall=[("Bronchitis", 0.31)],
        confusion_image="figures/confusion_matrix.png",
        recall_image="figures/per_class_recall.png",
    )


def test_report_contents() -> None:
    md = _render(with_test=True)
    assert "decision support, not diagnosis" in md
    assert "| xgboost |" in md and "0.800" in md and "0.790" in md
    assert "![Confusion matrix](figures/confusion_matrix.png)" in md
    assert "Bronchitis" in md and "Pneumonia" in md
    assert "**xgboost**" in md  # champion called out
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}  # docs style: no em dashes


def test_validation_only_report_says_test_pending() -> None:
    md = _render(with_test=False)
    assert "Test split not evaluated yet" in md
```

Run: `uv run pytest tests/eval/test_report.py -v` → FAIL.

- [ ] **Step 2: Implement the report renderer**

```python
# src/beaconai/eval/report.py
"""Render reports/model_comparison.md from evaluation results. Pure string building."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

COLUMNS = [
    ("top1_accuracy", "Top-1"),
    ("top3_accuracy", "Top-3"),
    ("top5_accuracy", "Top-5"),
    ("macro_f1", "Macro F1"),
    ("ddx_precision_at3", "DDx P@3"),
    ("ddx_recall_at5", "DDx R@5"),
]
DISCLAIMER = (
    "BeaconAI is decision support, not diagnosis. These models are trained only on "
    "synthetic data (DDXPlus) and are not a medical device."
)


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    params: dict[str, Any]
    val: dict[str, float]
    test: dict[str, float] | None
    fit_seconds: float
    latency_ms: float


@dataclass(frozen=True)
class ConfusionPair:
    true_name: str
    pred_name: str
    count: int
    rate: float
    true_severity: int | None
    pred_severity: int | None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _metric_table(rows: list[ComparisonRow], split: str) -> list[str]:
    def values(r: ComparisonRow) -> dict[str, float] | None:
        return r.val if split == "validate" else r.test

    present = [(k, t) for k, t in COLUMNS if any(k in (values(r) or {}) for r in rows)]
    lines = [
        "| Model | " + " | ".join(t for _, t in present) + " |",
        "|---|" + "---:|" * len(present),
    ]
    for r in rows:
        vals = values(r) or {}
        lines.append(f"| {r.model} | " + " | ".join(_fmt(vals.get(k)) for k, _ in present) + " |")
    return lines


def render_comparison(
    rows: list[ComparisonRow],
    *,
    split: str,
    champion: str,
    selection_metric: str,
    dataset_note: str,
    confusions: list[ConfusionPair],
    lowest_recall: list[tuple[str, float]],
    confusion_image: str,
    recall_image: str,
) -> str:
    out = [
        "# Model comparison",
        "",
        f"> {DISCLAIMER}",
        "",
        "Generated by `make eval`. Do not edit by hand; analysis lives in `docs/phase2_analysis.md`.",
        "",
        f"Data: {dataset_note}.",
        f"Selection: best validation `{selection_metric}`. Selected model: **{champion}**.",
        "",
        "## Validation",
        "",
        *_metric_table(rows, "validate"),
        "",
        "## Test",
        "",
    ]
    if split == "test":
        out += [*_metric_table(rows, "test"), ""]
    else:
        out += ["Test split not evaluated yet. It is scored once, on the full dataset.", ""]
    out += [
        "## Cost",
        "",
        "| Model | Fit time (s) | Latency per patient (ms) | Params |",
        "|---|---:|---:|---|",
        *[
            f"| {r.model} | {r.fit_seconds:.1f} | {r.latency_ms:.2f} | `{json.dumps(r.params)}` |"
            for r in rows
        ],
        "",
        "DDx P@k and R@k compare our top-k with the DDXPlus reference differential "
        "(conditions with non-zero probability). R@k is below 1 whenever the reference "
        "differential has more than k conditions.",
        "",
        f"## Confusion matrix ({champion}, {split})",
        "",
        f"![Confusion matrix]({confusion_image})",
        "",
        "Rows are normalised: each row shows where patients with that true condition were predicted.",
        "",
        "### Most frequent confusions",
        "",
        "| True | Predicted as | Count | Share of true | Severity (true / predicted) |",
        "|---|---|---:|---:|---|",
        *[
            f"| {c.true_name} | {c.pred_name} | {c.count} | {_fmt(c.rate)} | "
            f"{c.true_severity} / {c.pred_severity} |"
            for c in confusions
        ],
        "",
        "Severity is from DDXPlus `release_conditions.json` (1 is most severe).",
        "",
        "### Lowest per-class recall",
        "",
        f"![Per-class recall]({recall_image})",
        "",
        "| Condition | Recall |",
        "|---|---:|",
        *[f"| {name} | {_fmt(r)} |" for name, r in lowest_recall],
        "",
    ]
    return "\n".join(out)
```

Run: `uv run pytest tests/eval/test_report.py -v` → PASS.

- [ ] **Step 3: Write the failing evaluation tests**

```python
# tests/eval/test_evaluate.py
import json
from pathlib import Path

import pytest

from beaconai.eval.evaluate import TestGuardError, check_test_allowed, run_evaluation
from beaconai.models.train import run_training
from tests.conftest import make_config, write_toy_processed


def test_guard_blocks_dev_sample(tmp_path: Path) -> None:
    with pytest.raises(TestGuardError, match="full dataset"):
        check_test_allowed({"dev_sample": 50000, "parent_run_id": "r"}, tmp_path / "l.json", False)


def test_guard_blocks_second_run(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"train_parent_run_id": "old"}), encoding="utf-8")
    with pytest.raises(TestGuardError, match="already"):
        check_test_allowed({"dev_sample": None, "parent_run_id": "new"}, lock, False)


def test_guard_force_overrides(tmp_path: Path) -> None:
    check_test_allowed({"dev_sample": 50000, "parent_run_id": "r"}, tmp_path / "l.json", True)


def test_validate_report_on_dev(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=50000)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    report = run_evaluation(cfg, "validate")
    md = report.read_text(encoding="utf-8")
    assert "Test split not evaluated yet" in md
    assert (tmp_path / "reports" / "figures" / "confusion_matrix.png").is_file()
    with pytest.raises(TestGuardError):
        run_evaluation(cfg, "test")


def test_test_eval_runs_once_on_full_data(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    md = run_evaluation(cfg, "test").read_text(encoding="utf-8")
    assert "## Test" in md and "Test split not evaluated yet" not in md
    assert (tmp_path / "reports" / "test_eval_lock.json").is_file()
    with pytest.raises(TestGuardError, match="already"):
        run_evaluation(cfg, "test")


def test_unknown_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_evaluation(make_config(tmp_path, dev_sample=None), "train")
```

Run: `uv run pytest tests/eval/test_evaluate.py -v` → FAIL.

- [ ] **Step 4: Implement evaluation**

```python
# src/beaconai/eval/evaluate.py
"""Score the tuned models on a split, draw figures, and write reports/model_comparison.md.

Run: uv run python -m beaconai.eval.evaluate --split validate   (any time while iterating)
     uv run python -m beaconai.eval.evaluate --split test       (once, full data, at the end)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from beaconai.config import Config, load_config, project_path, require_training
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.metrics import EvalResult, evaluate_predictions, top_confusions
from beaconai.eval.plots import plot_confusion, plot_per_class_recall
from beaconai.eval.report import ComparisonRow, ConfusionPair, render_comparison
from beaconai.models.dataset import load_labels, load_split
from beaconai.models.train import SUMMARY_FILE, configure_mlflow, load_wrapper

LOCK_FILE = "test_eval_lock.json"
REPORT_FILE = "model_comparison.md"
SPLITS = ("validate", "test")


class TestGuardError(Exception):
    """Raised when a test-split evaluation would break the score-test-once rule."""

    __test__ = False  # not a pytest test class


def check_test_allowed(summary: dict[str, Any], lock_path: Path, force: bool) -> None:
    if force:
        return
    if summary["dev_sample"] is not None:
        raise TestGuardError(
            "Test evaluation needs the full dataset (data.dev_sample: null). "
            "Use --split validate while iterating."
        )
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        raise TestGuardError(
            f"Test split was already evaluated for training run {lock['train_parent_run_id']}. "
            "Scoring it again after changes leaks test information. Pass --force only if you "
            "accept that and will report it."
        )


def _confusion_pairs(
    result: EvalResult, labels: list[dict[str, Any]], n: int
) -> list[ConfusionPair]:
    support = result.confusion.sum(axis=1)
    return [
        ConfusionPair(
            labels[t]["name"],
            labels[p]["name"],
            c,
            c / support[t],
            labels[t].get("severity"),
            labels[p].get("severity"),
        )
        for t, p, c in top_confusions(result.confusion, n)
    ]


def run_evaluation(cfg: Config, split: str, *, force: bool = False) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; training data is never scored here")
    mlflow_cfg, train_cfg, _, eval_cfg = require_training(cfg)
    reports = project_path(eval_cfg.output_dir)
    summary = json.loads((reports / SUMMARY_FILE).read_text(encoding="utf-8"))
    lock_path = reports / LOCK_FILE
    if split == "test":
        check_test_allowed(summary, lock_path, force)

    processed = processed_dir_from_config(cfg.data.paths.processed)
    labels = load_labels(processed)
    names = [lab["name"] for lab in labels]
    data = load_split(processed, split)
    configure_mlflow(mlflow_cfg)

    rows, results = [], {}
    with mlflow.start_run(run_name=f"phase2-eval-{split}"):
        mlflow.set_tags({"split": split, "train_parent_run_id": summary["parent_run_id"]})
        for name, info in summary["models"].items():
            if split == "validate":
                metrics = info["val_metrics"]
                model = load_wrapper(info["model_uri"]) if name == summary["champion"] else None
                if model is not None:
                    results[name] = evaluate_predictions(
                        data.y, model.predict_proba(data.X), data.ddx, eval_cfg.ks, eval_cfg.ddx_ks
                    )
            else:
                model = load_wrapper(info["model_uri"])
                results[name] = evaluate_predictions(
                    data.y, model.predict_proba(data.X), data.ddx, eval_cfg.ks, eval_cfg.ddx_ks
                )
                metrics = results[name].metrics
                mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})
            rows.append(
                ComparisonRow(
                    name,
                    info["params"],
                    info["val_metrics"],
                    metrics if split == "test" else None,
                    info["fit_seconds"],
                    info["latency_ms"],
                )
            )

        champ = results[summary["champion"]]
        figures = reports / "figures"
        cm_path = plot_confusion(
            champ.confusion,
            names,
            figures / "confusion_matrix.png",
            f"{summary['champion']} ({split})",
        )
        recall_path = plot_per_class_recall(
            champ.recall,
            names,
            figures / "per_class_recall.png",
            f"{summary['champion']} per-class recall ({split})",
        )
        lowest = sorted(
            ((names[i], float(r)) for i, r in enumerate(champ.recall) if not math.isnan(r)),
            key=lambda x: x[1],
        )[: eval_cfg.lowest_recall]
        note = (
            f"DDXPlus {'dev subsample' if summary['dev_sample'] else 'full dataset'}, "
            f"{data.n:,} {split} patients"
        )
        md = render_comparison(
            rows,
            split=split,
            champion=summary["champion"],
            selection_metric=train_cfg.selection_metric,
            dataset_note=note,
            confusions=_confusion_pairs(champ, labels, eval_cfg.top_confusions),
            lowest_recall=lowest,
            confusion_image="figures/confusion_matrix.png",
            recall_image="figures/per_class_recall.png",
        )
        report = reports / REPORT_FILE
        report.write_text(md, encoding="utf-8")
        for p in (report, cm_path, recall_path):
            mlflow.log_artifact(str(p))

    if split == "test":
        lock_path.write_text(
            json.dumps(
                {
                    "train_parent_run_id": summary["parent_run_id"],
                    "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="validate")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        print(run_evaluation(load_config(), args.split, force=args.force))
    except TestGuardError as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 6: Wire the Makefile**

```make
eval:  ## Score test ONCE on full data and write reports/model_comparison.md
	uv run python -m beaconai.eval.evaluate --split test

eval-val:  ## Validation-only report, safe to run while iterating
	uv run python -m beaconai.eval.evaluate --split validate
```

Add `eval-val` to `.PHONY`.

- [ ] **Step 7: Dev report on real data**

Run: `make eval-val`
Expected: `reports/model_comparison.md` with a validation table for 3 models, "Test split not evaluated yet", and both figures under `reports/figures/`. Then run `make eval` and confirm it is refused with the dev-sample message (exit code 2).

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/beaconai/eval tests/eval Makefile
git commit -m "Add evaluation with test-once guard and model comparison report"
```

---

### Task 9: Final full-data run

No new code. Run only after Tasks 0 to 8 are merged into the branch and the dev results look sane.

- [ ] **Step 1: Regenerate processed data on the full set**

Run: `BEACONAI_DEV_SAMPLE=null make data`
Expected: `manifest.json` shows about 1.03M train, 132k validate, 135k test, and `"dev_sample": null`.

- [ ] **Step 2: Retrain the dev winners on full data**

Run: `BEACONAI_DEV_SAMPLE=null make train-final`
Expected: 3 trials (one per model), new registered version with the `champion` alias. Note wall time; expect XGBoost to dominate.

Selecting hyperparameters on the dev validation subsample and refitting on the full training set keeps tuning on validation only. Record this in the analysis doc.

- [ ] **Step 3: Score test once**

Run: `BEACONAI_DEV_SAMPLE=null make eval`
Expected: `reports/model_comparison.md` with validation and test tables, `reports/test_eval_lock.json` written. Do not re-run.

- [ ] **Step 4: Restore the dev data for daily work**

Run: `make data`

---

### Task 10: Analysis, docs, and phase close-out

**Files:**
- Create: `docs/phase2_analysis.md`
- Modify: `reports/CHANGELOG.md`, `README.md`, and the local project notes file (gitignored: update the "Current phase" line only)

- [ ] **Step 1: Write `docs/phase2_analysis.md`** (plain style, no em dashes) with three sections:
  - **Model choice:** a table of accuracy (val and test top-3), fit time, latency, and explainability (logreg: coefficients; XGBoost: exact TreeSHAP; MLP: approximate SHAP only) and the reason the registered champion was chosen.
  - **Confusions:** for each of the top 5 pairs in the report, whether the two conditions share presenting symptoms (for example both respiratory with cough and fever) and whether the severity gap makes the confusion dangerous (a severity-1 condition predicted as a severity-4 one).
  - **Method notes:** tuning on the dev validation subsample, refit on full data, test scored once, differential-overlap definition.

- [ ] **Step 2: Add the CHANGELOG entry**

```markdown
## Phase 2: Baseline models and evaluation (YYYY-MM-DD)

- Three models behind one wrapper (fit, predict_proba, save, load): logistic regression, XGBoost, PyTorch MLP.
- Grid search on validation only; every trial logged to MLflow with params, metrics, confusion plot, and model.
- Champion registered as `beaconai-classifier` (alias `champion`) by validation top-3 accuracy.
- Test split scored once on the full dataset; guard prevents repeat scoring.
- Metrics (test, full data): <champion> top-1 X.XXX, top-3 X.XXX, top-5 X.XXX, macro F1 X.XXX, DDx normalized recall@3 X.XXX.
```

Fill the date and numbers from `reports/model_comparison.md`.

- [ ] **Step 3: Update README commands** (`make train`, `make train-final`, `make eval-val`, `make eval`, `make mlflow`), and set the local notes file's phase line to `Phase 3 (Phase 2 done <date>; ...)`.

- [ ] **Step 4: Full verification**

Run: `make lint && make test`
Expected: clean and all PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/phase2_analysis.md reports/CHANGELOG.md reports/model_comparison.md reports/figures reports/train_summary.json reports/test_eval_lock.json README.md
git commit -m "Phase 2: model comparison report, analysis, and changelog"
```

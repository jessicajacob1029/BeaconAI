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


def test_tie_breakers_default_and_validation() -> None:
    _, train, _, _ = require_training(load_config())
    assert train.tie_breakers == ["top1_accuracy", "macro_f1"]
    raw = _base_raw()
    raw["train"]["tie_breakers"] = ["auc"]
    with pytest.raises(ValidationError, match="tie_breakers"):
        Config.model_validate(raw)

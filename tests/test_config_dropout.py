import pytest
import yaml
from pydantic import ValidationError

from beaconai.config import (
    DEFAULT_CONFIG_PATH,
    CalibrationConfig,
    Config,
    DropoutConfig,
    FinalEvalConfig,
    load_config,
    require_dropout,
)


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_has_dropout_and_new_calibration_keys() -> None:
    cfg = load_config()
    drop = require_dropout(cfg)
    assert drop.models == ["logreg", "xgboost"]
    assert drop.rate_range == (0.5, 0.95)
    assert drop.copies >= 1 and drop.keep_original
    assert cfg.calibration.variants == ["baseline", "dropout"]
    assert cfg.calibration.fit_on == ["full", "partial"]
    assert cfg.calibration.evidence_curve == [0, 1, 2, 3, 5, 8, 12]
    assert cfg.final_eval.name == "classifier_final"


@pytest.mark.parametrize("bounds", [[0.0, 0.5], [0.5, 1.0], [0.9, 0.5], [-0.1, 0.5]])
def test_rate_range_inside_zero_one_and_ordered(bounds: list[float]) -> None:
    raw = _raw()["dropout"]
    raw["rate_range"] = bounds
    with pytest.raises(ValidationError, match="rate_range"):
        DropoutConfig.model_validate(raw)


def test_fixed_rate_allowed() -> None:
    raw = _raw()["dropout"]
    raw["rate_range"] = [0.75, 0.75]
    assert DropoutConfig.model_validate(raw).rate_range == (0.75, 0.75)


def test_no_training_rows_rejected() -> None:
    raw = _raw()["dropout"]
    raw.update(copies=0, keep_original=False)
    with pytest.raises(ValidationError, match="no training rows"):
        DropoutConfig.model_validate(raw)


def test_new_calibration_keys_default_to_phase3_behaviour() -> None:
    raw = _raw()["calibration"]
    del raw["variants"], raw["fit_on"], raw["evidence_curve"]
    cal = CalibrationConfig.model_validate(raw)
    assert cal.variants == ["baseline"] and cal.fit_on == ["full"]
    assert cal.evidence_curve == []


def test_evidence_curve_values_non_negative() -> None:
    raw = _raw()["calibration"]
    raw["evidence_curve"] = [0, -1]
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate(raw)


@pytest.mark.parametrize("name", ["", "../escape", "has space"])
def test_final_eval_name_is_a_safe_file_stem(name: str) -> None:
    with pytest.raises(ValidationError):
        FinalEvalConfig(name=name)


def test_require_dropout_missing() -> None:
    raw = _raw()
    del raw["dropout"]
    with pytest.raises(ValueError, match="dropout"):
        require_dropout(Config.model_validate(raw))

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
    assert cal.top3_floor.full == 0.99
    assert 0.0 <= cal.top3_floor.partial <= 1.0
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


def test_top3_floor_needs_both_settings_in_range() -> None:
    raw = _raw()["calibration"]
    raw["top3_floor"] = {"full": 0.99}
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate(raw)
    raw["top3_floor"] = {"full": 0.99, "partial": 1.5}
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate(raw)

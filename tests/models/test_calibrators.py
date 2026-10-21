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
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().transform(np.array([[0.5, 0.5]]))


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        make_calibrator("platt", BOUNDS)
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrator_from_dict({"method": "platt"})

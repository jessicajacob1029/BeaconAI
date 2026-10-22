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


def test_coverage_curve_with_tie_break() -> None:
    proba = np.array([[0.9, 0.05, 0.05]])
    raw = np.array([[0.8, 0.05, 0.15]])
    curve = coverage_curve(np.array([2]), proba, [0.0], k=2, tie_break=raw)
    assert curve.topk_accuracy[0] == 1.0
    assert coverage_curve(np.array([2]), proba, [0.0], k=2).topk_accuracy[0] == 0.0

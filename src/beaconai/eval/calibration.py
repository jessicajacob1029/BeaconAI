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
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Sequence[float],
    k: int,
    tie_break: np.ndarray | None = None,
) -> CoverageCurve:
    y, p = check_inputs(y_true, proba)
    conf = p.max(axis=1)
    top1_hit = p.argmax(axis=1) == y
    topk_hit = (top_k_indices(p, k, tie_break) == y[:, None]).any(axis=1)
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

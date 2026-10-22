"""Exact explanations for a multinomial linear model, in logit space.

Logits are z = W x + b and the base model's probabilities are softmax(z). Contributions are
measured against a baseline patient m: for evidence columns m is 0 (evidence absent), for age
and sex columns m is the training-set mean. Feature j adds W[c, j] * (x[j] - m[j]) to
condition c's logit, and the baseline score is b + W m, so the baseline score plus all
contributions equals the logit exactly.

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


def _deltas(
    x_row: sparse.csr_matrix | np.ndarray, baseline: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """(columns, x - m) over every column where x or the baseline is non-zero."""
    cols, vals = _row(x_row)
    if baseline is None:
        return cols, vals
    m = np.asarray(baseline, dtype=np.float64)
    extra = np.setdiff1d(np.flatnonzero(m), cols)
    all_cols = np.concatenate([cols, extra])
    all_vals = np.concatenate([vals, np.zeros(len(extra))])
    order = np.argsort(all_cols, kind="stable")
    all_cols, all_vals = all_cols[order], all_vals[order]
    return all_cols, all_vals - m[all_cols]


def logits(W: np.ndarray, b: np.ndarray, x_row: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    cols, vals = _row(x_row)
    return b + W[:, cols] @ vals


def baseline_logits(W: np.ndarray, b: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    """Logits of the baseline patient: b when the baseline is all-absent, else b + W m."""
    if baseline is None:
        return np.asarray(b, dtype=np.float64).copy()
    return b + W @ np.asarray(baseline, dtype=np.float64)


def contributions(
    W: np.ndarray,
    x_row: sparse.csr_matrix | np.ndarray,
    c: int,
    baseline: np.ndarray | None = None,
) -> Pairs:
    """(feature index, W[c, j] * (x[j] - m[j])) for every column where x or m is non-zero."""
    cols, deltas = _deltas(x_row, baseline)
    return [(int(j), float(W[c, j] * d)) for j, d in zip(cols, deltas, strict=True)]


def contrast_contributions(
    W: np.ndarray,
    x_row: sparse.csr_matrix | np.ndarray,
    a: int,
    b: int,
    baseline: np.ndarray | None = None,
) -> Pairs:
    """(feature index, (W[a, j] - W[b, j]) * (x[j] - m[j])): positive favors a, negative b."""
    cols, deltas = _deltas(x_row, baseline)
    diff = W[a, cols] - W[b, cols]
    return [(int(j), float(dw * d)) for j, dw, d in zip(cols, diff, deltas, strict=True)]


def top_contributors(pairs: Pairs, n_pos: int, n_neg: int) -> tuple[Pairs, Pairs]:
    """Largest positive and most negative contributions; zeros excluded; ties by feature index."""
    pos = sorted((p for p in pairs if p[1] > 0), key=lambda p: (-p[1], p[0]))[:n_pos]
    neg = sorted((p for p in pairs if p[1] < 0), key=lambda p: (p[1], p[0]))[:n_neg]
    return pos, neg

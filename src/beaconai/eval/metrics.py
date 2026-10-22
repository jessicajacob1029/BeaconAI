"""Classifier metrics.

Inputs are integer class ids and an (n_patients, n_classes) probability matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import f1_score


def check_inputs(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def top_k_indices(proba: np.ndarray, k: int, tie_break: np.ndarray | None = None) -> np.ndarray:
    """Top-k class ids per row, highest first.

    Equal probabilities are ordered by `tie_break` (for example the model's uncalibrated
    scores) when given, then by the lower class id.
    """
    p = np.asarray(proba, dtype=np.float64)
    if not 1 <= k <= p.shape[1]:
        raise ValueError(f"k must be in [1, {p.shape[1]}], got {k}")
    if tie_break is None:
        return np.argsort(-p, axis=1, kind="stable")[:, :k]
    t = np.asarray(tie_break, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"tie_break shape {t.shape} does not match proba shape {p.shape}")
    return np.lexsort((-t, -p), axis=-1)[:, :k]


def top_k_accuracy(
    y_true: np.ndarray, proba: np.ndarray, k: int, tie_break: np.ndarray | None = None
) -> float:
    """Share of rows whose true class is in the top k (ties ordered by `tie_break` if given)."""
    y, p = check_inputs(y_true, proba)
    return float((top_k_indices(p, k, tie_break) == y[:, None]).any(axis=1).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over classes present in y_true or y_pred."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    return _sk_confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(np.int64)


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Recall per true class; nan for classes with no true examples."""
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


def ddx_overlap_at_k(
    proba: np.ndarray, ddx: sparse.csr_matrix, k: int, tie_break: np.ndarray | None = None
) -> dict[str, float]:
    """Overlap between our top-k and the DDXPlus ground-truth differential.

    D = classes with non-zero reference probability, P = our top-k classes.
    precision = |P & D| / k, recall = |P & D| / |D|, recall_norm = |P & D| / min(k, |D|).
    Each is averaged over patients with a non-empty D; n_rows is how many that was.
    If no patient has a D, the three averages are nan.
    """
    p = np.asarray(proba, dtype=np.float64)
    if ddx.shape != p.shape:
        raise ValueError(f"ddx shape {ddx.shape} does not match proba shape {p.shape}")
    in_ref = (ddx.toarray() if sparse.issparse(ddx) else np.asarray(ddx)) > 0
    top = top_k_indices(p, k, tie_break)
    hits = np.take_along_axis(in_ref, top, axis=1).sum(axis=1)
    ref_size = in_ref.sum(axis=1)

    keep = ref_size > 0
    if not keep.any():
        return {"precision": np.nan, "recall": np.nan, "recall_norm": np.nan, "n_rows": 0}
    hits, ref_size = hits[keep], ref_size[keep]
    return {
        "precision": float(np.mean(hits / k)),
        "recall": float(np.mean(hits / ref_size)),
        "recall_norm": float(np.mean(hits / np.minimum(k, ref_size))),
        "n_rows": int(keep.sum()),
    }


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
    tie_break: np.ndarray | None = None,
) -> EvalResult:
    """All metrics. `tie_break` (the model's uncalibrated scores) orders exactly tied
    probabilities the way predictions are served."""
    y, p = check_inputs(y_true, proba)
    y_pred = p.argmax(axis=1)
    metrics = {f"top{k}_accuracy": top_k_accuracy(y, p, k, tie_break) for k in ks}
    metrics["macro_f1"] = macro_f1(y, y_pred)
    if ddx is not None:
        for k in ddx_ks:
            overlap = ddx_overlap_at_k(p, ddx, k, tie_break)
            metrics[f"ddx_precision_at{k}"] = overlap["precision"]
            metrics[f"ddx_recall_at{k}"] = overlap["recall"]
            metrics[f"ddx_recall_norm_at{k}"] = overlap["recall_norm"]
    cm = confusion_matrix(y, y_pred, p.shape[1])
    return EvalResult(metrics=metrics, confusion=cm, recall=per_class_recall(cm))

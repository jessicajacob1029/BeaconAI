"""A 'partial evidence' version of a split, closer to what users tell us.

Each patient keeps the initial evidence (what they came in with) plus a few other random
evidences. An evidence is one question with all its selected values (for example E_55 and every
E_55_@_V_* column), so a multi-choice answer is kept or dropped as a whole. Age and sex are
always kept. Dropped evidences become 0, which the models read the same as "no": this measures
a model trained on complete answers when most answers are missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def column_evidence(features: list[dict[str, Any]]) -> np.ndarray:
    """Evidence code per column, '' for columns that are not evidence (age, sex)."""
    ordered = sorted(features, key=lambda f: f["index"])
    return np.array([f.get("evidence") or "" for f in ordered], dtype=object)


def partial_evidence(
    X: sparse.csr_matrix,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    extra: int,
    seed: int,
) -> sparse.csr_matrix:
    X = sparse.csr_matrix(X)
    if len(initial_evidence) != X.shape[0]:
        raise ValueError(f"{len(initial_evidence)} initial evidences for {X.shape[0]} rows")
    if len(col_evidence) != X.shape[1]:
        raise ValueError(f"{len(col_evidence)} column codes for {X.shape[1]} columns")
    rng = np.random.default_rng(seed)
    indptr, indices, data = [0], [], []
    for i in range(X.shape[0]):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        cols, vals = X.indices[lo:hi], X.data[lo:hi]
        codes = col_evidence[cols]
        others = sorted({c for c in codes if c and c != initial_evidence[i]})
        n_pick = min(extra, len(others))
        picked = set(rng.choice(others, size=n_pick, replace=False)) if n_pick else set()
        keep = picked | {str(initial_evidence[i])}
        mask = np.array([c == "" or c in keep for c in codes], dtype=bool)
        indices.extend(cols[mask])
        data.extend(vals[mask])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=X.shape,
    )

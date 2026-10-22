"""Global view of the linear champion: top features per condition, sanity flags, and how often
calibration changes the ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from beaconai.eval.metrics import top_k_indices
from beaconai.explain.baseline import DEMOGRAPHIC_KINDS
from beaconai.explain.labels import FeatureLabeler

RARE = "rarely seen with this condition"
REVERSED = "less common with this condition than overall"
DEMOGRAPHIC = "demographic"


@dataclass(frozen=True)
class TopFeature:
    index: int
    name: str
    label: str
    coef: float
    prevalence_condition: float
    prevalence_all: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def global_top_features(
    W: np.ndarray,
    y: np.ndarray,
    X: sparse.csr_matrix,
    labels: list[dict[str, Any]],
    labeler: FeatureLabeler,
    conditions: list[str],
    n: int,
    min_prevalence: float,
) -> dict[str, list[TopFeature]]:
    """For each condition, up to n features with the largest positive coefficient.

    A coefficient is the feature's contribution when fully present (features are scaled to
    [0, 1]), relative to the average condition. Prevalence comes from the given split.
    """
    names = [lab["name"] for lab in labels]
    unknown = [c for c in conditions if c not in names]
    if unknown:
        raise ValueError(f"unknown conditions in explain.global_conditions: {unknown}")
    present = sparse.csr_matrix(X, copy=True)
    present.data = (present.data != 0).astype(np.float64)
    prevalence_all = np.asarray(present.mean(axis=0)).ravel()
    out: dict[str, list[TopFeature]] = {}
    for cond in conditions:
        c = names.index(cond)
        rows = np.asarray(y) == c
        prev_c = (
            np.asarray(present[rows].mean(axis=0)).ravel() if rows.any() else np.zeros(X.shape[1])
        )
        order = [j for j in np.argsort(-W[c], kind="stable") if W[c, j] > 0][:n]
        items = []
        for j in order:
            flags = []
            if prev_c[j] < min_prevalence:
                flags.append(RARE)
            if prev_c[j] < prevalence_all[j]:
                flags.append(REVERSED)
            if labeler.kind(int(j)) in DEMOGRAPHIC_KINDS:
                flags.append(DEMOGRAPHIC)
            items.append(
                TopFeature(
                    int(j),
                    labeler.name(int(j)),
                    labeler.label(int(j)),
                    float(W[c, j]),
                    float(prev_c[j]),
                    float(prevalence_all[j]),
                    flags,
                )
            )
        out[cond] = items
    return out


def ranking_change_rates(
    raw_proba: np.ndarray, calibrated_proba: np.ndarray, k: int, min_prob: float
) -> dict[str, float]:
    """How often calibration changes what the user sees, compared with the model's own order.

    The calibrated order breaks ties by the model's own score, as predictions do.
    - top1_changed: the top condition differs.
    - topk_set_changed: the set of top-k conditions differs.
    - shown_set_changed: the set differs among conditions with probability >= min_prob (raw
      for the model's set, calibrated for the shown set): the items an API would display.
    """
    raw = np.asarray(raw_proba, dtype=np.float64)
    cal = np.asarray(calibrated_proba, dtype=np.float64)
    raw_top = top_k_indices(raw, k)
    cal_top = top_k_indices(cal, k, tie_break=raw)
    top1 = raw_top[:, 0] != cal_top[:, 0]
    sets, shown = [], []
    for i, (r, c) in enumerate(zip(raw_top.tolist(), cal_top.tolist(), strict=True)):
        sets.append(set(r) != set(c))
        shown.append(
            {j for j in r if raw[i, j] >= min_prob} != {j for j in c if cal[i, j] >= min_prob}
        )
    return {
        "top1_changed": float(top1.mean()),
        "topk_set_changed": float(np.mean(sets)),
        "shown_set_changed": float(np.mean(shown)),
    }

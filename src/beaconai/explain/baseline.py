"""The baseline patient that explanations are measured against.

Evidence columns use the absent baseline (0). Age and sex columns use the training-set mean,
so every patient, male or female, young or old, gets a demographic contribution with the same
meaning: how their age or sex moves the score compared with an average training patient.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEMOGRAPHIC_KINDS = {"age_bucket", "age_scaled", "sex"}


def _ordered(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(features, key=lambda f: f["index"])


def demographic_means(
    ages: Sequence[int] | np.ndarray,
    sexes: Sequence[str] | np.ndarray,
    features: list[dict[str, Any]],
    age_bins: Sequence[int],
    age_scale: float,
) -> dict[str, float]:
    """Training-set mean of every demographic column, encoded exactly as Phase 1 encodes them."""
    ages = np.asarray(ages, dtype=np.float64)
    sexes = np.asarray(sexes)
    feats = _ordered(features)
    buckets = [f for f in feats if f.get("kind") == "age_bucket"]
    if len(buckets) != len(age_bins):
        raise ValueError(f"{len(buckets)} age bucket columns for {len(age_bins)} age bins")
    bucket = np.searchsorted(np.asarray(age_bins), ages, side="right") - 1
    means = {f["name"]: float(np.mean(bucket == i)) for i, f in enumerate(buckets)}
    for f in feats:
        if f.get("kind") == "age_scaled":
            means[f["name"]] = float(np.mean(np.minimum(ages / age_scale, 1.0)))
        elif f.get("kind") == "sex":
            means[f["name"]] = float(np.mean(sexes == "M"))
    return means


def baseline_vector(features: list[dict[str, Any]], means: dict[str, float]) -> np.ndarray:
    """m: 0 for evidence columns, the training mean for demographic columns."""
    feats = _ordered(features)
    m = np.zeros(len(feats))
    missing = []
    for f in feats:
        if f.get("kind") in DEMOGRAPHIC_KINDS:
            if f["name"] not in means:
                missing.append(f["name"])
            else:
                m[f["index"]] = means[f["name"]]
    if missing:
        raise ValueError(f"baseline means missing for demographic columns: {missing}")
    return m


def save_baseline(path: Path, means: dict[str, float], source: str, n_patients: int) -> None:
    payload = {"source": source, "n_patients": int(n_patients), "means": means}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["means"], {k: v for k, v in payload.items() if k != "means"}

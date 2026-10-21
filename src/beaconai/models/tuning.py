"""Grid expansion and winner selection. Pure functions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class TrialResult:
    model: str
    params: dict[str, Any]
    run_id: str
    model_uri: str
    val_metrics: dict[str, float]
    fit_seconds: float
    latency_ms: float


def expand_grid(fixed: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of grid values, each merged with the fixed params."""
    keys = sorted(grid)
    return [
        {**fixed, **dict(zip(keys, combo, strict=True))}
        for combo in product(*(grid[k] for k in keys))
    ]


def flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """MLflow params must be scalars; lists and dicts become JSON strings."""
    return {k: json.dumps(v) if isinstance(v, list | dict) else v for k, v in params.items()}


def select_best(
    results: list[TrialResult], metric: str, tie_breakers: Sequence[str] = ()
) -> TrialResult:
    """Highest validation `metric` wins; exact ties go to `tie_breakers` in order.

    If every metric ties, the earlier trial is kept.
    """
    if not results:
        raise ValueError("no trials to select from")
    for m in (metric, *tie_breakers):
        if m not in results[0].val_metrics:
            raise KeyError(f"metric {m!r} not in {sorted(results[0].val_metrics)}")

    def score(r: TrialResult) -> tuple[float, ...]:
        return tuple(r.val_metrics[m] for m in (metric, *tie_breakers))

    best = results[0]
    for r in results[1:]:
        if score(r) > score(best):
            best = r
    return best

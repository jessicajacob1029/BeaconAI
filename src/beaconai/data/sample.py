"""Dev subsampling for DDXPlus splits."""

from __future__ import annotations

import pandas as pd


def subsample(
    df: pd.DataFrame,
    n: int | None,
    seed: int,
    min_per_class: int = 200,
    label_col: str = "PATHOLOGY",
) -> pd.DataFrame:
    """Stratified sample of n rows, guaranteeing a minimum per class where possible.

    The original index is kept, so every returned row can be traced back to df.
    """
    if n is None or n >= len(df):
        return df

    counts = df[label_col].value_counts()
    floor = min(min_per_class, n // len(counts))
    base = counts.clip(upper=floor)
    spare = counts - base
    budget = n - int(base.sum())
    extra = (spare / spare.sum() * budget).astype(int) if spare.sum() else 0
    alloc = base + extra

    picked = pd.concat(
        group.sample(n=int(alloc[label]), random_state=seed)
        for label, group in df.groupby(label_col)
    )
    short = n - len(picked)
    if short > 0:
        rest = df.drop(picked.index)
        picked = pd.concat([picked, rest.sample(n=short, random_state=seed)])

    return picked.sample(frac=1, random_state=seed)

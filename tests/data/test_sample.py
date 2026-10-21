"""Contract tests for the dev subsample. They hold for any sampling strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from beaconai.data.sample import subsample


@pytest.fixture
def df() -> pd.DataFrame:
    # 1,000 patients over 5 conditions, one of them rare (10 patients).
    labels = ["A"] * 400 + ["B"] * 300 + ["C"] * 200 + ["D"] * 90 + ["RARE"] * 10
    return pd.DataFrame({"PATHOLOGY": labels, "AGE": range(len(labels))})


def test_none_returns_everything(df: pd.DataFrame) -> None:
    assert subsample(df, None, seed=42).equals(df)


def test_n_at_least_len_returns_everything(df: pd.DataFrame) -> None:
    assert subsample(df, len(df), seed=42).equals(df)
    assert subsample(df, len(df) + 5, seed=42).equals(df)


@pytest.mark.parametrize("n", [1, 50, 100, 999])
def test_returns_exactly_n_unique_rows_from_df(df: pd.DataFrame, n: int) -> None:
    out = subsample(df, n, seed=42)
    assert len(out) == n
    assert out.index.is_unique
    assert out.index.isin(df.index).all()
    pd.testing.assert_frame_equal(out, df.loc[out.index])  # rows unchanged


def test_same_seed_same_rows(df: pd.DataFrame) -> None:
    a = subsample(df, 100, seed=7)
    b = subsample(df, 100, seed=7)
    assert sorted(a.index) == sorted(b.index)


def test_different_seed_different_rows(df: pd.DataFrame) -> None:
    a = subsample(df, 100, seed=1)
    b = subsample(df, 100, seed=2)
    assert sorted(a.index) != sorted(b.index)


def test_input_not_modified(df: pd.DataFrame) -> None:
    before = df.copy()
    subsample(df, 100, seed=42)
    pd.testing.assert_frame_equal(df, before)

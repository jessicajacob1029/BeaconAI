import numpy as np
import pytest
from scipy import sparse

from beaconai.eval.partial import column_evidence, partial_evidence

FEATURES = [
    {"index": 0, "evidence": "E_1"},
    {"index": 1, "evidence": "E_2"},
    {"index": 2, "evidence": "E_3"},  # scale evidence, value 0.6
    {"index": 3, "evidence": "E_4"},  # multi-choice value V_1
    {"index": 4, "evidence": "E_4"},  # multi-choice value V_2
    {"index": 5, "evidence": "E_5"},
    {"index": 6, "evidence": "E_6"},
    {"index": 7, "evidence": None},  # age
    {"index": 8, "evidence": None},  # sex
]
COLS = column_evidence(FEATURES)
X = sparse.csr_matrix(
    np.array(
        [
            [1, 1, 0.6, 1, 1, 1, 1, 0.4, 1],  # initial E_1, five other evidences
            [1, 0, 0, 0, 0, 1, 0, 0.2, 0],  # initial E_5, one other evidence
        ],
        dtype=np.float32,
    )
)
INITIAL = np.array(["E_1", "E_5"])


def _kept_codes(row: sparse.csr_matrix) -> set[str]:
    return {c for c in COLS[row.indices] if c}


def test_column_evidence() -> None:
    assert COLS.tolist() == ["E_1", "E_2", "E_3", "E_4", "E_4", "E_5", "E_6", "", ""]


def test_keeps_initial_plus_extra_and_demographics() -> None:
    out = partial_evidence(X, INITIAL, COLS, extra=3, seed=0)
    kept = _kept_codes(out[0])
    assert "E_1" in kept and len(kept) == 4
    assert {7, 8} <= set(out[0].indices)  # age and sex always kept


def test_evidence_values_stay_together_and_unchanged() -> None:
    for seed in range(20):
        row = partial_evidence(X, INITIAL, COLS, extra=3, seed=seed)[0].toarray()[0]
        assert (row[3] == 0) == (row[4] == 0)  # both values of E_4 or neither
        assert row[2] in (0.0, pytest.approx(0.6))
        assert row[7] == pytest.approx(0.4)


def test_fewer_others_than_extra_keeps_everything() -> None:
    out = partial_evidence(X, INITIAL, COLS, extra=3, seed=0)
    np.testing.assert_array_equal(out[1].toarray(), X[1].toarray())


def test_extra_zero_keeps_only_initial() -> None:
    out = partial_evidence(X, INITIAL, COLS, extra=0, seed=0)
    assert _kept_codes(out[0]) == {"E_1"}


def test_seeded_and_varies_with_seed() -> None:
    a = partial_evidence(X, INITIAL, COLS, extra=2, seed=1).toarray()
    b = partial_evidence(X, INITIAL, COLS, extra=2, seed=1).toarray()
    np.testing.assert_array_equal(a, b)
    picks = {tuple(partial_evidence(X, INITIAL, COLS, 2, s)[0].indices) for s in range(20)}
    assert len(picks) > 1


def test_shape_mismatches_raise() -> None:
    with pytest.raises(ValueError):
        partial_evidence(X, INITIAL[:1], COLS, 3, 0)
    with pytest.raises(ValueError):
        partial_evidence(X, INITIAL, COLS[:5], 3, 0)

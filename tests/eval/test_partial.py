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


# Output of the Phase 3 implementation, captured before the refactor that added dropout.
GOLDEN = (
    [
        0,
        4,
        8,
        13,
        15,
        20,
        26,
        30,
        35,
        40,
        44,
        49,
        53,
        58,
        62,
        66,
        71,
        74,
        78,
        83,
        85,
        90,
        95,
        101,
        106,
        110,
        115,
        118,
        122,
        125,
        130,
        134,
        140,
        143,
        149,
        153,
        157,
        162,
        166,
        172,
        175,
    ],
    [
        2,
        3,
        4,
        7,
        1,
        2,
        4,
        8,
        2,
        5,
        6,
        7,
        8,
        5,
        8,
        2,
        3,
        4,
        6,
        8,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        4,
        6,
        8,
        0,
        1,
        3,
        7,
        8,
        1,
        2,
        3,
        7,
        8,
        1,
        2,
        6,
        8,
        3,
        4,
        5,
        6,
        8,
        1,
        2,
        4,
        8,
        3,
        4,
        5,
        6,
        7,
        2,
        4,
        7,
        8,
        0,
        2,
        6,
        7,
        1,
        3,
        4,
        6,
        8,
        0,
        5,
        7,
        0,
        3,
        4,
        7,
        0,
        1,
        4,
        7,
        8,
        1,
        4,
        1,
        3,
        4,
        6,
        7,
        0,
        5,
        6,
        7,
        8,
        1,
        3,
        4,
        6,
        7,
        8,
        2,
        3,
        4,
        5,
        8,
        0,
        4,
        5,
        8,
        1,
        2,
        3,
        4,
        7,
        2,
        3,
        8,
        2,
        5,
        6,
        7,
        0,
        3,
        5,
        0,
        1,
        3,
        4,
        8,
        1,
        2,
        4,
        7,
        0,
        3,
        4,
        6,
        7,
        8,
        2,
        7,
        8,
        1,
        3,
        4,
        5,
        7,
        8,
        2,
        4,
        6,
        7,
        0,
        4,
        6,
        7,
        1,
        2,
        5,
        7,
        8,
        1,
        2,
        3,
        7,
        2,
        3,
        4,
        5,
        7,
        8,
        2,
        3,
        5,
    ],
)


def test_partial_evidence_output_is_unchanged_by_refactor() -> None:
    rng = np.random.default_rng(123)
    big = sparse.csr_matrix((rng.random((40, 9)) < 0.6).astype(np.float32))
    initial = np.array([COLS[i % 7] for i in range(40)])
    out = partial_evidence(big, initial, COLS, extra=2, seed=7)
    assert (out.indptr.tolist(), out.indices.tolist()) == GOLDEN


def test_dropout_keeps_initial_demographics_and_whole_evidences() -> None:
    from beaconai.eval.partial import evidence_dropout

    for seed in range(30):
        row = evidence_dropout(X, INITIAL, COLS, (0.5, 0.95), seed=seed)[0].toarray()[0]
        assert row[0] == 1  # initial evidence E_1
        assert row[7] == pytest.approx(0.4) and row[8] == 1  # age and sex
        assert (row[3] == 0) == (row[4] == 0)  # E_4 kept or dropped as a whole


def test_fixed_rate_matches_on_average() -> None:
    from beaconai.eval.partial import evidence_dropout

    many = sparse.csr_matrix(np.tile(X[0].toarray(), (2000, 1)))
    out = evidence_dropout(many, np.array(["E_1"] * 2000), COLS, (0.75, 0.75), seed=11)
    kept_others = np.mean([len(_kept_codes(out[i])) - 1 for i in range(2000)])
    assert kept_others == pytest.approx(5 * 0.25, abs=0.1)  # 5 other evidences, keep 25%


def test_rate_range_spreads_evidence_counts() -> None:
    from beaconai.eval.partial import evidence_dropout

    many = sparse.csr_matrix(np.tile(X[0].toarray(), (4000, 1)))
    initial = np.array(["E_1"] * 4000)
    fixed = evidence_dropout(many, initial, COLS, (0.725, 0.725), seed=5)
    spread = evidence_dropout(many, initial, COLS, (0.5, 0.95), seed=5)
    count = lambda m: np.array([len(_kept_codes(m[i])) - 1 for i in range(4000)])  # noqa: E731
    # Same mean rate (0.725), but drawing a rate per row widens the spread of evidence counts.
    assert count(spread).mean() == pytest.approx(count(fixed).mean(), abs=0.1)
    assert count(spread).var() > count(fixed).var() * 1.1


def test_dropout_is_seeded_and_validates_range() -> None:
    from beaconai.eval.partial import evidence_dropout

    a = evidence_dropout(X, INITIAL, COLS, (0.5, 0.9), seed=3).toarray()
    np.testing.assert_array_equal(a, evidence_dropout(X, INITIAL, COLS, (0.5, 0.9), 3).toarray())
    for bad in [(0.0, 0.5), (0.5, 1.0), (0.9, 0.5)]:
        with pytest.raises(ValueError):
            evidence_dropout(X, INITIAL, COLS, bad, seed=0)

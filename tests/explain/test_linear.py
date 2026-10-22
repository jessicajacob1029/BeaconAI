import numpy as np
import pytest
import shap
from scipy import sparse

from beaconai.explain.linear import (
    baseline_logits,
    contrast_contributions,
    contributions,
    logits,
    top_contributors,
)

RNG = np.random.default_rng(0)
W = RNG.normal(size=(4, 7))
B = RNG.normal(size=4)
X = sparse.csr_matrix(
    np.array(
        [
            [1, 0, 0.6, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
)
# Columns 5 and 6 play the demographic role: baseline = their training mean; the rest 0.
M = np.array([0, 0, 0, 0, 0, 0.3, 0.55])


@pytest.mark.parametrize("m", [None, M])
def test_baseline_score_plus_contributions_equals_logit(m) -> None:
    for row in range(2):
        z = logits(W, B, X[row])
        base = baseline_logits(W, B, m)
        for c in range(4):
            total = sum(v for _, v in contributions(W, X[row], c, m))
            assert base[c] + total == pytest.approx(z[c])


def test_zero_baseline_only_present_features_contribute() -> None:
    assert {j for j, _ in contributions(W, X[0], 2)} == {0, 2, 4, 6}
    assert dict(contributions(W, X[0], 1))[2] == pytest.approx(W[1, 2] * 0.6, rel=1e-6)


def test_mean_baseline_includes_absent_demographic_columns() -> None:
    # Row 1 has column 6 = 0 (for example a female patient): it still contributes.
    pairs = dict(contributions(W, X[1], 0, M))
    assert set(pairs) == {1, 5, 6}
    assert pairs[6] == pytest.approx(W[0, 6] * (0 - 0.55))
    assert pairs[1] == pytest.approx(W[0, 1] * 1.0)  # evidence keeps the absent baseline


@pytest.mark.parametrize("m", [np.zeros(7), M])
def test_matches_shap_linear_explainer_with_the_same_baseline(m) -> None:
    dense = X.toarray()
    for c in range(4):
        explainer = shap.LinearExplainer(
            (W[c], B[c]), shap.maskers.Independent(m[None, :], max_samples=1)
        )
        expected = np.asarray(explainer.shap_values(dense))
        for row in range(dense.shape[0]):
            ours = np.zeros(W.shape[1])
            for j, v in contributions(W, X[row], c, m):
                ours[j] = v
            np.testing.assert_allclose(ours, expected[row], atol=1e-6)


def test_top_contributors_order_and_no_padding() -> None:
    pairs = [(0, 2.0), (1, -0.5), (2, 0.0), (3, 3.0), (4, -1.5)]
    pos, neg = top_contributors(pairs, n_pos=5, n_neg=3)
    assert pos == [(3, 3.0), (0, 2.0)]  # zero excluded, only 2 positives exist
    assert neg == [(4, -1.5), (1, -0.5)]


def test_top_contributors_ties_by_feature_index() -> None:
    pos, _ = top_contributors([(5, 1.0), (2, 1.0)], 1, 0)
    assert pos == [(2, 1.0)]


@pytest.mark.parametrize("m", [None, M])
def test_contrast_is_difference_of_contributions(m) -> None:
    a, b = 0, 3
    by_a = dict(contributions(W, X[0], a, m))
    by_b = dict(contributions(W, X[0], b, m))
    for j, v in contrast_contributions(W, X[0], a, b, m):
        assert v == pytest.approx(by_a[j] - by_b[j])
    z = logits(W, B, X[0])
    base = baseline_logits(W, B, m)
    total = sum(v for _, v in contrast_contributions(W, X[0], a, b, m))
    assert (base[a] - base[b]) + total == pytest.approx(z[a] - z[b])


def test_empty_row_with_zero_baseline_has_no_contributions() -> None:
    empty = sparse.csr_matrix((1, 7), dtype=np.float32)
    assert contributions(W, empty, 0) == []

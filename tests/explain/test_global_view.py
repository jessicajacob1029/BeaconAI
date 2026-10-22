from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from beaconai.explain.global_view import global_top_features, ranking_change_rates
from beaconai.explain.labels import FeatureLabeler

FEATURES = [
    {"index": 0, "name": "E_0", "kind": "binary", "label": "Cough?"},
    {"index": 1, "name": "E_1", "kind": "binary", "label": "Fever?"},
    {"index": 2, "name": "E_2", "kind": "binary", "label": "Rare thing?"},
    {"index": 3, "name": "SEX_M", "kind": "sex", "label": "Sex: male"},
]
LABELS = [{"name": "Flu"}, {"name": "Cold"}]
W = np.array([[2.0, 1.0, 3.0, 0.5], [-1.0, 0.5, -3.0, -0.5]])
X = sparse.csr_matrix(
    np.array([[1, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
)
Y = np.array([0, 0, 1, 1])
LAB = FeatureLabeler(FEATURES, 10, 110)


def test_top_features_sorted_with_prevalence_and_flags() -> None:
    out = global_top_features(W, Y, X, LABELS, LAB, ["Flu"], n=3, min_prevalence=0.01)
    top = out["Flu"]
    assert [t.name for t in top] == ["E_2", "E_0", "E_1"]
    rare = top[0]
    assert rare.prevalence_condition == 0.0 and rare.prevalence_all == 0.25
    assert "rarely seen with this condition" in rare.flags
    assert "less common with this condition than overall" in rare.flags
    assert top[1].flags == []  # cough: 100% of flu vs 50% overall


def test_only_positive_coefficients_listed() -> None:
    out = global_top_features(W, Y, X, LABELS, LAB, ["Cold"], n=4, min_prevalence=0.01)
    assert [t.name for t in out["Cold"]] == ["E_1"]  # the only positive coefficient for Cold


def test_demographic_flag() -> None:
    out = global_top_features(W, Y, X, LABELS, LAB, ["Flu"], n=4, min_prevalence=0.01)
    sex = next(t for t in out["Flu"] if t.name == "SEX_M")
    assert "demographic" in sex.flags


def test_unknown_condition_is_an_error() -> None:
    with pytest.raises(ValueError, match="Measles"):
        global_top_features(W, Y, X, LABELS, LAB, ["Flu", "Measles"], 3, 0.01)


def test_ranking_change_rates() -> None:
    raw = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])
    cal = np.array([[0.6, 0.3, 0.1], [0.4, 0.5, 0.1], [0.3, 0.1, 0.6]])
    rates = ranking_change_rates(raw, cal, k=2)
    # Row 2: top-1 changes (0 -> 1), top-2 set stays {0, 1}.
    # Row 3: top-1 stays 2, top-2 set changes ({2, 1} -> {2, 0}).
    assert rates["top1_changed"] == pytest.approx(1 / 3)
    assert rates["topk_set_changed"] == pytest.approx(1 / 3)


def test_plot_top_features_writes_png(tmp_path: Path) -> None:
    from beaconai.eval.plots import plot_top_features

    top = global_top_features(W, Y, X, LABELS, LAB, ["Flu"], n=4, min_prevalence=0.01)["Flu"]
    path = plot_top_features(top, "Flu", tmp_path / "flu.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

import numpy as np
import pytest
from scipy import sparse

from beaconai.eval.metrics import (
    confusion_matrix,
    ddx_overlap_at_k,
    evaluate_predictions,
    macro_f1,
    per_class_recall,
    top_confusions,
    top_k_accuracy,
    top_k_indices,
)

# 4 patients, 4 classes.
PROBA = np.array(
    [
        [0.70, 0.20, 0.05, 0.05],  # true 0 -> top-1 hit
        [0.10, 0.20, 0.60, 0.10],  # true 1 -> top-2 hit
        [0.40, 0.30, 0.20, 0.10],  # true 3 -> rank 4
        [0.05, 0.05, 0.10, 0.80],  # true 3 -> top-1 hit
    ]
)
Y = np.array([0, 1, 3, 3])


def test_top_k_accuracy() -> None:
    assert top_k_accuracy(Y, PROBA, 1) == pytest.approx(0.5)
    assert top_k_accuracy(Y, PROBA, 2) == pytest.approx(0.75)
    assert top_k_accuracy(Y, PROBA, 4) == pytest.approx(1.0)


def test_top_k_ties_prefer_lower_class_id() -> None:
    uniform = np.full((1, 4), 0.25)
    assert top_k_indices(uniform, 2).tolist() == [[0, 1]]


def test_top_k_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        top_k_accuracy(Y, PROBA, 5)  # k > n_classes
    with pytest.raises(ValueError):
        top_k_accuracy(Y[:3], PROBA, 1)  # length mismatch
    with pytest.raises(ValueError):
        top_k_accuracy(np.array([0, 1, 3, 9]), PROBA, 1)  # label out of range


def test_confusion_and_recall() -> None:
    y_pred = PROBA.argmax(axis=1)  # [0, 2, 0, 3]
    cm = confusion_matrix(Y, y_pred, 4)
    assert cm.shape == (4, 4)
    assert cm[1, 2] == 1 and cm[3, 0] == 1 and cm[3, 3] == 1
    recall = per_class_recall(cm)
    assert recall[0] == 1.0 and recall[1] == 0.0 and recall[3] == 0.5
    assert np.isnan(recall[2])  # no true examples of class 2


def test_macro_f1_ignores_classes_absent_everywhere() -> None:
    # Classes 2 and 3 never appear; perfect predictions on 0 and 1 give F1 1.0.
    assert macro_f1(np.array([0, 1, 1]), np.array([0, 1, 1])) == pytest.approx(1.0)


def test_top_confusions_ranks_off_diagonal() -> None:
    cm = np.array([[5, 3, 0], [1, 4, 0], [0, 7, 2]])
    assert top_confusions(cm, 2) == [(2, 1, 7), (0, 1, 3)]
    assert len(top_confusions(cm, 10)) == 3  # zero cells are skipped


def test_ddx_overlap() -> None:
    # Reference differential: patient 0 -> {0, 1}; patient 1 -> {1, 2, 3}.
    ddx = sparse.csr_matrix(
        np.array([[0.5, 0.5, 0, 0], [0, 0.4, 0.3, 0.3], [0, 0, 0, 0], [0, 0, 0, 1.0]])
    )
    out = ddx_overlap_at_k(PROBA[:2], ddx[:2], 2)
    # Top-2: p0 = {0, 1} -> 2 hits; p1 = {2, 1} -> 2 hits.
    assert out["precision"] == pytest.approx((2 / 2 + 2 / 2) / 2)
    assert out["recall"] == pytest.approx((2 / 2 + 2 / 3) / 2)
    assert out["recall_norm"] == pytest.approx((2 / 2 + 2 / 2) / 2)
    assert out["n_rows"] == 2


def test_ddx_overlap_small_reference_differential() -> None:
    # |D| = 1 < k = 3: one hit fills the only slot that can be filled.
    ddx = sparse.csr_matrix(np.array([[1.0, 0, 0, 0]]))
    out = ddx_overlap_at_k(PROBA[:1], ddx, 3)
    assert out["precision"] == pytest.approx(1 / 3)
    assert out["recall"] == pytest.approx(1.0)
    assert out["recall_norm"] == pytest.approx(1.0)


def test_ddx_overlap_partial_hits() -> None:
    # Patient 1 top-3 = {2, 1, 0}; D = {3} gives 0 hits; patient 0 top-3 = {0, 1, 2}; D = {3, 1}.
    ddx = sparse.csr_matrix(np.array([[0, 0.5, 0, 0.5], [0, 0, 0, 1.0]]))
    out = ddx_overlap_at_k(PROBA[:2], ddx, 3)
    assert out["precision"] == pytest.approx((1 / 3 + 0) / 2)
    assert out["recall"] == pytest.approx((1 / 2 + 0) / 2)
    assert out["recall_norm"] == pytest.approx((1 / 2 + 0) / 2)


def test_ddx_overlap_ignores_stored_zeros() -> None:
    # An explicitly stored 0.0 is not part of the reference differential.
    ddx = sparse.csr_matrix(
        (np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 2])), shape=(1, 4)
    )
    out = ddx_overlap_at_k(PROBA[:1], ddx, 2)
    assert out["recall"] == pytest.approx(1.0)  # D = {0}, not {0, 1}
    assert out["precision"] == pytest.approx(1 / 2)


def test_ddx_overlap_skips_empty_rows() -> None:
    ddx = sparse.csr_matrix(np.array([[1.0, 0, 0, 0], [0, 0, 0, 0]]))
    out = ddx_overlap_at_k(PROBA[:2], ddx, 1)
    assert out["n_rows"] == 1
    assert out["precision"] == pytest.approx(1.0)


def test_ddx_overlap_all_empty_is_nan() -> None:
    out = ddx_overlap_at_k(PROBA[:1], sparse.csr_matrix((1, 4)), 1)
    assert out["n_rows"] == 0
    assert np.isnan(out["precision"]) and np.isnan(out["recall_norm"])


def test_ddx_overlap_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ddx_overlap_at_k(PROBA, sparse.csr_matrix((3, 4)), 1)


def test_evaluate_predictions_keys() -> None:
    ddx = sparse.csr_matrix(np.eye(4)[Y])
    res = evaluate_predictions(Y, PROBA, ddx, ks=[1, 3], ddx_ks=[3])
    assert set(res.metrics) == {
        "top1_accuracy",
        "top3_accuracy",
        "macro_f1",
        "ddx_precision_at3",
        "ddx_recall_at3",
        "ddx_recall_norm_at3",
    }
    assert res.confusion.shape == (4, 4)


def test_evaluate_predictions_without_ddx() -> None:
    res = evaluate_predictions(Y, PROBA, None, ks=[1], ddx_ks=[3])
    assert "ddx_precision_at3" not in res.metrics


def test_top_k_tie_break_uses_secondary_scores() -> None:
    proba = np.array([[0.9, 0.0, 0.0, 0.1]])  # classes 1 and 2 tie at 0
    raw = np.array([[0.8, 0.05, 0.15, 0.0]])
    assert top_k_indices(proba, 3, tie_break=raw).tolist() == [[0, 3, 2]]
    assert top_k_indices(proba, 3).tolist() == [[0, 3, 1]]  # without: lower class id


def test_top_k_tie_break_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        top_k_indices(np.ones((1, 3)), 2, tie_break=np.ones((1, 4)))


def test_top_k_accuracy_with_tie_break() -> None:
    proba = np.array([[0.9, 0.05, 0.05]])  # classes 1 and 2 tie
    raw = np.array([[0.8, 0.05, 0.15]])
    y = np.array([2])
    assert top_k_accuracy(y, proba, 2) == 0.0  # label order picks class 1
    assert top_k_accuracy(y, proba, 2, tie_break=raw) == 1.0  # served order picks class 2


def test_evaluate_predictions_threads_tie_break() -> None:
    proba = np.array([[0.9, 0.05, 0.05]])
    raw = np.array([[0.8, 0.05, 0.15]])
    ddx = sparse.csr_matrix(np.array([[0.5, 0.0, 0.5]]))
    res = evaluate_predictions(np.array([2]), proba, ddx, ks=[2], ddx_ks=[2], tie_break=raw)
    assert res.metrics["top2_accuracy"] == 1.0
    assert res.metrics["ddx_precision_at2"] == 1.0  # {0, 2} both in the reference

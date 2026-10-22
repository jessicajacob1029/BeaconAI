from pathlib import Path

import numpy as np
import pytest

from beaconai.models.dataset import load_labels, load_manifest, load_split
from tests.conftest import TOY_CLASSES, TOY_FEATURES, TOY_ROWS


def test_load_split_shapes_and_types(toy_processed: Path) -> None:
    data = load_split(toy_processed, "train")
    assert data.X.shape == (TOY_ROWS["train"], TOY_FEATURES)
    assert data.y.dtype == np.int64
    assert data.ddx is not None and data.ddx.shape == (TOY_ROWS["train"], TOY_CLASSES)
    assert data.n == TOY_ROWS["train"]


def test_load_split_without_ddx(toy_processed: Path) -> None:
    (toy_processed / "test_ddx.npz").unlink()
    assert load_split(toy_processed, "test").ddx is None


def test_labels_and_manifest(toy_processed: Path) -> None:
    assert load_labels(toy_processed)[0]["name"] == "Condition 0"
    assert load_manifest(toy_processed)["n_classes"] == TOY_CLASSES


def test_missing_split_says_run_make_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_split(tmp_path, "train")


def test_feature_names_in_column_order(toy_processed: Path) -> None:
    from beaconai.models.dataset import load_feature_names

    assert load_feature_names(toy_processed) == [f"F_{i}" for i in range(TOY_FEATURES)]


def test_subset_keeps_rows_aligned(toy_processed: Path) -> None:
    from beaconai.models.dataset import subset

    data = load_split(toy_processed, "train")
    idx = np.array([4, 0, 7])
    part = subset(data, idx)
    assert part.n == 3
    np.testing.assert_array_equal(part.y, data.y[idx])
    np.testing.assert_array_equal(part.X.toarray(), data.X[idx].toarray())
    assert part.ddx is not None and data.ddx is not None
    np.testing.assert_array_equal(part.ddx.toarray(), data.ddx[idx].toarray())


def test_load_meta_and_features(toy_processed: Path) -> None:
    from beaconai.models.dataset import load_features, load_meta

    meta = load_meta(toy_processed, "validate")
    assert len(meta["initial_evidence"]) == TOY_ROWS["validate"]
    assert meta["initial_evidence"][1] == "E_2"  # row 1 is class 1, which always has F_2
    assert [f["evidence"] for f in load_features(toy_processed)][:2] == ["E_0", "E_1"]


def test_load_meta_missing_says_run_make_data(tmp_path: Path) -> None:
    from beaconai.models.dataset import load_meta

    with pytest.raises(FileNotFoundError, match="make data"):
        load_meta(tmp_path, "validate")


def test_augment_with_dropout_stacks_original_and_copies(toy_processed: Path) -> None:
    from beaconai.eval.partial import column_evidence
    from beaconai.models.dataset import augment_with_dropout, load_features, load_meta

    data = load_split(toy_processed, "train")
    initial = load_meta(toy_processed, "train")["initial_evidence"]
    cols = column_evidence(load_features(toy_processed))
    aug = augment_with_dropout(data, initial, cols, (0.5, 0.95), 2, True, seed=0)
    n = data.n
    assert aug.n == 3 * n and aug.ddx is None
    np.testing.assert_array_equal(aug.y, np.tile(data.y, 3))
    np.testing.assert_array_equal(aug.X[:n].toarray(), data.X.toarray())  # originals first
    assert aug.X[n:].nnz < 2 * data.X.nnz  # copies lost evidence
    assert not np.array_equal(aug.X[n : 2 * n].toarray(), aug.X[2 * n :].toarray())
    assert augment_with_dropout(data, initial, cols, (0.5, 0.95), 1, False, 0).n == n
    with pytest.raises(ValueError, match="no training rows"):
        augment_with_dropout(data, initial, cols, (0.5, 0.95), 0, False, 0)


def test_split_indices_still_importable_from_calibrate() -> None:
    from beaconai.models.calibrate import split_indices as a
    from beaconai.models.dataset import split_indices as b

    assert a is b

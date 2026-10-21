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

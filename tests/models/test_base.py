from pathlib import Path

import numpy as np
import pytest

from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import MODEL_REGISTRY, build_model


def test_registry_names() -> None:
    assert set(MODEL_REGISTRY) >= {"logreg"}


def test_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("svm", {}, 3, 0)


def test_predict_before_fit_raises(toy_processed: Path) -> None:
    model = build_model("logreg", {}, 3, 0)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(load_split(toy_processed, "test").X)


def test_feature_count_mismatch_raises(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("logreg", {}, 3, 0).fit(train.X, train.y)
    with pytest.raises(ValueError, match="features"):
        model.predict_proba(train.X[:, :4])


def test_label_out_of_range_raises(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    with pytest.raises(ValueError, match="labels"):
        build_model("logreg", {}, 2, 0).fit(train.X, train.y)  # toy has 3 classes


def test_val_x_without_y_raises(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    with pytest.raises(ValueError, match="together"):
        build_model("logreg", {}, 3, 0).fit(train.X, train.y, X_val=train.X)


def test_save_unfitted_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unfitted"):
        build_model("logreg", {}, 3, 0).save(tmp_path / "m")


def test_load_unknown_name_raises(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text('{"name": "svm"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown model"):
        ModelWrapper.load(tmp_path)


def test_predict_is_argmax(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("logreg", {}, 3, 0).fit(train.X, train.y)
    np.testing.assert_array_equal(model.predict(train.X), model.predict_proba(train.X).argmax(1))

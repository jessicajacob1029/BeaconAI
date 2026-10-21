from pathlib import Path

import numpy as np

from beaconai.eval.metrics import top_k_accuracy
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import build_model


def _fit(toy_processed: Path) -> ModelWrapper:
    train = load_split(toy_processed, "train")
    return build_model("logreg", {"C": 1.0, "max_iter": 200}, 3, 42).fit(train.X, train.y)


def test_logreg_learns_toy_task(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    test = load_split(toy_processed, "test")
    proba = model.predict_proba(test.X)
    assert proba.shape == (test.n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert top_k_accuracy(test.y, proba, 1) > 0.9


def test_logreg_handles_missing_class(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    keep = train.y != 2
    model = build_model("logreg", {}, 3, 42).fit(train.X[keep], train.y[keep])
    proba = model.predict_proba(train.X)
    assert proba.shape[1] == 3
    assert np.all(proba[:, 2] == 0.0)


def test_logreg_save_load_round_trip(toy_processed: Path, tmp_path: Path) -> None:
    model = _fit(toy_processed)
    model.save(tmp_path / "m")
    loaded = ModelWrapper.load(tmp_path / "m")
    X = load_split(toy_processed, "test").X
    assert loaded.name == "logreg" and loaded.params == model.params
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))

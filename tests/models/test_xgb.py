from pathlib import Path

import numpy as np

from beaconai.eval.metrics import top_k_accuracy
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import build_model

PARAMS = {"n_estimators": 50, "early_stopping_rounds": 3, "max_depth": 2, "tree_method": "hist"}


def _fit(toy_processed: Path, params: dict = PARAMS) -> ModelWrapper:
    train, val = load_split(toy_processed, "train"), load_split(toy_processed, "validate")
    return build_model("xgboost", params, 3, 42).fit(train.X, train.y, val.X, val.y)


def test_xgb_learns_toy_task(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    test = load_split(toy_processed, "test")
    proba = model.predict_proba(test.X)
    assert proba.shape == (test.n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert top_k_accuracy(test.y, proba, 1) > 0.9


def test_xgb_early_stops_on_validation(toy_processed: Path) -> None:
    # A very high learning rate makes validation loss plateau fast, so early stopping must fire.
    model = _fit(toy_processed, {**PARAMS, "n_estimators": 200, "learning_rate": 1.0})
    assert model.best_iteration < 199


def test_xgb_without_validation_uses_all_rounds(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("xgboost", PARAMS, 3, 42).fit(train.X, train.y)
    assert model.best_iteration == PARAMS["n_estimators"] - 1


def test_xgb_handles_missing_class(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    keep = train.y != 2
    model = build_model("xgboost", PARAMS, 3, 42).fit(train.X[keep], train.y[keep])
    assert model.predict_proba(train.X).shape[1] == 3


def test_xgb_save_load_round_trip(toy_processed: Path, tmp_path: Path) -> None:
    model = _fit(toy_processed, {**PARAMS, "n_estimators": 200, "learning_rate": 1.0})
    model.save(tmp_path / "m")
    loaded = ModelWrapper.load(tmp_path / "m")
    X = load_split(toy_processed, "test").X
    assert loaded.best_iteration == model.best_iteration
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)

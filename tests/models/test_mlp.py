from pathlib import Path

import numpy as np

from beaconai.eval.metrics import top_k_accuracy
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import load_split
from beaconai.models.registry import build_model

PARAMS = {
    "hidden_sizes": [16],
    "dropout": 0.0,
    "lr": 0.01,
    "weight_decay": 0.0,
    "epochs": 40,
    "batch_size": 32,
    "patience": 3,
}


def _fit(toy_processed: Path, seed: int = 42) -> ModelWrapper:
    train, val = load_split(toy_processed, "train"), load_split(toy_processed, "validate")
    return build_model("mlp", PARAMS, 3, seed).fit(train.X, train.y, val.X, val.y)


def test_mlp_learns_toy_task(toy_processed: Path) -> None:
    model = _fit(toy_processed)
    test = load_split(toy_processed, "test")
    proba = model.predict_proba(test.X)
    assert proba.shape == (test.n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert top_k_accuracy(test.y, proba, 1) > 0.9


def test_mlp_is_reproducible(toy_processed: Path) -> None:
    X = load_split(toy_processed, "test").X
    np.testing.assert_allclose(
        _fit(toy_processed).predict_proba(X), _fit(toy_processed).predict_proba(X), rtol=1e-6
    )


def test_mlp_early_stops(toy_processed: Path) -> None:
    # The toy task is learned in a few epochs, so validation loss stops improving well before 40.
    assert _fit(toy_processed).epochs_trained < PARAMS["epochs"]


def test_mlp_without_validation_runs_all_epochs(toy_processed: Path) -> None:
    train = load_split(toy_processed, "train")
    model = build_model("mlp", {**PARAMS, "epochs": 4}, 3, 42).fit(train.X, train.y)
    assert model.epochs_trained == 4


def test_mlp_save_load_round_trip(toy_processed: Path, tmp_path: Path) -> None:
    model = _fit(toy_processed)
    model.save(tmp_path / "m")
    loaded = ModelWrapper.load(tmp_path / "m")
    X = load_split(toy_processed, "test").X
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)

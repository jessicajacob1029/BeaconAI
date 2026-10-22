import json
from pathlib import Path

import mlflow
import numpy as np
import pytest

from beaconai.models.dataset import load_features, load_meta, load_split, split_indices
from beaconai.models.train import DataScopeError, configure_mlflow, load_wrapper, run_training
from beaconai.models.train_dropout import early_stopping_rows, run_dropout_training
from tests.conftest import make_config, write_toy_processed


def _setup(tmp_path: Path):
    cfg = make_config(tmp_path, dev_sample=None)
    processed = write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    return cfg, processed


def test_early_stopping_rows_come_only_from_the_fit_half(tmp_path: Path) -> None:
    cfg, processed = _setup(tmp_path)
    val = load_split(processed, "validate")
    meta = load_meta(processed, "validate")
    stop, idx = early_stopping_rows(val, meta, load_features(processed), cfg)
    fit_idx, held_idx = split_indices(val.y, cfg.calibration.fit_fraction, cfg.seed)
    np.testing.assert_array_equal(idx, fit_idx)
    assert set(idx).isdisjoint(held_idx)
    assert stop.n == len(fit_idx)
    assert stop.X.nnz <= val.X[fit_idx].nnz
    np.testing.assert_array_equal(stop.y, val.y[fit_idx])


def test_run_dropout_training(tmp_path: Path) -> None:
    cfg, processed = _setup(tmp_path)
    summary = run_dropout_training(cfg)
    assert set(summary["models"]) == {"logreg", "xgboost"}
    assert summary["n_train_rows"] == 2 * 150  # original + 1 copy
    assert summary["rate_range"] == [0.5, 0.95]
    reports = tmp_path / "reports"
    assert json.loads((reports / "train_dropout_summary.json").read_text()) == summary
    model = load_wrapper(summary["models"]["xgboost"]["model_uri"])
    assert model.predict_proba(load_split(processed, "test").X).shape == (60, 3)
    train_params = json.loads((reports / "train_summary.json").read_text())["models"]
    assert summary["models"]["logreg"]["params"] == train_params["logreg"]["params"]
    configure_mlflow(cfg.mlflow, experiment=cfg.dropout.experiment)
    runs = mlflow.search_runs(experiment_names=[cfg.dropout.experiment])
    assert len(runs) == 3  # parent + 2 models
    child = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert (child["params.dropout_rate_range"] == "[0.5, 0.95]").all()


def test_refuses_on_data_scope_mismatch(tmp_path: Path) -> None:
    cfg, processed = _setup(tmp_path)
    manifest = json.loads((processed / "manifest.json").read_text())
    manifest["settings"]["dev_sample"] = 50000
    (processed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataScopeError):
        run_dropout_training(cfg)


def test_missing_train_summary(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    with pytest.raises(FileNotFoundError, match="make train"):
        run_dropout_training(cfg)

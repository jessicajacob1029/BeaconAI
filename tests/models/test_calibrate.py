import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from beaconai.models.calibrate import (
    CalibrationResult,
    load_calibrated,
    run_calibration,
    select_calibrated,
    split_indices,
)
from beaconai.models.calibrated import Abstain, Differential
from beaconai.models.dataset import load_split
from beaconai.models.train import DataScopeError, configure_mlflow, run_training
from tests.conftest import make_config, write_toy_processed


def test_split_indices_disjoint_cover_and_stratified() -> None:
    y = np.repeat([0, 1, 2], [100, 40, 1])
    fit, held = split_indices(y, 0.5, seed=0)
    assert set(fit).isdisjoint(held)
    assert sorted([*fit, *held]) == list(range(len(y)))
    assert np.sum(y[fit] == 0) == 50 and np.sum(y[fit] == 1) == 20
    assert 140 in held  # a class with one patient goes to the held-out part


def test_split_indices_seeded() -> None:
    y = np.arange(200) % 4
    a, b = split_indices(y, 0.5, 3), split_indices(y, 0.5, 3)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], split_indices(y, 0.5, 4)[0])


def _r(model: str, method: str, partial_ece: float, top3: float, partial_top3: float):
    metrics = {
        "ece": 0.001,
        "top3_accuracy": top3,
        "partial_ece": partial_ece,
        "partial_top3_accuracy": partial_top3,
    }
    return CalibrationResult(model, method, "run", "uri", metrics, None)


def test_select_by_partial_ece_with_both_floors() -> None:
    results = [
        _r("xgboost", "none", 0.55, 1.0, 0.71),
        _r("logreg", "temperature", 0.03, 1.0, 0.77),
        _r("logreg", "isotonic", 0.02, 0.98, 0.77),  # best partial ECE, fails the full floor
        _r("xgboost", "isotonic", 0.01, 1.0, 0.60),  # fails the partial floor
    ]
    best = select_calibrated(results, floor_full=0.99, floor_partial=0.65)
    assert (best.model, best.method) == ("logreg", "temperature")


def test_select_ties_keep_order() -> None:
    results = [_r("a", "none", 0.1, 1.0, 0.8), _r("b", "none", 0.1, 1.0, 0.8)]
    assert select_calibrated(results, 0.9, 0.5).model == "a"


def test_select_none_eligible() -> None:
    with pytest.raises(ValueError, match="floor"):
        select_calibrated([_r("logreg", "none", 0.01, 0.5, 0.9)], 0.99, 0.65)


def _trained(tmp_path: Path, dev_sample: int | None):
    cfg = make_config(tmp_path, dev_sample=dev_sample)
    write_toy_processed(tmp_path / "processed", dev_sample=dev_sample)
    run_training(cfg)
    return cfg


def test_run_calibration_full_data(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, None)
    summary = run_calibration(cfg)

    assert len(summary["candidates"]) == 6  # 2 models x 3 methods
    assert summary["registered"]["alias"] == "champion"
    assert summary["n_fit"] + summary["n_held"] == 60
    assert summary["top3_floor"] == {"full": 0.5, "partial": 0.0}
    eligible = [
        c
        for c in summary["candidates"]
        if c["metrics"]["top3_accuracy"] >= 0.5 and c["metrics"]["partial_top3_accuracy"] >= 0.0
    ]
    best = min(c["metrics"]["partial_ece"] for c in eligible)
    assert summary["champion"]["metrics"]["partial_ece"] == best

    reports = tmp_path / "reports"
    assert json.loads((reports / "calibration_summary.json").read_text()) == summary
    md = (reports / "calibration.md").read_text(encoding="utf-8")
    assert "Champion:" in md and "Limitation: not asked is read as no" in md
    assert (reports / "figures" / "reliability_xgboost.png").is_file()
    assert (reports / "figures" / "coverage_accuracy.png").is_file()

    configure_mlflow(cfg.mlflow, experiment=cfg.calibration.experiment)
    runs = mlflow.search_runs(experiment_names=[cfg.calibration.experiment])
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert len(children) == 6
    assert children["metrics.ece"].notna().all()
    assert children["metrics.partial_ece"].notna().all()
    mv = MlflowClient().get_model_version_by_alias("beaconai-classifier", "champion")
    assert int(mv.version) == summary["registered"]["version"]
    assert mv.tags["calibration"] == summary["champion"]["method"]

    clf = load_calibrated(summary["champion"]["model_uri"])
    preds = clf.predictions(load_split(tmp_path / "processed" / "ddxplus", "test").X)
    assert all(isinstance(p, Differential | Abstain) for p in preds)


def test_dev_run_writes_dev_outputs_and_dev_alias(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, 50000)
    summary = run_calibration(cfg)
    assert summary["registered"]["alias"] == "dev"
    assert (tmp_path / "reports" / "calibration_dev.md").is_file()
    assert not (tmp_path / "reports" / "calibration.md").exists()


def test_refuses_when_data_on_disk_differs(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, None)
    manifest_path = tmp_path / "processed" / "ddxplus" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["settings"]["dev_sample"] = 50000
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataScopeError):
        run_calibration(cfg)


def test_missing_train_summary_says_run_make_train(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    with pytest.raises(FileNotFoundError, match="make train"):
        run_calibration(cfg)


def test_retraining_after_calibration_keeps_calibrated_champion(tmp_path: Path) -> None:
    cfg = _trained(tmp_path, None)
    calibrated = run_calibration(cfg)
    retrained = run_training(cfg, winners_only=True)
    configure_mlflow(cfg.mlflow)
    client = MlflowClient()
    champ = client.get_model_version_by_alias("beaconai-classifier", "champion")
    assert int(champ.version) == calibrated["registered"]["version"]
    raw = client.get_model_version_by_alias("beaconai-classifier", "raw-champion")
    assert int(raw.version) == retrained["registered"]["version"]

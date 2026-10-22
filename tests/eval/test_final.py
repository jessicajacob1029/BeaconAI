import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from beaconai.eval.final import FinalEvalError, check_final_allowed, lock_path, run_final_evaluation
from beaconai.models.calibrate import run_calibration
from beaconai.models.train import configure_mlflow, run_training
from tests.conftest import make_config, write_toy_processed

CALIBRATED = {"calibration": "temperature", "dev_sample": "None", "phase": "3b"}


def test_guard_rules(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    check_final_allowed(None, CALIBRATED, lock)  # allowed
    with pytest.raises(FinalEvalError, match="full dataset"):
        check_final_allowed(50000, CALIBRATED, lock)
    with pytest.raises(FinalEvalError, match="calibrated"):
        check_final_allowed(None, {"model_type": "xgboost", "dev_sample": "None"}, lock)
    with pytest.raises(FinalEvalError, match="dev subsample"):
        check_final_allowed(None, {**CALIBRATED, "dev_sample": "50000"}, lock)
    lock.write_text("{}")
    with pytest.raises(FinalEvalError, match="already"):
        check_final_allowed(None, CALIBRATED, lock)


def test_lock_is_per_evaluation_name(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    assert lock_path(cfg).name == "classifier_final.json"
    assert lock_path(cfg).parent.name == "test_locks"


def _pipeline(tmp_path: Path):
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    run_calibration(cfg)
    return cfg


def test_final_evaluation_scores_the_champion_once(tmp_path: Path) -> None:
    cfg = _pipeline(tmp_path)
    # Another evaluation's lock does not block this one.
    other = tmp_path / "reports" / "test_locks" / "agent_simulation.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}")
    report = run_final_evaluation(cfg)
    md = report.read_text(encoding="utf-8")
    assert "decision support, not diagnosis" in md
    assert "Partial evidence" in md and "Abstention" in md
    assert "validated on validation data only" in md
    data = json.loads((tmp_path / "reports" / "final_evaluation.json").read_text())
    assert data["registered"]["alias"] == "champion"
    assert data["evaluation"] == "classifier_final"
    assert {"full", "partial"} <= set(data["metrics"])
    lock = json.loads(lock_path(cfg).read_text())
    assert lock["evaluation"] == "classifier_final"
    assert lock["model_version"] == data["registered"]["version"]
    with pytest.raises(FinalEvalError, match="already"):
        run_final_evaluation(cfg)


def test_final_evaluation_refuses_a_raw_model_at_the_alias(tmp_path: Path) -> None:
    cfg = _pipeline(tmp_path)
    configure_mlflow(cfg.mlflow)
    client = MlflowClient()
    raw = client.get_model_version_by_alias("beaconai-classifier", "raw-champion")
    client.set_registered_model_alias("beaconai-classifier", "champion", raw.version)
    with pytest.raises(FinalEvalError, match="calibrated"):
        run_final_evaluation(cfg)
    assert not lock_path(cfg).exists()


def test_cli_refusal_exits_2(tmp_path: Path, monkeypatch, capsys) -> None:
    import beaconai.eval.final as final

    cfg = make_config(tmp_path, dev_sample=50000)
    write_toy_processed(tmp_path / "processed", dev_sample=50000)
    run_training(cfg)
    run_calibration(cfg)  # dev run: moves `dev`, not `champion`
    monkeypatch.setattr(final, "load_config", lambda: cfg)
    assert final.main() == 2
    assert "Refused" in capsys.readouterr().err


def test_lock_is_written_before_scoring_so_a_crash_cannot_allow_a_rerun(
    tmp_path: Path, monkeypatch
) -> None:
    import beaconai.eval.final as final

    cfg = _pipeline(tmp_path)

    def boom(summary):
        raise RuntimeError("report failed after test predictions were made")

    monkeypatch.setattr(final, "render_final", boom)
    with pytest.raises(RuntimeError, match="report failed"):
        run_final_evaluation(cfg)
    lock = json.loads(lock_path(cfg).read_text())
    assert lock["status"] == "started"
    monkeypatch.undo()
    with pytest.raises(FinalEvalError, match="already"):
        run_final_evaluation(cfg)


def test_completed_lock_records_status(tmp_path: Path) -> None:
    cfg = _pipeline(tmp_path)
    run_final_evaluation(cfg)
    assert json.loads(lock_path(cfg).read_text())["status"] == "completed"

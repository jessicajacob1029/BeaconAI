import json
from pathlib import Path

import pytest

from beaconai.eval.evaluate import TestGuardError, run_evaluation
from beaconai.models.train import run_training
from tests.conftest import make_config, write_toy_processed


def test_validate_report_on_dev(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=50000)
    write_toy_processed(tmp_path / "processed", dev_sample=50000)
    run_training(cfg)
    md = run_evaluation(cfg, "validate").read_text(encoding="utf-8")
    assert "Test split not evaluated yet" in md
    assert "DDx nR@3" in md
    assert (tmp_path / "reports" / "figures" / "confusion_matrix_val.png").is_file()
    assert (tmp_path / "reports" / "figures" / "per_class_recall_val.png").is_file()
    with pytest.raises(TestGuardError):
        run_evaluation(cfg, "test")
    assert not (tmp_path / "reports" / "test_eval_lock.json").exists()


def test_unknown_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        run_evaluation(make_config(tmp_path, dev_sample=None), "train")


def test_missing_summary_says_run_make_train(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        run_evaluation(make_config(tmp_path, dev_sample=None), "validate")


def test_eval_refuses_when_data_on_disk_changed_since_training(tmp_path: Path) -> None:
    from beaconai.models.train import DataScopeError

    cfg = make_config(tmp_path, dev_sample=None)
    processed = write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    manifest = json.loads((processed / "manifest.json").read_text())
    manifest["settings"]["dev_sample"] = 50000  # e.g. `make data` restored the dev subsample
    (processed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataScopeError, match=r"make data|preprocess"):
        run_evaluation(cfg, "validate")


def test_cli_reports_data_scope_mismatch_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import beaconai.eval.evaluate as evaluate

    cfg = make_config(tmp_path, dev_sample=None)
    processed = write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    manifest = json.loads((processed / "manifest.json").read_text())
    manifest["settings"]["dev_sample"] = 50000
    (processed / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(evaluate, "load_config", lambda: cfg)
    monkeypatch.setattr("sys.argv", ["evaluate", "--split", "validate"])
    assert evaluate.main() == 2
    assert "Refused" in capsys.readouterr().err


def test_test_split_is_refused_here(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    with pytest.raises(TestGuardError, match="make eval"):
        run_evaluation(cfg, "test")

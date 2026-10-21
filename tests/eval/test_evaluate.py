import json
from pathlib import Path

import pytest

from beaconai.eval.evaluate import TestGuardError, check_test_allowed, run_evaluation
from beaconai.models.train import run_training
from tests.conftest import make_config, write_toy_processed


def test_guard_blocks_dev_sample(tmp_path: Path) -> None:
    with pytest.raises(TestGuardError, match="full dataset"):
        check_test_allowed({"dev_sample": 50000, "parent_run_id": "r"}, tmp_path / "l.json", False)


def test_guard_blocks_second_run(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"train_parent_run_id": "old"}), encoding="utf-8")
    with pytest.raises(TestGuardError, match="already"):
        check_test_allowed({"dev_sample": None, "parent_run_id": "new"}, lock, False)


def test_guard_allows_first_full_run(tmp_path: Path) -> None:
    check_test_allowed({"dev_sample": None, "parent_run_id": "r"}, tmp_path / "l.json", False)


def test_guard_force_overrides(tmp_path: Path) -> None:
    check_test_allowed({"dev_sample": 50000, "parent_run_id": "r"}, tmp_path / "l.json", True)


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


def test_test_eval_runs_once_on_full_data(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    md = run_evaluation(cfg, "test").read_text(encoding="utf-8")
    assert "Test split not evaluated yet" not in md
    assert md.count("| Model | Top-1") == 2  # validation and test tables
    assert (tmp_path / "reports" / "test_eval_lock.json").is_file()
    with pytest.raises(TestGuardError, match="already"):
        run_evaluation(cfg, "test")


def test_unknown_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        run_evaluation(make_config(tmp_path, dev_sample=None), "train")


def test_missing_summary_says_run_make_train(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        run_evaluation(make_config(tmp_path, dev_sample=None), "validate")


def test_validation_report_never_overwrites_test_report(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    test_report = run_evaluation(cfg, "test")
    figures = tmp_path / "reports" / "figures"
    before = {p.name: p.read_bytes() for p in [test_report, *figures.iterdir()]}
    val_report = run_evaluation(cfg, "validate")
    assert val_report != test_report
    assert "Test split not evaluated yet" in val_report.read_text(encoding="utf-8")
    assert test_report.read_bytes() == before[test_report.name]
    for name, data in before.items():
        if name != test_report.name:
            assert (figures / name).read_bytes() == data


def test_eval_refuses_when_data_on_disk_changed_since_training(tmp_path: Path) -> None:
    from beaconai.models.train import DataScopeError

    cfg = make_config(tmp_path, dev_sample=None)
    processed = write_toy_processed(tmp_path / "processed")
    run_training(cfg)
    manifest = json.loads((processed / "manifest.json").read_text())
    manifest["settings"]["dev_sample"] = 50000  # e.g. `make data` restored the dev subsample
    (processed / "manifest.json").write_text(json.dumps(manifest))
    for split in ("validate", "test"):
        with pytest.raises(DataScopeError, match=r"make data|preprocess"):
            run_evaluation(cfg, split)


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

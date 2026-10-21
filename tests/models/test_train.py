import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from beaconai.models.dataset import load_split
from beaconai.models.train import configure_mlflow, load_wrapper, run_training
from tests.conftest import make_config, write_toy_processed


def test_run_training_logs_registers_and_summarises(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=50000)
    processed = write_toy_processed(tmp_path / "processed")
    summary = run_training(cfg)

    # 2 logreg + 1 xgboost + 1 mlp trials, each its own nested run.
    assert {m: v["n_trials"] for m, v in summary["models"].items()} == {
        "logreg": 2,
        "xgboost": 1,
        "mlp": 1,
    }
    assert summary["mode"] == "grid" and summary["dev_sample"] == 50000
    assert summary["champion"] in summary["models"]
    written = json.loads((tmp_path / "reports" / "train_summary.json").read_text())
    assert written == summary

    configure_mlflow(cfg.mlflow)
    mv = MlflowClient().get_model_version_by_alias("beaconai-classifier", "champion")
    assert int(mv.version) == summary["registered"]["version"]
    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) == 5  # parent + 4 trials
    child = runs[runs["tags.model_type"] == "logreg"].iloc[0]
    assert not np.isnan(child["metrics.val_top3_accuracy"])
    assert not np.isnan(child["metrics.val_ddx_recall_norm_at3"])
    artifacts = {a.path for a in MlflowClient().list_artifacts(child["run_id"])}
    assert "val_confusion.png" in artifacts

    champ = summary["models"][summary["champion"]]
    model = load_wrapper(champ["model_uri"])
    test = load_split(processed, "test")
    assert model.predict_proba(test.X).shape == (test.n, 3)


def test_winners_only_reuses_best_params(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=50000)
    write_toy_processed(tmp_path / "processed")
    first = run_training(cfg)
    second = run_training(cfg, winners_only=True)
    assert second["mode"] == "winners"
    for name, info in second["models"].items():
        assert info["n_trials"] == 1
        assert info["params"] == first["models"][name]["params"]
    assert second["registered"]["version"] == first["registered"]["version"] + 1


def test_training_is_reproducible(tmp_path: Path) -> None:
    a = make_config(tmp_path / "a", dev_sample=None)
    b = make_config(tmp_path / "b", dev_sample=None)
    write_toy_processed(tmp_path / "a" / "processed")
    write_toy_processed(tmp_path / "b" / "processed")
    ma, mb = run_training(a)["models"], run_training(b)["models"]
    for name in ma:
        assert ma[name]["val_metrics"] == mb[name]["val_metrics"]


def test_logged_model_serves_named_feature_frames(tmp_path: Path) -> None:
    import pandas as pd

    from beaconai.models.dataset import load_feature_names

    cfg = make_config(tmp_path, dev_sample=None)
    processed = write_toy_processed(tmp_path / "processed")
    summary = run_training(cfg)
    uri = summary["models"][summary["champion"]]["model_uri"]
    configure_mlflow(cfg.mlflow)
    served = mlflow.pyfunc.load_model(uri)
    assert served.metadata.get_input_schema().input_names() == load_feature_names(processed)
    test = load_split(processed, "test")
    frame = pd.DataFrame(test.X.toarray(), columns=load_feature_names(processed))
    np.testing.assert_allclose(
        np.asarray(served.predict(frame)), load_wrapper(uri).predict_proba(test.X), rtol=1e-6
    )

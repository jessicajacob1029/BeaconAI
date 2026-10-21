from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from beaconai.models.calibrated import CalibratedClassifier
from beaconai.models.calibrators import TemperatureCalibrator
from beaconai.models.dataset import load_feature_names, load_labels, load_split
from beaconai.models.pyfunc import CalibratedPyfunc, log_pyfunc
from beaconai.models.registry import build_model
from beaconai.models.train import configure_mlflow, register_version
from tests.conftest import make_config


def test_calibrated_model_logs_and_serves(tmp_path: Path, toy_processed: Path) -> None:
    cfg = make_config(tmp_path, dev_sample=None)
    configure_mlflow(cfg.mlflow, experiment="pyfunc-test")
    train, test = load_split(toy_processed, "train"), load_split(toy_processed, "test")
    names = load_feature_names(toy_processed)
    model = build_model("logreg", {"max_iter": 200}, 3, 0).fit(train.X, train.y)
    clf = CalibratedClassifier(
        model,
        TemperatureCalibrator((0.05, 20.0), temperature=2.0),
        load_labels(toy_processed),
        names,
        0.5,
        3,
    )
    clf.save(tmp_path / "c")
    example = pd.DataFrame(test.X[:5].toarray().astype(np.float32), columns=names)
    with mlflow.start_run():
        uri = log_pyfunc(CalibratedPyfunc(), tmp_path / "c", example, clf.predict_proba(test.X[:5]))
    served = mlflow.pyfunc.load_model(uri)
    frame = pd.DataFrame(test.X.toarray(), columns=names)
    np.testing.assert_allclose(np.asarray(served.predict(frame)), clf.predict_proba(test.X))
    assert mlflow.get_experiment_by_name("pyfunc-test") is not None

    version, alias = register_version(uri, cfg.mlflow, None, {"phase": "3"})
    assert (version, alias) == (1, "champion")
    assert register_version(uri, cfg.mlflow, 50000, {})[1] == "dev"

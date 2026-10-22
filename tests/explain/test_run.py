import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from scipy import sparse

from beaconai.config import ExplainConfig
from beaconai.data.encode import FeatureSpace
from beaconai.data.schema import Catalog, parse_conditions, parse_evidences
from beaconai.models.calibrate import run_calibration
from beaconai.models.calibrated import PatientEvidence
from beaconai.models.train import configure_mlflow, run_training
from tests.conftest import make_config
from tests.data.conftest import AGE_BINS, RAW_CONDITIONS, RAW_EVIDENCES

TOKENS = {
    0: ["E_1", "E_10_@_7", "E_30_@_V_5"],
    1: ["E_2", "E_30_@_V_6", "E_20_@_V_2"],
    2: ["E_1", "E_2", "E_10_@_3"],
}
CONDITIONS = ["Influenza", "Unstable angina", "Croup"]


def _space() -> FeatureSpace:
    catalog = Catalog(
        evidences=parse_evidences(RAW_EVIDENCES), conditions=parse_conditions(RAW_CONDITIONS)
    )
    return FeatureSpace(catalog, AGE_BINS, age_scale=110, scale_max=10)


def write_catalog_processed(root: Path) -> Path:
    """Processed data whose columns are the Phase 1 toy catalog's, so the encoder lines up."""
    space = _space()
    out = root / "ddxplus"
    out.mkdir(parents=True)
    for split, n in {"train": 150, "validate": 60, "test": 30}.items():
        y = np.arange(n) % 3
        tokens = [TOKENS[c] for c in y]
        ages = [20 + (i % 50) for i in range(n)]
        sexes = ["F", "M"] * (n // 2)
        X = space.encode_many(tokens, ages, sexes)
        ddx = np.zeros((n, 3), dtype=np.float32)
        ddx[np.arange(n), y] = 1.0
        sparse.save_npz(out / f"{split}_X.npz", X)
        np.save(out / f"{split}_y.npy", y.astype(np.int16))
        sparse.save_npz(out / f"{split}_ddx.npz", sparse.csr_matrix(ddx))
        np.savez_compressed(
            out / f"{split}_meta.npz",
            age=np.array(ages, dtype=np.int16),
            sex=np.array(sexes, dtype="U1"),
            initial_evidence=np.array([t[0] for t in tokens]),
            n_evidences=np.array([len(t) for t in tokens], dtype=np.int16),
        )
    labels = [
        {"index": i, "name": n, "name_fr": "", "icd10": "", "severity": 3}
        for i, n in enumerate(CONDITIONS)
    ]
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "features.json").write_text(json.dumps([f.to_dict() for f in space.features]))
    manifest = {
        "splits": {},
        "n_features": space.n_features,
        "n_classes": 3,
        "settings": {"dev_sample": None},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def _cfg(tmp_path: Path, conditions: list[str]):
    cfg = make_config(tmp_path, dev_sample=None)
    write_catalog_processed(tmp_path / "processed")
    run_training(cfg)
    # The explainer needs a logistic regression champion, as in the real pipeline.
    cal = cfg.calibration.model_copy(update={"models": ["logreg"]})
    cfg = cfg.model_copy(update={"calibration": cal})
    run_calibration(cfg)
    # model_construct: the toy has 3 conditions, below the 6-to-8 rule meant for real data.
    explain = ExplainConfig.model_construct(
        top_positive=5,
        top_negative=3,
        global_top=4,
        global_conditions=conditions,
        min_prevalence=0.01,
        ranking_min_prob=0.01,
        latency_budget_ms=50,
        latency_calls=5,
        experiment="explain-test",
    )
    # No raw CSV in tests: the baseline means come from the processed train metadata.
    paths = cfg.data.paths.model_copy(update={"raw": tmp_path / "raw"})
    encoding = cfg.data.encoding.model_copy(update={"age_bins": AGE_BINS})  # the toy catalog's
    data = cfg.data.model_copy(update={"paths": paths, "encoding": encoding})
    return cfg.model_copy(update={"explain": explain, "data": data})


PATIENTS = [PatientEvidence(evidence=TOKENS[c], age=40, sex="F") for c in (0, 1, 2, 0, 1)]


def test_run_explain_writes_report_figures_baseline_and_mlflow(tmp_path: Path, monkeypatch) -> None:
    import beaconai.explain.run as run

    cfg = _cfg(tmp_path, CONDITIONS)
    monkeypatch.setattr(run, "build_feature_space", lambda _cfg: _space())
    summary = run.run_explain(cfg, patients=PATIENTS)
    reports = tmp_path / "reports"
    assert (reports / "explainability.md").is_file()
    assert json.loads((reports / "explainability_summary.json").read_text()) == summary
    assert summary["latency_ms"]["n"] == len(PATIENTS)
    assert set(summary["global"]) == set(CONDITIONS)
    for name in CONDITIONS:
        assert (reports / summary["images"][name]).is_file()
    assert summary["examples"]
    baseline = json.loads((reports / "explain_baseline.json").read_text())
    assert baseline["source"].startswith("processed train split")
    assert baseline["means"]["SEX_M"] == pytest.approx(0.5)
    configure_mlflow(cfg.mlflow, experiment="explain-test")
    assert len(mlflow.search_runs(experiment_names=["explain-test"])) == 1


def test_unknown_condition_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    import beaconai.explain.run as run

    cfg = _cfg(tmp_path, ["Influenza", "Measles"])
    monkeypatch.setattr(run, "build_feature_space", lambda _cfg: _space())
    with pytest.raises(ValueError, match="Measles"):
        run.run_explain(cfg, patients=PATIENTS)

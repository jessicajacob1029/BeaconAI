"""End-to-end pipeline test on the tiny synthetic raw folder from conftest.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from beaconai.data.eda import run_eda
from beaconai.data.preprocess import SPLITS, run_preprocess
from tests.data.conftest import AGE_BINS, PATIENT_ROWS, write_patients_csv


def _run(raw_dir: Path, out_dir: Path) -> dict:
    return run_preprocess(
        raw_dir,
        out_dir,
        age_bins=AGE_BINS,
        age_scale=110,
        scale_max=10,
        dev_sample=None,
        seed=42,
    )


def test_pipeline_writes_all_outputs(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    manifest = _run(raw_dir, out)

    features = json.loads((out / "features.json").read_text(encoding="utf-8"))
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    assert manifest["n_features"] == len(features) == 14
    assert [c["name"] for c in labels] == ["Influenza", "Unstable angina"]
    assert labels[1]["severity"] == 1
    assert manifest["splits"] == {s: len(PATIENT_ROWS) for s in SPLITS}
    assert manifest["features_by_kind"]["scale"] == 1

    for split in SPLITS:
        X = sparse.load_npz(out / f"{split}_X.npz")
        y = np.load(out / f"{split}_y.npy")
        ddx = sparse.load_npz(out / f"{split}_ddx.npz")
        meta = np.load(out / f"{split}_meta.npz")
        assert X.shape == (3, len(features))
        assert X.dtype == np.float32
        assert y.tolist() == [0, 1, 0]
        np.testing.assert_allclose(ddx.toarray().sum(axis=1), 1.0, rtol=1e-5)
        assert meta["age"].tolist() == [34, 70, 5]
        assert meta["sex"].tolist() == ["F", "M", "M"]
        assert meta["initial_evidence"].tolist() == ["E_1", "E_2", "E_1"]
        assert meta["n_evidences"].tolist() == [5, 2, 1]

    # Every column maps back to a readable label; spot-check the scale column.
    X = sparse.load_npz(out / "train_X.npz")
    scale_col = next(f["index"] for f in features if f["kind"] == "scale")
    assert X[0, scale_col] == pytest.approx(0.6)


def test_pipeline_clears_stale_outputs(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    out.mkdir()
    (out / "stale.npz").write_text("old")
    _run(raw_dir, out)
    assert not (out / "stale.npz").exists()


def test_pipeline_rejects_unknown_pathology(raw_dir: Path, tmp_path: Path) -> None:
    bad = [(*PATIENT_ROWS[0][:3], "Not a condition", *PATIENT_ROWS[0][4:])]
    write_patients_csv(raw_dir / "release_train_patients.csv", bad)
    with pytest.raises(KeyError, match="Unknown PATHOLOGY"):
        _run(raw_dir, tmp_path / "processed")


def test_eda_saves_three_plots(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    _run(raw_dir, out)
    paths = run_eda(out, "train", tmp_path / "reports")
    assert sorted(p.name for p in paths) == [
        "eda_age_by_sex.png",
        "eda_class_balance.png",
        "eda_evidences_per_patient.png",
    ]
    assert all(p.stat().st_size > 0 for p in paths)

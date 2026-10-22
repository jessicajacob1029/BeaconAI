"""Tiny synthetic processed dataset and config factory. No real data is used in tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from beaconai.config import DEFAULT_CONFIG_PATH, Config

TOY_CLASSES = 3
TOY_FEATURES = 6
TOY_ROWS = {"train": 150, "validate": 60, "test": 60}


def write_toy_processed(root: Path, seed: int = 0, dev_sample: int | None = None) -> Path:
    """Write Phase 1 shaped outputs. Class c switches on features 2c and (mostly) 2c+1.

    `dev_sample` is recorded in the manifest, as preprocess does for the real data.
    """
    out = root / "ddxplus"
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for split, n in TOY_ROWS.items():
        y = np.arange(n) % TOY_CLASSES
        X = (rng.random((n, TOY_FEATURES)) < 0.1).astype(np.float32)
        X[np.arange(n), 2 * y] = 1.0
        X[np.arange(n), 2 * y + 1] = (rng.random(n) < 0.8).astype(np.float32)
        ddx = np.zeros((n, TOY_CLASSES), dtype=np.float32)
        ddx[np.arange(n), y] = 0.6
        ddx[np.arange(n), (y + 1) % TOY_CLASSES] = 0.4
        sparse.save_npz(out / f"{split}_X.npz", sparse.csr_matrix(X))
        np.save(out / f"{split}_y.npy", y.astype(np.int16))
        sparse.save_npz(out / f"{split}_ddx.npz", sparse.csr_matrix(ddx))
        np.savez_compressed(
            out / f"{split}_meta.npz",
            age=np.full(n, 40, dtype=np.int16),
            sex=np.array(["F", "M"] * (n // 2) + ["F"] * (n % 2), dtype="U1"),
            initial_evidence=np.array([f"E_{2 * c}" for c in y]),
            n_evidences=np.asarray((X > 0).sum(axis=1), dtype=np.int16),
        )
    labels = [
        {"index": i, "name": f"Condition {i}", "name_fr": "", "icd10": "", "severity": i + 1}
        for i in range(TOY_CLASSES)
    ]
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    features = [
        {
            "index": i,
            "name": f"F_{i}",
            "kind": "binary",
            "label": f"Feature {i}",
            "evidence": f"E_{i}",
        }
        for i in range(TOY_FEATURES)
    ]
    (out / "features.json").write_text(json.dumps(features), encoding="utf-8")
    manifest = {
        "splits": TOY_ROWS,
        "n_features": TOY_FEATURES,
        "n_classes": TOY_CLASSES,
        "settings": {"dev_sample": dev_sample},
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


def make_config(tmp_path: Path, *, dev_sample: int | None) -> Config:
    """Base config pointed at tmp_path, with tiny grids so tests run in seconds."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["dev_sample"] = dev_sample
    raw["data"]["paths"]["processed"] = str(tmp_path / "processed")
    raw["mlflow"]["tracking_db"] = str(tmp_path / "mlruns" / "mlflow.db")
    raw["mlflow"]["artifact_root"] = str(tmp_path / "mlruns" / "artifacts")
    raw["train"]["latency_calls"] = 5
    raw["eval"].update(output_dir=str(tmp_path / "reports"), ks=[1, 3], ddx_ks=[1, 3])
    raw["models"] = {
        "logreg": {"fixed": {"max_iter": 200}, "grid": {"C": [0.1, 1.0]}},
        "xgboost": {
            "fixed": {"n_estimators": 10, "early_stopping_rounds": 3, "tree_method": "hist"},
            "grid": {"max_depth": [2]},
        },
        "mlp": {
            "fixed": {"epochs": 3, "batch_size": 32, "patience": 2, "weight_decay": 0.0},
            "grid": {"hidden_sizes": [[8]], "dropout": [0.0], "lr": [0.01]},
        },
    }
    raw["calibration"].update(
        ece_bins=10,
        top3_floor={"full": 0.5, "partial": 0.0},
        partial_evidence={"extra_evidences": 1},
    )
    raw["abstention"].update(curve_points=11, target_accuracy=0.9)
    # Toy runs stay small; dropout-specific tests opt in to the dropout variant.
    raw["calibration"].update(variants=["baseline"], evidence_curve=[0, 1, 2])
    raw["dropout"].update(copies=1)
    return Config.model_validate(raw)


@pytest.fixture
def toy_processed(tmp_path: Path) -> Path:
    return write_toy_processed(tmp_path / "processed")

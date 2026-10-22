"""Phase 3b: retrain the Phase 2 winners with random evidence dropout.

Run: uv run python -m beaconai.models.train_dropout

The training set is the original rows plus copies in which each evidence other than the initial
one is dropped. Every copied row draws its own dropout rate from `dropout.rate_range`, so
training covers both few-symptom and many-symptom patients and the models learn that a missing
evidence is not necessarily a "no". Hyperparameters are the Phase 2 winners. Early stopping uses
the partial-evidence version of the calibration fit half of validation; the held-out half is
never touched here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from beaconai.config import (
    Config,
    load_config,
    project_path,
    require_calibration,
    require_dropout,
    require_training,
)
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.models.dataset import (
    SplitData,
    augment_with_dropout,
    load_features,
    load_labels,
    load_meta,
    load_split,
    split_indices,
)
from beaconai.models.pyfunc import BeaconPyfunc, log_pyfunc
from beaconai.models.registry import build_model
from beaconai.models.seed import set_global_seed
from beaconai.models.train import SUMMARY_FILE, check_data_scope, configure_mlflow, data_scope
from beaconai.models.tuning import flatten_params

DROPOUT_SUMMARY_FILE = "train_dropout_summary.json"


def early_stopping_rows(
    val: SplitData, meta: dict[str, np.ndarray], features: list[dict[str, Any]], cfg: Config
) -> tuple[SplitData, np.ndarray]:
    """Partial-evidence version of the calibration fit half (same split as calibration)."""
    cal_cfg, _ = require_calibration(cfg)
    fit_idx, _ = split_indices(val.y, cal_cfg.fit_fraction, cfg.seed)
    X = partial_evidence(
        val.X[fit_idx],
        meta["initial_evidence"][fit_idx],
        column_evidence(features),
        cal_cfg.partial_evidence.extra_evidences,
        cfg.seed,
    )
    return SplitData(X, val.y[fit_idx], None), fit_idx


def run_dropout_training(cfg: Config) -> dict[str, Any]:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    drop = require_dropout(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    train_summary_path = reports / SUMMARY_FILE
    if not train_summary_path.is_file():
        raise FileNotFoundError(f"{train_summary_path} not found. Run `make train` first.")
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    dev_sample = data_scope(processed)
    check_data_scope(train_summary["dev_sample"], dev_sample)

    labels, features = load_labels(processed), load_features(processed)
    feature_names = [f["name"] for f in features]
    train = load_split(processed, "train")
    aug = augment_with_dropout(
        train,
        load_meta(processed, "train")["initial_evidence"],
        column_evidence(features),
        drop.rate_range,
        drop.copies,
        drop.keep_original,
        cfg.seed,
    )
    val = load_split(processed, "validate")
    stop, _ = early_stopping_rows(val, load_meta(processed, "validate"), features, cfg)
    configure_mlflow(mlflow_cfg, experiment=drop.experiment)

    models: dict[str, Any] = {}
    with mlflow.start_run(run_name="phase3b-train-dropout") as parent:
        mlflow.set_tags({"dev_sample": str(dev_sample), "phase": "3b"})
        for name in drop.models:
            params = train_summary["models"][name]["params"]
            set_global_seed(cfg.seed)
            with mlflow.start_run(run_name=f"{name}+dropout", nested=True) as run:
                mlflow.set_tags({"model_type": name, "variant": "dropout"})
                mlflow.log_params(
                    {
                        **flatten_params(params),
                        "dropout_rate_range": json.dumps(list(drop.rate_range)),
                        "dropout_copies": drop.copies,
                        "keep_original": drop.keep_original,
                        "n_train_rows": aug.n,
                    }
                )
                model = build_model(name, params, len(labels), cfg.seed)
                start = time.perf_counter()
                model.fit(aug.X, aug.y, stop.X, stop.y)
                fit_seconds = time.perf_counter() - start
                stopped_at = {
                    attr: int(getattr(model, attr))
                    for attr in ("best_iteration", "epochs_trained")
                    if hasattr(model, attr)
                }
                mlflow.log_metrics({"fit_seconds": fit_seconds, **stopped_at})
                with tempfile.TemporaryDirectory() as tmp:
                    model_dir = Path(tmp) / "wrapper"
                    model.save(model_dir)
                    rows = stop.X[:5]
                    example = pd.DataFrame(rows.toarray().astype(np.float32), columns=feature_names)
                    uri = log_pyfunc(BeaconPyfunc(), model_dir, example, model.predict_proba(rows))
            models[name] = {
                "run_id": run.info.run_id,
                "model_uri": uri,
                "params": params,
                "fit_seconds": fit_seconds,
                **stopped_at,
            }

    summary = {
        "parent_run_id": parent.info.run_id,
        "dev_sample": dev_sample,
        "rate_range": list(drop.rate_range),
        "copies": drop.copies,
        "keep_original": drop.keep_original,
        "n_train_rows": aug.n,
        "n_stop_rows": stop.n,
        "models": models,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / DROPOUT_SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = run_dropout_training(load_config())
    keys = ("dev_sample", "rate_range", "n_train_rows")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

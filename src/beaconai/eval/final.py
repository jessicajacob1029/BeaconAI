"""The one-time test evaluation of the calibrated champion (make eval).

Run: uv run python -m beaconai.eval.final

This is the only code that scores the test split. It evaluates `beaconai-classifier@champion`
and refuses unless that version is a full-data calibrated model, the processed data on disk is
the full dataset, and this evaluation (by name) has not run before. Each evaluation name has its
own lock in reports/test_locks/, so a later evaluation (for example a Phase 8 agent simulation)
can have its own one-time test run.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from beaconai.config import (
    Config,
    load_config,
    project_path,
    require_calibration,
    require_training,
)
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.final_report import render_final
from beaconai.eval.metrics import evaluate_predictions
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.eval.plots import plot_confusion
from beaconai.models.calibrate import answered_at, calibration_metrics, load_calibrated
from beaconai.models.dataset import load_features, load_labels, load_meta, load_split
from beaconai.models.train import configure_mlflow, data_scope

REPORT_FILE = "final_evaluation.md"
SUMMARY_FILE = "final_evaluation.json"
LOCK_DIR = "test_locks"


class FinalEvalError(Exception):
    """Raised when the final test evaluation would break its rules."""


def lock_path(cfg: Config) -> Path:
    _, _, _, eval_cfg = require_training(cfg)
    return project_path(eval_cfg.output_dir) / LOCK_DIR / f"{cfg.final_eval.name}.json"


def check_final_allowed(
    dev_sample_on_disk: int | None, version_tags: dict[str, str], lock: Path
) -> None:
    if dev_sample_on_disk is not None:
        raise FinalEvalError(
            "The final evaluation needs the full dataset on disk "
            "(BEACONAI_DEV_SAMPLE=null uv run python -m beaconai.data.preprocess)."
        )
    if "calibration" not in version_tags:
        raise FinalEvalError(
            "The champion alias does not point at a calibrated model. make eval scores only "
            "the calibrated champion; run `make calibrate` on full data first."
        )
    if version_tags.get("dev_sample") != "None":
        raise FinalEvalError("The champion was calibrated on the dev subsample, not full data.")
    if lock.is_file():
        raise FinalEvalError(
            f"This evaluation already scored the test split ({lock}). It runs once; "
            "re-scoring after changes would leak test information."
        )


def _write_lock(
    lock: Path,
    name: str,
    model_version: int,
    status: str,
    started_at: str,
    completed_at: str | None = None,
) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "evaluation": name,
        "model_version": model_version,
        "status": status,
        "started_at": started_at,
    }
    if completed_at is not None:
        payload["completed_at"] = completed_at
    lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_final_evaluation(cfg: Config) -> Path:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    cal_cfg, abst = require_calibration(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    lock = lock_path(cfg)
    configure_mlflow(mlflow_cfg, experiment=cfg.final_eval.experiment)
    name, alias = mlflow_cfg.registered_model, mlflow_cfg.champion_alias
    on_disk = data_scope(processed)
    try:
        version = MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException as err:
        check_final_allowed(on_disk, {}, lock)  # a clearer reason first, if one applies
        raise FinalEvalError(
            f"No `{alias}` model is registered; run `make calibrate` on full data first."
        ) from err
    tags = dict(version.tags)
    check_final_allowed(on_disk, tags, lock)
    # Lock before the test split is read: if anything fails after this point, the test split
    # may already have been seen, so a rerun must still be refused.
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    _write_lock(lock, cfg.final_eval.name, int(version.version), "started", started_at)

    clf = load_calibrated(f"models:/{name}@{alias}")
    labels, features = load_labels(processed), load_features(processed)
    test, meta = load_split(processed, "test"), load_meta(processed, "test")
    partial_X = partial_evidence(
        test.X,
        meta["initial_evidence"],
        column_evidence(features),
        cal_cfg.partial_evidence.extra_evidences,
        cfg.seed,
    )
    k = abst.differential_size
    metrics: dict[str, dict[str, float]] = {}
    abstention: dict[str, dict[str, Any]] = {}
    confusion = None
    for key, X in (("full", test.X), ("partial", partial_X)):
        proba, raw = clf.predict_raw_and_calibrated(X)
        result = evaluate_predictions(
            test.y, proba, test.ddx, eval_cfg.ks, eval_cfg.ddx_ks, tie_break=raw
        )
        metrics[key] = result.metrics | calibration_metrics(
            test.y, proba, cal_cfg.ece_bins, tie_break=raw
        )
        abstention[key] = answered_at(test.y, proba, clf.threshold, k, raw)
        if key == "full":
            confusion = result.confusion

    figure = plot_confusion(
        confusion,
        [lab["name"] for lab in labels],
        reports / "figures" / "final_confusion_matrix.png",
        "Final champion (test, full evidence)",
    )
    summary = {
        "evaluation": cfg.final_eval.name,
        "registered": {"name": name, "version": int(version.version), "alias": alias},
        "champion": {
            "model": tags.get("model_type"),
            "method": tags.get("calibration"),
            "fit_on": tags.get("calibrated_on", "full"),
        },
        "n_test": test.n,
        "threshold": clf.threshold,
        "metrics": metrics,
        "abstention": abstention,
    }
    summary_path, report = reports / SUMMARY_FILE, reports / REPORT_FILE
    with mlflow.start_run(run_name=f"test-{cfg.final_eval.name}"):
        mlflow.set_tags({"model_version": version.version, "split": "test"})
        for key, values in metrics.items():
            mlflow.log_metrics({f"test_{key}_{m}": v for m, v in values.items()})
        report.write_text(render_final(summary), encoding="utf-8")
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for path in (report, summary_path, figure):
            mlflow.log_artifact(str(path))
    _write_lock(
        lock,
        cfg.final_eval.name,
        int(version.version),
        "completed",
        started_at,
        completed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return report


def main() -> int:
    try:
        print(run_final_evaluation(load_config()))
    except FinalEvalError as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 3: calibrate the tuned models, pick the champion by partial-evidence ECE, set abstention.

Run: uv run python -m beaconai.models.calibrate

Validation is split per class: calibrators are fit on one part and every number is reported on
the other part, so no calibrator is scored on the rows it was fit to. The test split is not used.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy import sparse

from beaconai.config import (
    AbstentionConfig,
    Config,
    load_config,
    project_path,
    require_calibration,
    require_training,
)
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.calibration import (
    coverage_curve,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    suggest_threshold,
)
from beaconai.eval.calibration_report import render_calibration
from beaconai.eval.metrics import top_k_accuracy
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.eval.plots import plot_coverage, plot_evidence_curve, plot_reliability
from beaconai.models.base import ModelWrapper
from beaconai.models.calibrated import CalibratedClassifier
from beaconai.models.calibrators import Calibrator, TemperatureCalibrator, make_calibrator
from beaconai.models.dataset import (
    SplitData,
    load_features,
    load_labels,
    load_meta,
    load_split,
    split_indices,
    subset,
)
from beaconai.models.pyfunc import CalibratedPyfunc, log_pyfunc
from beaconai.models.train import SUMMARY_FILE as TRAIN_SUMMARY_FILE
from beaconai.models.train import (
    check_data_scope,
    configure_mlflow,
    data_scope,
    load_wrapper,
    register_version,
)
from beaconai.models.train_dropout import DROPOUT_SUMMARY_FILE

TOP3 = 3
BASELINE_FILE = "phase3_baseline_summary.json"


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    method: str
    run_id: str
    model_uri: str
    metrics: dict[str, float]
    temperature: float | None
    fit_on: str = "full"


def calibration_metrics(
    y: np.ndarray, proba: np.ndarray, n_bins: int, tie_break: np.ndarray | None = None
) -> dict[str, float]:
    k3 = min(TOP3, proba.shape[1])
    return {
        "ece": expected_calibration_error(y, proba, n_bins),
        "nll": negative_log_likelihood(y, proba),
        "top1_accuracy": top_k_accuracy(y, proba, 1, tie_break),
        "top3_accuracy": top_k_accuracy(y, proba, k3, tie_break),
    }


def select_calibrated(
    results: list[CalibrationResult], floor_full: float, floor_partial: float
) -> CalibrationResult:
    """Lowest partial-evidence ECE among pairs meeting both top-3 floors; ties keep order."""
    eligible = [
        r
        for r in results
        if r.metrics["top3_accuracy"] >= floor_full
        and r.metrics["partial_top3_accuracy"] >= floor_partial
    ]
    if not eligible:
        raise ValueError(
            f"No (model, method) pair meets the top-3 floors (full {floor_full}, "
            f"partial {floor_partial})"
        )
    best = eligible[0]
    for r in eligible[1:]:
        if r.metrics["partial_ece"] < best.metrics["partial_ece"]:
            best = r
    return best


def load_calibrated(model_uri: str) -> CalibratedClassifier:
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().model


def _log_candidate(
    name: str,
    method: str,
    base: ModelWrapper,
    calibrator: Calibrator,
    metrics: dict[str, float],
    labels: list[dict[str, Any]],
    feature_names: list[str],
    abst: AbstentionConfig,
    held: SplitData,
    fit_on: str,
) -> CalibrationResult:
    temperature = calibrator.temperature if isinstance(calibrator, TemperatureCalibrator) else None
    with mlflow.start_run(run_name=f"{name}-{method}-{fit_on}", nested=True) as run:
        mlflow.set_tags({"model_type": name, "calibration": method, "calibrated_on": fit_on})
        mlflow.log_params({"method": method, "temperature": temperature})
        mlflow.log_metrics(metrics)
        clf = CalibratedClassifier(
            base, calibrator, labels, feature_names, abst.threshold, abst.differential_size
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "calibrated"
            clf.save(model_dir)
            rows = held.X[:5]
            example = pd.DataFrame(rows.toarray().astype(np.float32), columns=feature_names)
            uri = log_pyfunc(CalibratedPyfunc(), model_dir, example, clf.predict_proba(rows))
    return CalibrationResult(name, method, run.info.run_id, uri, metrics, temperature, fit_on)


def answered_at(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    k: int,
    tie_break: np.ndarray | None = None,
) -> dict[str, Any]:
    """Coverage and accuracy at one threshold. Accuracy is None (not NaN) if nobody is answered."""
    curve = coverage_curve(y, proba, [threshold], k, tie_break)
    coverage = float(curve.coverage[0])
    out: dict[str, Any] = {
        "coverage": coverage,
        "top1_accuracy": None if coverage == 0 else float(curve.top1_accuracy[0]),
        "topk_accuracy": None if coverage == 0 else float(curve.topk_accuracy[0]),
    }
    if coverage == 0:
        out["note"] = (
            f"No patients are answered at threshold {threshold:.3f}; accuracy is undefined."
        )
    return out


def evidence_curve(
    predict: Callable[[sparse.csr_matrix], tuple[np.ndarray, np.ndarray]],
    X: sparse.csr_matrix,
    y: np.ndarray,
    initial_evidence: np.ndarray,
    col_evidence: np.ndarray,
    extras: list[int],
    seed: int,
    n_bins: int,
) -> dict[str, Any]:
    """Top-3 accuracy and ECE when patients keep the initial evidence plus n other evidences.

    `predict` returns (calibrated, raw) probabilities; raw breaks exact ties as served.
    """
    top3, ece = [], []
    for n in extras:
        proba, raw = predict(partial_evidence(X, initial_evidence, col_evidence, n, seed))
        m = calibration_metrics(y, proba, n_bins, tie_break=raw)
        top3.append(m["top3_accuracy"])
        ece.append(m["ece"])
    return {"extra": list(extras), "top3_accuracy": top3, "ece": ece}


def _base_models(
    cfg: Config, train_summary: dict[str, Any], reports: Path, dev_sample: int | None
) -> list[tuple[str, str]]:
    """(label, model_uri) for every base model to calibrate: Phase 2 models and dropout retrains."""
    cal_cfg, _ = require_calibration(cfg)
    bases: list[tuple[str, str]] = []
    if "baseline" in cal_cfg.variants:
        bases += [(m, train_summary["models"][m]["model_uri"]) for m in cal_cfg.models]
    if "dropout" in cal_cfg.variants:
        path = reports / DROPOUT_SUMMARY_FILE
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `make train-dropout` first.")
        dropout_summary = json.loads(path.read_text(encoding="utf-8"))
        check_data_scope(dropout_summary["dev_sample"], dev_sample)
        bases += [(f"{m}+dropout", v["model_uri"]) for m, v in dropout_summary["models"].items()]
    return bases


def run_calibration(cfg: Config) -> dict[str, Any]:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    cal_cfg, abst = require_calibration(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    train_summary_path = reports / TRAIN_SUMMARY_FILE
    if not train_summary_path.is_file():
        raise FileNotFoundError(f"{train_summary_path} not found. Run `make train` first.")
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    dev_sample = data_scope(processed)
    check_data_scope(train_summary["dev_sample"], dev_sample)
    missing = [m for m in cal_cfg.models if m not in train_summary["models"]]
    if missing:
        raise ValueError(f"{missing} not in {train_summary_path}; add them to train.models")

    labels = load_labels(processed)
    features = load_features(processed)
    feature_names = [f["name"] for f in features]
    val = load_split(processed, "validate")
    meta = load_meta(processed, "validate")
    fit_idx, held_idx = split_indices(val.y, cal_cfg.fit_fraction, cfg.seed)
    fit, held = subset(val, fit_idx), subset(val, held_idx)
    partial = SplitData(
        partial_evidence(
            held.X,
            meta["initial_evidence"][held_idx],
            column_evidence(features),
            cal_cfg.partial_evidence.extra_evidences,
            cfg.seed,
        ),
        held.y,
        held.ddx,
    )
    cols = column_evidence(features)
    extra = cal_cfg.partial_evidence.extra_evidences
    # Partial-evidence version of the fit half, for calibrators fit on partial rows.
    fit_partial = SplitData(
        partial_evidence(fit.X, meta["initial_evidence"][fit_idx], cols, extra, cfg.seed),
        fit.y,
        None,
    )
    bases = _base_models(cfg, train_summary, reports, dev_sample)
    baseline_path = reports / BASELINE_FILE
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    )
    if baseline is not None and baseline.get("dev_sample") != dev_sample:
        baseline = None  # never compare dev numbers with full-data numbers
    sfx = "" if dev_sample is None else "_dev"
    figures = reports / "figures"
    k = abst.differential_size
    floor = cal_cfg.top3_floor
    configure_mlflow(mlflow_cfg, experiment=cal_cfg.experiment)
    # One phase label for the MLflow run and the registry version.
    phase = "3b" if "dropout" in cal_cfg.variants else "3"

    results: list[CalibrationResult] = []
    reliability_images: dict[str, str] = {}
    with mlflow.start_run(run_name="calibrate") as parent:
        mlflow.set_tags(
            {
                "dev_sample": str(dev_sample),
                "train_parent_run_id": train_summary["parent_run_id"],
                "phase": phase,
                "selection": "lowest partial-evidence ECE, top-3 floors on full and partial",
            }
        )
        mlflow.log_params(
            {
                "fit_fraction": cal_cfg.fit_fraction,
                "ece_bins": cal_cfg.ece_bins,
                "top3_floor_full": floor.full,
                "top3_floor_partial": floor.partial,
                "abstention_threshold": abst.threshold,
                "extra_evidences": cal_cfg.partial_evidence.extra_evidences,
                "n_fit": len(fit_idx),
                "n_held": len(held_idx),
            }
        )
        for name, base_uri in bases:
            base = load_wrapper(base_uri)
            raw = {
                "full": base.predict_proba(fit.X),
                "partial": base.predict_proba(fit_partial.X),
            }
            raw_held = base.predict_proba(held.X)
            raw_partial = base.predict_proba(partial.X)
            panels: dict[str, dict[str, Any]] = {"Full evidence": {}, "Partial evidence": {}}
            for fit_on in cal_cfg.fit_on:
                for method in cal_cfg.methods:
                    if method == "none" and fit_on != cal_cfg.fit_on[0]:
                        continue  # raw probabilities do not depend on the fit rows
                    calibrator = make_calibrator(method, cal_cfg.temperature_bounds).fit(
                        raw[fit_on], fit.y
                    )
                    q_held = calibrator.transform(raw_held)
                    q_partial = calibrator.transform(raw_partial)
                    metrics = calibration_metrics(
                        held.y, q_held, cal_cfg.ece_bins, tie_break=raw_held
                    )
                    partial_m = calibration_metrics(
                        partial.y, q_partial, cal_cfg.ece_bins, tie_break=raw_partial
                    )
                    metrics |= {f"partial_{key}": v for key, v in partial_m.items()}
                    label = method if method == "none" else f"{method} ({fit_on})"
                    panels["Full evidence"][label] = reliability_bins(
                        held.y, q_held, cal_cfg.ece_bins
                    )
                    panels["Partial evidence"][label] = reliability_bins(
                        partial.y, q_partial, cal_cfg.ece_bins
                    )
                    results.append(
                        _log_candidate(
                            name,
                            method,
                            base,
                            calibrator,
                            metrics,
                            labels,
                            feature_names,
                            abst,
                            held,
                            fit_on,
                        )
                    )
            image = f"reliability_{name.replace('+', '_')}{sfx}.png"
            path = plot_reliability(panels, figures / image, f"{name}: reliability (held-out)")
            mlflow.log_artifact(str(path))
            reliability_images[name] = f"figures/{image}"

        champion = select_calibrated(results, floor.full, floor.partial)
        clf = load_calibrated(champion.model_uri)
        p_held, r_held = clf.predict_raw_and_calibrated(held.X)
        p_partial, r_partial = clf.predict_raw_and_calibrated(partial.X)
        thresholds = np.linspace(0.0, 1.0, abst.curve_points)
        curves = {
            "Full evidence": coverage_curve(held.y, p_held, thresholds, k, r_held),
            "Partial evidence": coverage_curve(partial.y, p_partial, thresholds, k, r_partial),
        }
        held_initial = meta["initial_evidence"][held_idx]
        suggest_set = None
        if abst.suggest_on_extra is None:
            suggested = suggest_threshold(curves["Partial evidence"], abst.target_accuracy)
        else:
            n = abst.suggest_on_extra
            X_s = partial_evidence(held.X, held_initial, cols, n, cfg.seed)
            p_s, r_s = clf.predict_raw_and_calibrated(X_s)
            curves[f"Initial evidence + {n}"] = coverage_curve(held.y, p_s, thresholds, k, r_s)
            suggested = suggest_threshold(curves[f"Initial evidence + {n}"], abst.target_accuracy)
            suggest_set = answered_at(held.y, p_s, abst.threshold, k, r_s)
        coverage_image = f"coverage_accuracy{sfx}.png"
        cov_path = plot_coverage(
            curves,
            abst.threshold,
            figures / coverage_image,
            f"{champion.model} + {champion.method}",
        )
        mlflow.log_artifact(str(cov_path))
        full_at = answered_at(held.y, p_held, abst.threshold, k, r_held)
        partial_at = answered_at(partial.y, p_partial, abst.threshold, k, r_partial)
        mlflow.log_metrics(
            {
                "champion_partial_ece": champion.metrics["partial_ece"],
                "champion_ece": champion.metrics["ece"],
                "abstain_rate_full": 1 - full_at["coverage"],
                "abstain_rate_partial": 1 - partial_at["coverage"],
                **({"suggested_threshold": suggested} if suggested is not None else {}),
            }
        )
        curve = {
            "extra": list(cal_cfg.evidence_curve),
            "champion": evidence_curve(
                clf.predict_raw_and_calibrated,
                held.X,
                held.y,
                held_initial,
                cols,
                cal_cfg.evidence_curve,
                cfg.seed,
                cal_cfg.ece_bins,
            ),
            "phase3_champion": None,
        }
        if baseline is not None:
            old_clf = load_calibrated(baseline["champion"]["model_uri"])
            curve["phase3_champion"] = evidence_curve(
                old_clf.predict_raw_and_calibrated,
                held.X,
                held.y,
                held_initial,
                cols,
                cal_cfg.evidence_curve,
                cfg.seed,
                cal_cfg.ece_bins,
            )
        curve_image = f"evidence_curve{sfx}.png"
        if cal_cfg.evidence_curve:
            series = {"Phase 3b champion": curve["champion"]}
            if curve["phase3_champion"] is not None:
                series = {"Phase 3 champion": curve["phase3_champion"], **series}
            mlflow.log_artifact(
                str(plot_evidence_curve(series, figures / curve_image, "Known evidences"))
            )
        mlflow.set_tags(
            {
                "champion_model": champion.model,
                "champion_method": champion.method,
                "champion_calibrated_on": champion.fit_on,
            }
        )

    version, alias = register_version(
        champion.model_uri,
        mlflow_cfg,
        dev_sample,
        {
            "model_type": champion.model,
            "calibration": champion.method,
            "calibrated_on": champion.fit_on,
            "held_out_partial_ece": f"{champion.metrics['partial_ece']:.5f}",
            "held_out_ece": f"{champion.metrics['ece']:.5f}",
            "phase": phase,
        },
    )
    summary = {
        "parent_run_id": parent.info.run_id,
        "train_parent_run_id": train_summary["parent_run_id"],
        "dev_sample": dev_sample,
        "n_fit": len(fit_idx),
        "n_held": len(held_idx),
        "top3_floor": floor.model_dump(),
        "champion": {
            "model": champion.model,
            "method": champion.method,
            "fit_on": champion.fit_on,
            "run_id": champion.run_id,
            "model_uri": champion.model_uri,
            "metrics": champion.metrics,
        },
        "registered": {"name": mlflow_cfg.registered_model, "version": version, "alias": alias},
        "candidates": [
            {
                "model": r.model,
                "method": r.method,
                "fit_on": r.fit_on,
                "temperature": r.temperature,
                "metrics": r.metrics,
            }
            for r in results
        ],
        "abstention": {
            "threshold": abst.threshold,
            "suggested_threshold": suggested,
            "target_accuracy": abst.target_accuracy,
            "k": k,
            "full": full_at,
            "partial": partial_at,
            "suggest_on_extra": abst.suggest_on_extra,
            "suggest_set": suggest_set,
        },
        "evidence_curve": curve,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"calibration_summary{sfx}.json").write_text(
        # allow_nan=False: the summary must be strict JSON (missing values are null).
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md = render_calibration(
        summary,
        ece_bins=cal_cfg.ece_bins,
        reliability_images=reliability_images,
        coverage_image=f"figures/{coverage_image}",
        baseline=baseline,
        curve_image=f"figures/{curve_image}" if cal_cfg.evidence_curve else None,
    )
    (reports / f"calibration{sfx}.md").write_text(md, encoding="utf-8")
    return summary


def main() -> int:
    summary = run_calibration(load_config())
    print(json.dumps({k: summary[k] for k in ("champion", "registered")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

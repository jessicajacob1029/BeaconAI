"""Explain the champion: global view, ranking changes, latency, examples (make explain).

Run: uv run python -m beaconai.explain.run

Writes reports/explainability{sfx}.md, its JSON summary, one figure per listed condition, and
reports/explain_baseline.json (the training-mean age and sex baseline the API will load).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from beaconai.config import (
    PROJECT_ROOT,
    Config,
    load_config,
    project_path,
    require_calibration,
    require_explain,
    require_training,
)
from beaconai.data.download import raw_dir_from_config
from beaconai.data.encode import parse_evidence_list
from beaconai.data.preprocess import processed_dir_from_config, split_csv_name
from beaconai.eval.partial import column_evidence, partial_evidence
from beaconai.eval.plots import plot_top_features
from beaconai.explain.baseline import baseline_vector, demographic_means, save_baseline
from beaconai.explain.explainer import ChampionExplainer
from beaconai.explain.global_view import global_top_features, ranking_change_rates
from beaconai.explain.labels import FeatureLabeler
from beaconai.explain.report import contrast_sentence, render_explainability
from beaconai.models.calibrate import load_calibrated
from beaconai.models.calibrated import PatientEvidence, build_feature_space
from beaconai.models.dataset import (
    load_features,
    load_labels,
    load_meta,
    load_split,
    split_indices,
    subset,
)
from beaconai.models.train import configure_mlflow, data_scope

BASELINE_FILE = "explain_baseline.json"
REVIEW_FILE = PROJECT_ROOT / "docs" / "phase4_sanity_review.md"
N_EXAMPLES = 3


def _slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _training_demographics(cfg: Config, processed: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Ages and sexes of the full training set (raw CSV), or the processed split as a fallback."""
    raw = raw_dir_from_config(cfg.data.paths.raw) / split_csv_name("train")
    if raw.is_file():
        df = pd.read_csv(raw, usecols=["AGE", "SEX"])
        return df["AGE"].to_numpy(), df["SEX"].to_numpy(), "raw train CSV (full training set)"
    meta = load_meta(processed, "train")
    scope = data_scope(processed)
    return meta["age"], meta["sex"], f"processed train split (dev_sample={scope})"


def _validation_patients(cfg: Config, n: int) -> list[PatientEvidence]:
    raw = raw_dir_from_config(cfg.data.paths.raw) / split_csv_name("validate")
    df = pd.read_csv(raw, nrows=n, dtype=str, keep_default_na=False)
    return [
        PatientEvidence(evidence=parse_evidence_list(r.EVIDENCES), age=int(r.AGE), sex=r.SEX)
        for r in df.itertuples()
    ]


def run_explain(cfg: Config, *, patients: list[PatientEvidence] | None = None) -> dict[str, Any]:
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    cal_cfg, abst = require_calibration(cfg)
    ex = require_explain(cfg)
    enc = cfg.data.encoding
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    dev_sample = data_scope(processed)
    sfx = "" if dev_sample is None else "_dev"
    labels, features = load_labels(processed), load_features(processed)
    names = [lab["name"] for lab in labels]
    unknown = [c for c in ex.global_conditions if c not in names]
    if unknown:
        raise ValueError(f"unknown conditions in explain.global_conditions: {unknown}")

    configure_mlflow(mlflow_cfg, experiment=ex.experiment)
    name, alias = mlflow_cfg.registered_model, mlflow_cfg.champion_alias
    version = MlflowClient().get_model_version_by_alias(name, alias)
    clf = load_calibrated(f"models:/{name}@{alias}").with_encoder(build_feature_space(cfg))

    ages, sexes, source = _training_demographics(cfg, processed)
    means = demographic_means(ages, sexes, features, enc.age_bins, enc.age_scale)
    baseline_path = reports / BASELINE_FILE
    save_baseline(baseline_path, means, source, len(ages))
    labeler = FeatureLabeler(features, enc.scale_max, enc.age_scale)
    explainer = ChampionExplainer(
        clf, labeler, baseline_vector(features, means), ex.top_positive, ex.top_negative
    )

    train = load_split(processed, "train")
    top = global_top_features(
        explainer.W,
        train.y,
        train.X,
        labels,
        labeler,
        ex.global_conditions,
        ex.global_top,
        ex.min_prevalence,
    )
    figures = reports / "figures"
    images = {}
    for cond, items in top.items():
        image = f"top_features_{_slug(cond)}{sfx}.png"
        plot_top_features(items, cond, figures / image)
        images[cond] = f"figures/{image}"

    val, meta = load_split(processed, "validate"), load_meta(processed, "validate")
    _, held_idx = split_indices(val.y, cal_cfg.fit_fraction, cfg.seed)
    held = subset(val, held_idx)
    partial_X = partial_evidence(
        held.X,
        meta["initial_evidence"][held_idx],
        column_evidence(features),
        cal_cfg.partial_evidence.extra_evidences,
        cfg.seed,
    )
    k = abst.differential_size
    ranking = {
        label: ranking_change_rates(
            clf.model.predict_proba(X), clf.predict_proba(X), k, ex.ranking_min_prob
        )
        for label, X in (("full", held.X), ("partial", partial_X))
    }

    patients = patients if patients is not None else _validation_patients(cfg, ex.latency_calls)
    times = []
    for patient in patients:
        start = time.perf_counter()
        explainer.explain_evidence(patient)
        times.append((time.perf_counter() - start) * 1000)
    latency = {
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "budget": ex.latency_budget_ms,
        "n": len(times),
    }

    examples = []
    for i in range(min(N_EXAMPLES, held.n)):
        explained = explainer.explain_row(partial_X[i])
        examples.append(contrast_sentence(explained, names[int(held.y[i])]))

    age_cols = [f for f in features if f.get("kind") == "age_scaled"]
    sex_cols = [f for f in features if f.get("kind") == "sex"]
    summary = {
        "dev_sample": dev_sample,
        "champion": {
            "version": int(version.version),
            "model": version.tags.get("model_type"),
            "method": version.tags.get("calibration"),
            "calibrated_on": version.tags.get("calibrated_on", "full"),
        },
        "baseline": {
            "source": source,
            "n_patients": len(ages),
            "age_mean": means[age_cols[0]["name"]] * enc.age_scale if age_cols else None,
            "male_share": means[sex_cols[0]["name"]] if sex_cols else None,
        },
        "centering_max_abs_mean": float(np.abs(explainer.W.mean(axis=0)).max()),
        "ranking_min_prob": ex.ranking_min_prob,
        "ranking": ranking,
        "latency_ms": latency,
        "examples": examples,
        "global": {cond: [t.to_dict() for t in items] for cond, items in top.items()},
        "images": images,
    }
    review = REVIEW_FILE.read_text(encoding="utf-8") if REVIEW_FILE.is_file() else None
    report = reports / f"explainability{sfx}.md"
    summary_path = reports / f"explainability_summary{sfx}.json"
    with mlflow.start_run(run_name="explain"):
        mlflow.set_tags({"model_version": version.version, "dev_sample": str(dev_sample)})
        mlflow.log_metrics(
            {
                "latency_median_ms": latency["median"],
                "latency_p95_ms": latency["p95"],
                "centering_max_abs_mean": summary["centering_max_abs_mean"],
                **{f"{e}_{m}": v for e, r in ranking.items() for m, v in r.items()},
            }
        )
        report.write_text(render_explainability(summary, review_md=review), encoding="utf-8")
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for path in (report, summary_path, baseline_path, *[reports / i for i in images.values()]):
            mlflow.log_artifact(str(path))
    return summary


def main() -> int:
    summary = run_explain(load_config())
    print(json.dumps({k: summary[k] for k in ("champion", "latency_ms", "ranking")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

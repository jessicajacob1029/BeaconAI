"""Tune every model on the validation split, log each trial to MLflow, register the champion.

Run: uv run python -m beaconai.models.train [--winners-only]
--winners-only retrains only the best params per model from the last train_summary.json
(used for the final full-data run, so the grid search itself stays on the dev subsample).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from beaconai.config import (
    Config,
    MlflowConfig,
    TrainConfig,
    load_config,
    project_path,
    require_training,
)
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.metrics import evaluate_predictions
from beaconai.eval.plots import plot_confusion
from beaconai.models.base import ModelWrapper
from beaconai.models.dataset import (
    SplitData,
    load_feature_names,
    load_labels,
    load_manifest,
    load_split,
)
from beaconai.models.pyfunc import ARTIFACT_KEY, BeaconPyfunc
from beaconai.models.registry import build_model
from beaconai.models.seed import set_global_seed
from beaconai.models.tuning import TrialResult, expand_grid, flatten_params, select_best

SUMMARY_FILE = "train_summary.json"


class DataScopeError(RuntimeError):
    """The processed data on disk is not the data the config or a training run expects."""


def data_scope(processed: Path) -> int | None:
    """dev_sample the processed data on disk was built with (None = full dataset)."""
    return load_manifest(processed).get("settings", {}).get("dev_sample")


def check_data_scope(expected: int | None, on_disk: int | None) -> None:
    if expected != on_disk:
        raise DataScopeError(
            f"Processed data was built with dev_sample={on_disk}, but dev_sample={expected} "
            "is expected. Rebuild it first: `make data` for the dev subsample, or "
            "`BEACONAI_DEV_SAMPLE=null uv run python -m beaconai.data.preprocess` for full data."
        )


PIP_REQUIREMENTS = ["numpy", "scipy", "pandas", "scikit-learn", "xgboost", "torch"]


def configure_mlflow(cfg: MlflowConfig) -> None:
    db = project_path(cfg.tracking_db)
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    if mlflow.get_experiment_by_name(cfg.experiment) is None:
        root = project_path(cfg.artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(cfg.experiment, artifact_location=root.as_uri())
    mlflow.set_experiment(cfg.experiment)


def load_wrapper(model_uri: str) -> ModelWrapper:
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().model


def measure_latency_ms(model: ModelWrapper, data: SplitData, n_calls: int) -> float:
    """Median time of a single-patient predict_proba call, the shape the API will use."""
    times = []
    for i in range(min(n_calls, data.n)):
        row = data.X[i : i + 1]
        start = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def trial_plan(cfg: Config, winners_only: bool, summary_path: Path) -> list[tuple[str, dict]]:
    _, train_cfg, models_cfg, _ = require_training(cfg)
    if winners_only:
        if not summary_path.is_file():
            raise FileNotFoundError(f"{summary_path} not found; run a grid search first")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))["models"]
        return [(name, previous[name]["params"]) for name in train_cfg.models]
    plan = []
    for name in train_cfg.models:
        spec = getattr(models_cfg, name)
        plan += [(name, params) for params in expand_grid(spec.fixed, spec.grid)]
    return plan


def run_trial(
    name: str,
    params: dict[str, Any],
    index: int,
    train: SplitData,
    val: SplitData,
    class_names: list[str],
    feature_names: list[str],
    cfg: Config,
    train_cfg: TrainConfig,
) -> TrialResult:
    _, _, _, eval_cfg = require_training(cfg)
    set_global_seed(cfg.seed)
    with mlflow.start_run(run_name=f"{name}-{index}", nested=True) as run:
        mlflow.set_tags({"model_type": name, "tuned_on": "validate"})
        mlflow.log_params(flatten_params(params))
        model = build_model(name, params, len(class_names), cfg.seed)
        start = time.perf_counter()
        model.fit(train.X, train.y, val.X, val.y)
        fit_seconds = time.perf_counter() - start
        val_proba = model.predict_proba(val.X)
        result = evaluate_predictions(val.y, val_proba, val.ddx, eval_cfg.ks, eval_cfg.ddx_ks)
        latency = measure_latency_ms(model, val, train_cfg.latency_calls)
        mlflow.log_metrics({f"val_{k}": v for k, v in result.metrics.items()})
        mlflow.log_metrics({"fit_seconds": fit_seconds, "latency_ms": latency})
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            figure = plot_confusion(
                result.confusion,
                class_names,
                tmp_path / "val_confusion.png",
                f"{name} (validation)",
            )
            mlflow.log_artifact(str(figure))
            model.save(tmp_path / "wrapper")
            # Named columns give the logged model a feature-level input schema for serving.
            example = pd.DataFrame(val.X[:5].toarray().astype(np.float32), columns=feature_names)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=BeaconPyfunc(),
                artifacts={ARTIFACT_KEY: str(tmp_path / "wrapper")},
                pip_requirements=PIP_REQUIREMENTS,
                signature=infer_signature(example, val_proba[:5]),
                input_example=example,
            )
    return TrialResult(
        name, params, run.info.run_id, info.model_uri, result.metrics, fit_seconds, latency
    )


def register_champion(
    champion: TrialResult, cfg: MlflowConfig, metric: str, dev_sample: int | None, mode: str
) -> tuple[int, str]:
    """Register the winner. Only full-data runs move the champion alias."""
    mv = mlflow.register_model(champion.model_uri, cfg.registered_model)
    alias = cfg.champion_alias if dev_sample is None else cfg.dev_alias
    client = MlflowClient()
    client.set_registered_model_alias(cfg.registered_model, alias, mv.version)
    tags = {
        "model_type": champion.model,
        f"val_{metric}": str(champion.val_metrics[metric]),
        "dev_sample": str(dev_sample),
        "mode": mode,
    }
    for key, value in tags.items():
        client.set_model_version_tag(cfg.registered_model, mv.version, key, value)
    return int(mv.version), alias


def run_training(cfg: Config, *, winners_only: bool = False) -> dict[str, Any]:
    mlflow_cfg, train_cfg, _, eval_cfg = require_training(cfg)
    processed = processed_dir_from_config(cfg.data.paths.processed)
    reports = project_path(eval_cfg.output_dir)
    summary_path = reports / SUMMARY_FILE
    dev_sample = data_scope(processed)
    check_data_scope(cfg.data.dev_sample, dev_sample)
    plan = trial_plan(cfg, winners_only, summary_path)

    class_names = [lab["name"] for lab in load_labels(processed)]
    feature_names = load_feature_names(processed)
    train, val = load_split(processed, "train"), load_split(processed, "validate")
    configure_mlflow(mlflow_cfg)
    metric = train_cfg.selection_metric
    mode = "winners" if winners_only else "grid"

    with mlflow.start_run(run_name="phase2-train") as parent:
        mlflow.set_tags(
            {
                "dev_sample": str(dev_sample),
                "mode": mode,
                "selection_metric": f"val_{metric}",
            }
        )
        mlflow.log_dict(load_manifest(processed), "data_manifest.json")
        results = [
            run_trial(name, params, i, train, val, class_names, feature_names, cfg, train_cfg)
            for i, (name, params) in enumerate(plan)
        ]
        ties = train_cfg.tie_breakers
        best = {
            name: select_best([r for r in results if r.model == name], metric, ties)
            for name in train_cfg.models
        }
        champion = select_best(list(best.values()), metric, ties)
        mlflow.log_metric(f"champion_val_{metric}", champion.val_metrics[metric])
        mlflow.set_tag("champion", champion.model)

    version, alias = register_champion(champion, mlflow_cfg, metric, dev_sample, mode)
    summary = {
        "parent_run_id": parent.info.run_id,
        "dev_sample": dev_sample,
        "mode": mode,
        "selection_metric": metric,
        "tie_breakers": list(train_cfg.tie_breakers),
        "champion": champion.model,
        "registered": {"name": mlflow_cfg.registered_model, "version": version, "alias": alias},
        "models": {
            name: {
                "run_id": r.run_id,
                "model_uri": r.model_uri,
                "params": r.params,
                "val_metrics": r.val_metrics,
                "fit_seconds": r.fit_seconds,
                "latency_ms": r.latency_ms,
                "n_trials": sum(1 for t in results if t.model == name),
            }
            for name, r in best.items()
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winners-only", action="store_true")
    args = parser.parse_args()
    summary = run_training(load_config(), winners_only=args.winners_only)
    print(json.dumps({k: summary[k] for k in ("champion", "registered")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

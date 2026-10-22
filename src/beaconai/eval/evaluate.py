"""Compare the tuned Phase 2 models on validation and write reports/model_comparison_val.md.

Run: uv run python -m beaconai.eval.evaluate --split validate   (any time while iterating)

The test split is not scored here. The only test evaluation is the one-time evaluation of the
calibrated champion: `make eval` (beaconai.eval.final).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mlflow

from beaconai.config import Config, load_config, project_path, require_training
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.metrics import EvalResult, evaluate_predictions, top_confusions
from beaconai.eval.plots import plot_confusion, plot_per_class_recall
from beaconai.eval.report import ComparisonRow, ConfusionPair, render_comparison
from beaconai.models.dataset import SplitData, load_labels, load_split
from beaconai.models.train import (
    SUMMARY_FILE,
    DataScopeError,
    check_data_scope,
    configure_mlflow,
    data_scope,
    load_wrapper,
)

SPLITS = ("validate",)
SUFFIX = {"validate": "_val"}


class TestGuardError(Exception):
    """Raised when something asks this module to score the test split."""

    __test__ = False  # not a pytest test class


def _score(model_uri: str, data: SplitData, cfg: Config) -> EvalResult:
    _, _, _, eval_cfg = require_training(cfg)
    proba = load_wrapper(model_uri).predict_proba(data.X)
    return evaluate_predictions(data.y, proba, data.ddx, eval_cfg.ks, eval_cfg.ddx_ks)


def _confusion_pairs(
    result: EvalResult, labels: list[dict[str, Any]], n: int
) -> list[ConfusionPair]:
    support = result.confusion.sum(axis=1)
    return [
        ConfusionPair(
            labels[t]["name"],
            labels[p]["name"],
            c,
            c / support[t],
            labels[t].get("severity"),
            labels[p].get("severity"),
        )
        for t, p, c in top_confusions(result.confusion, n)
    ]


def run_evaluation(cfg: Config, split: str) -> Path:
    """Write the validation comparison report and return its path."""
    if split == "test":
        raise TestGuardError(
            "The test split is scored only for the calibrated champion: run `make eval`."
        )
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; training data is never scored here")
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    reports = project_path(eval_cfg.output_dir)
    summary_path = reports / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} not found. Run `make train` first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    processed = processed_dir_from_config(cfg.data.paths.processed)
    check_data_scope(summary["dev_sample"], data_scope(processed))

    labels = load_labels(processed)
    names = [lab["name"] for lab in labels]
    data = load_split(processed, split)
    champion = summary["champion"]
    configure_mlflow(mlflow_cfg)

    with mlflow.start_run(run_name=f"phase2-eval-{split}"):
        mlflow.set_tags({"split": split, "train_parent_run_id": summary["parent_run_id"]})
        results: dict[str, EvalResult] = {}
        for name, info in summary["models"].items():
            # Validation metrics were computed at training time; only the champion is
            # re-scored on validation, for its confusion matrix and recall plot.
            if name == champion:
                results[name] = _score(info["model_uri"], data, cfg)
        rows = [
            ComparisonRow(
                name,
                info["params"],
                info["val_metrics"],
                None,
                info["fit_seconds"],
                info["latency_ms"],
            )
            for name, info in summary["models"].items()
        ]

        champ = results[champion]
        sfx = SUFFIX[split]
        cm_name, recall_name = f"confusion_matrix{sfx}.png", f"per_class_recall{sfx}.png"
        figures = reports / "figures"
        cm_path = plot_confusion(champ.confusion, names, figures / cm_name, f"{champion} ({split})")
        recall_path = plot_per_class_recall(
            champ.recall,
            names,
            figures / recall_name,
            f"{champion} per-class recall ({split})",
        )
        lowest = sorted(
            ((names[i], float(r)) for i, r in enumerate(champ.recall) if not math.isnan(r)),
            key=lambda item: item[1],
        )[: eval_cfg.lowest_recall]
        scope = "dev subsample" if summary["dev_sample"] else "full dataset"
        md = render_comparison(
            rows,
            split=split,
            champion=champion,
            selection_metric=summary["selection_metric"],
            tie_breakers=summary.get("tie_breakers", []),
            dataset_note=f"DDXPlus {scope}, {data.n:,} {split} patients",
            confusions=_confusion_pairs(champ, labels, eval_cfg.top_confusions),
            lowest_recall=lowest,
            confusion_image=f"figures/{cm_name}",
            recall_image=f"figures/{recall_name}",
        )
        report = reports / f"model_comparison{sfx}.md"
        report.write_text(md, encoding="utf-8")
        for path in (report, cm_path, recall_path):
            mlflow.log_artifact(str(path))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validate")
    args = parser.parse_args()
    try:
        print(run_evaluation(load_config(), args.split))
    except (TestGuardError, DataScopeError) as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

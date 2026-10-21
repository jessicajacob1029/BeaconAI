"""Score the tuned models on a split, draw figures, and write reports/model_comparison.md.

Run: uv run python -m beaconai.eval.evaluate --split validate   (any time while iterating)
     uv run python -m beaconai.eval.evaluate --split test       (once, full data, at the end)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from beaconai.config import Config, load_config, project_path, require_training
from beaconai.data.preprocess import processed_dir_from_config
from beaconai.eval.metrics import EvalResult, evaluate_predictions, top_confusions
from beaconai.eval.plots import plot_confusion, plot_per_class_recall
from beaconai.eval.report import ComparisonRow, ConfusionPair, render_comparison
from beaconai.models.dataset import SplitData, load_labels, load_split
from beaconai.models.train import SUMMARY_FILE, configure_mlflow, load_wrapper

LOCK_FILE = "test_eval_lock.json"
REPORT_FILE = "model_comparison.md"
SPLITS = ("validate", "test")


class TestGuardError(Exception):
    """Raised when a test-split evaluation would break the score-test-once rule."""

    __test__ = False  # not a pytest test class


def check_test_allowed(summary: dict[str, Any], lock_path: Path, force: bool) -> None:
    if force:
        return
    if summary["dev_sample"] is not None:
        raise TestGuardError(
            "Test evaluation needs the full dataset (data.dev_sample: null). "
            "Use --split validate while iterating."
        )
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        raise TestGuardError(
            f"Test split was already evaluated for training run {lock['train_parent_run_id']}. "
            "Scoring it again after changes leaks test information. Pass --force only if you "
            "accept that and will report it."
        )


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


def run_evaluation(cfg: Config, split: str, *, force: bool = False) -> Path:
    """Write the comparison report for `split` and return its path."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; training data is never scored here")
    mlflow_cfg, _, _, eval_cfg = require_training(cfg)
    reports = project_path(eval_cfg.output_dir)
    summary_path = reports / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} not found. Run `make train` first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock_path = reports / LOCK_FILE
    if split == "test":
        check_test_allowed(summary, lock_path, force)

    processed = processed_dir_from_config(cfg.data.paths.processed)
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
            if split == "test" or name == champion:
                results[name] = _score(info["model_uri"], data, cfg)
            if split == "test":
                metrics = results[name].metrics
                mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})
        rows = [
            ComparisonRow(
                name,
                info["params"],
                info["val_metrics"],
                results[name].metrics if split == "test" else None,
                info["fit_seconds"],
                info["latency_ms"],
            )
            for name, info in summary["models"].items()
        ]

        champ = results[champion]
        figures = reports / "figures"
        cm_path = plot_confusion(
            champ.confusion, names, figures / "confusion_matrix.png", f"{champion} ({split})"
        )
        recall_path = plot_per_class_recall(
            champ.recall,
            names,
            figures / "per_class_recall.png",
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
            confusion_image="figures/confusion_matrix.png",
            recall_image="figures/per_class_recall.png",
        )
        report = reports / REPORT_FILE
        report.write_text(md, encoding="utf-8")
        for path in (report, cm_path, recall_path):
            mlflow.log_artifact(str(path))

    if split == "test":
        lock = {
            "train_parent_run_id": summary["parent_run_id"],
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="validate")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        print(run_evaluation(load_config(), args.split, force=args.force))
    except TestGuardError as err:
        print(f"Refused: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

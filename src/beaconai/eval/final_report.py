"""Render reports/final_evaluation.md. Pure string building."""

from __future__ import annotations

from typing import Any

DISCLAIMER = (
    "BeaconAI is decision support, not diagnosis. This model is trained only on synthetic "
    "data (DDXPlus) and is not a medical device."
)
ROWS = (("Full evidence", "full"), ("Partial evidence", "partial"))


def _f(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_final(summary: dict[str, Any]) -> str:
    champ, reg = summary["champion"], summary["registered"]
    m, abst = summary["metrics"], summary["abstention"]
    out = [
        "# Final test evaluation",
        "",
        f"> {DISCLAIMER}",
        "",
        f"Model: `{reg['name']}` version {reg['version']} (alias `{reg['alias']}`): "
        f"{champ['model']} + {champ['method']} (calibrated on {champ['fit_on']}).",
        f"Test patients: {summary['n_test']:,}. The test split is scored once for this "
        f"evaluation; `reports/test_locks/{summary['evaluation']}.json` blocks repeat scoring.",
        "Partial evidence: each patient keeps the initial evidence plus 3 other random "
        "evidences, the same rule used on validation.",
        "",
        "**Any classifier change after this point is validated on validation data only.** "
        "These test numbers describe this model version and are not re-measured.",
        "",
        "## Accuracy and calibration",
        "",
        "| Evidence | Top-1 | Top-3 | Top-5 | Macro F1 | DDx nR@3 | ECE | NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {label} | {_f(m[key].get('top1_accuracy'))} | {_f(m[key].get('top3_accuracy'))} | "
            f"{_f(m[key].get('top5_accuracy'))} | {_f(m[key].get('macro_f1'))} | "
            f"{_f(m[key].get('ddx_recall_norm_at3'))} | {_f(m[key].get('ece'), 4)} | "
            f"{_f(m[key].get('nll'))} |"
            for label, key in ROWS
        ],
        "",
        "## Abstention",
        "",
        f"Threshold: {summary['threshold']:.3f}. BeaconAI abstains when the calibrated "
        "probability of the most likely condition is below it.",
        "",
        "| Evidence | Answered | Top-1 on answered | Top-3 on answered |",
        "|---|---:|---:|---:|",
        *[
            f"| {label} | {abst[key]['coverage']:.1%} | {_f(abst[key]['top1_accuracy'])} | "
            f"{_f(abst[key]['topk_accuracy'])} |"
            for label, key in ROWS
        ],
        "",
        *[
            f"Note ({label.lower()}): {abst[key]['note']}"
            for label, key in ROWS
            if "note" in abst[key]
        ],
        "",
        "![Confusion matrix](figures/final_confusion_matrix.png)",
        "",
    ]
    return "\n".join(out)

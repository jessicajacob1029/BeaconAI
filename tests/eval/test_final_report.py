import unicodedata

from beaconai.eval.final_report import render_final

SUMMARY = {
    "evaluation": "classifier_final",
    "registered": {"name": "beaconai-classifier", "version": 9, "alias": "champion"},
    "champion": {"model": "logreg+dropout", "method": "temperature", "fit_on": "partial"},
    "n_test": 134529,
    "threshold": 0.84,
    "metrics": {
        "full": {
            "top1_accuracy": 0.996,
            "top3_accuracy": 1.0,
            "top5_accuracy": 1.0,
            "macro_f1": 0.995,
            "ece": 0.001,
            "nll": 0.01,
            "ddx_recall_norm_at3": 0.70,
        },
        "partial": {
            "top1_accuracy": 0.7,
            "top3_accuracy": 0.85,
            "top5_accuracy": 0.9,
            "macro_f1": 0.6,
            "ece": 0.03,
            "nll": 1.0,
            "ddx_recall_norm_at3": 0.5,
        },
    },
    "abstention": {
        "full": {"coverage": 0.99, "top1_accuracy": 0.999, "topk_accuracy": 1.0},
        "partial": {
            "coverage": 0.0,
            "top1_accuracy": None,
            "topk_accuracy": None,
            "note": "No patients are answered at threshold 0.840; accuracy is undefined.",
        },
    },
}


def test_render_final() -> None:
    md = render_final(SUMMARY)
    assert "decision support, not diagnosis" in md
    assert "logreg+dropout + temperature (calibrated on partial)" in md
    assert "| Partial evidence | 0.700 | 0.850 |" in md
    assert "No patients are answered" in md  # a None accuracy shows its note, not NaN
    assert "scored once" in md and "test_locks/classifier_final.json" in md
    assert "validated on validation data only" in md
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}  # docs style: no em dashes

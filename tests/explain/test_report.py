import unicodedata

from beaconai.explain.report import render_explainability

SUMMARY = {
    "dev_sample": None,
    "champion": {
        "version": 13,
        "model": "logreg+dropout",
        "method": "isotonic",
        "calibrated_on": "partial",
    },
    "baseline": {
        "source": "raw train CSV (full training set)",
        "n_patients": 1025602,
        "age_mean": 39.7,
        "male_share": 0.48,
    },
    "centering_max_abs_mean": 0.015,
    "ranking_min_prob": 0.01,
    "ranking": {
        "full": {"top1_changed": 0.001, "topk_set_changed": 0.004, "shown_set_changed": 0.0006},
        "partial": {"top1_changed": 0.012, "topk_set_changed": 0.05, "shown_set_changed": 0.003},
    },
    "latency_ms": {"median": 2.1, "p95": 3.4, "budget": 50, "n": 300},
    "examples": ["Why Pneumonia rather than Bronchitis: Fever?: yes (+1.20)"],
    "global": {
        "Pneumonia": [
            {
                "index": 3,
                "name": "E_91",
                "label": "Fever?",
                "coef": 2.5,
                "prevalence_condition": 0.9,
                "prevalence_all": 0.3,
                "flags": [],
            },
            {
                "index": 9,
                "name": "SEX_M",
                "label": "Sex: male",
                "coef": 0.4,
                "prevalence_condition": 0.5,
                "prevalence_all": 0.5,
                "flags": ["demographic"],
            },
        ]
    },
    "images": {"Pneumonia": "figures/top_features_pneumonia.png"},
}


def test_report_sections_and_wording() -> None:
    md = render_explainability(SUMMARY, review_md="## Human review\n\nLooks sensible.")
    assert "decision support, not diagnosis" in md
    assert "logit space before isotonic calibration" in md
    assert "relative to the average condition" in md
    assert "training-average age and sex" in md and "1,025,602" in md and "48% male" in md
    assert "| Full evidence | 0.1% | 0.4% | 0.06% |" in md
    assert "2.1" in md and "3.4" in md and "50" in md
    assert "![Pneumonia](figures/top_features_pneumonia.png)" in md
    assert "| Fever? | 2.500 | 90.0% | 30.0% |  |" in md
    assert "demographic" in md and "## Human review" in md
    assert "Why Pneumonia rather than Bronchitis" in md
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}  # docs style: no em dashes


def test_report_without_review_says_pending() -> None:
    assert "Human review pending" in render_explainability(SUMMARY, review_md=None)


def test_latency_over_budget_is_called_out() -> None:
    slow = {**SUMMARY, "latency_ms": {**SUMMARY["latency_ms"], "p95": 80}}
    assert "over the 50 ms budget" in render_explainability(slow, review_md=None)
    assert "within the 50 ms budget" in render_explainability(SUMMARY, review_md=None)

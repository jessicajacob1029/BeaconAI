import unicodedata

from beaconai.eval.calibration_report import render_calibration


def _metrics(ece: float, top3: float, partial_ece: float, partial_top3: float) -> dict[str, float]:
    return {
        "ece": ece,
        "nll": 0.01,
        "top1_accuracy": 0.99,
        "top3_accuracy": top3,
        "partial_ece": partial_ece,
        "partial_nll": 1.5,
        "partial_top1_accuracy": 0.6,
        "partial_top3_accuracy": partial_top3,
    }


def _cand(model: str, method: str, temp: float | None, *m: float) -> dict:
    return {"model": model, "method": method, "temperature": temp, "metrics": _metrics(*m)}


SUMMARY = {
    "dev_sample": None,
    "n_fit": 66224,
    "n_held": 66224,
    "top3_floor": {"full": 0.99, "partial": 0.65},
    "candidates": [
        _cand("xgboost", "none", None, 0.0004, 1.0, 0.557, 0.713),
        _cand("logreg", "temperature", 1.3, 0.0010, 1.0, 0.031, 0.767),
        _cand("logreg", "isotonic", None, 0.0002, 0.98, 0.020, 0.770),  # fails the full floor
        _cand("xgboost", "isotonic", None, 0.0003, 1.0, 0.010, 0.600),  # fails the partial floor
    ],
    "champion": {"model": "logreg", "method": "temperature"},
    "registered": {"name": "beaconai-classifier", "version": 4, "alias": "champion"},
    "abstention": {
        "threshold": 0.5,
        "suggested_threshold": 0.62,
        "target_accuracy": 0.95,
        "k": 3,
        "full": {"coverage": 0.999, "top1_accuracy": 0.998, "topk_accuracy": 1.0},
        "partial": {"coverage": 0.41, "top1_accuracy": 0.81, "topk_accuracy": 0.93},
    },
}


def _render(summary: dict = SUMMARY) -> str:
    return render_calibration(
        summary,
        ece_bins=15,
        reliability_images={"xgboost": "figures/reliability_xgboost.png"},
        coverage_image="figures/coverage_accuracy.png",
    )


def test_champion_rule_is_partial_ece_with_both_floors() -> None:
    md = _render()
    assert "Champion: **logreg + temperature**" in md
    assert "lowest partial-evidence ECE" in md
    assert "at least 0.990 on full evidence" in md
    assert "at least 0.650 on partial evidence" in md
    assert "no, below the full top-3 floor" in md
    assert "no, below the partial top-3 floor" in md


def test_partial_table_is_selection_basis_and_full_is_alongside() -> None:
    md = _render()
    partial_at = md.index("## Partial evidence")
    full_at = md.index("## Full evidence")
    assert partial_at < full_at
    assert "| logreg | temperature | 0.0310 |" in md  # partial ECE
    assert "| logreg | temperature | 0.0010 |" in md  # full ECE


def test_limitation_section_quantifies_the_drop() -> None:
    md = _render()
    assert "## Limitation: not asked is read as no" in md
    assert "| logreg | temperature | 1.000 | 0.767 | 23.3 | 0.0010 | 0.0310 | +0.0300 |" in md
    assert "top-3 accuracy drops from 1.000 to 0.767 (23.3 points)" in md
    assert "ECE rises from 0.0010 to 0.0310" in md


def test_report_abstention_section() -> None:
    md = _render()
    assert "Threshold: 0.50" in md and "Suggested threshold: 0.62" in md
    assert "| Partial evidence | 41.0% | 0.810 | 0.930 |" in md


def test_report_when_no_threshold_meets_target() -> None:
    summary = {**SUMMARY, "abstention": {**SUMMARY["abstention"], "suggested_threshold": None}}
    assert "No threshold reaches" in _render(summary)


def test_report_basics() -> None:
    md = _render()
    assert "decision support, not diagnosis" in md
    assert "![xgboost reliability](figures/reliability_xgboost.png)" in md
    assert "EM DASH" not in {unicodedata.name(ch, "") for ch in md}  # docs style: no em dashes


def test_report_states_held_out_caveat() -> None:
    md = _render()
    assert "not fully independent" in md
    assert "early stopping" in md
    assert "confirmed on the test split" in md

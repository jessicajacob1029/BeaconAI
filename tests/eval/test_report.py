from beaconai.eval.report import ComparisonRow, ConfusionPair, render_comparison

VAL = {
    "top1_accuracy": 0.5,
    "top3_accuracy": 0.8,
    "macro_f1": 0.4,
    "ddx_precision_at3": 0.61,
    "ddx_recall_at3": 0.33,
    "ddx_recall_norm_at3": 0.72,
}
TEST = {**VAL, "top3_accuracy": 0.79, "ddx_recall_norm_at3": 0.71}


def _render(with_test: bool, tie_breakers: tuple[str, ...] = ()) -> str:
    rows = [ComparisonRow("xgboost", {"max_depth": 6}, VAL, TEST if with_test else None, 12.3, 0.8)]
    return render_comparison(
        rows,
        split="test" if with_test else "validate",
        champion="xgboost",
        selection_metric="top3_accuracy",
        tie_breakers=tie_breakers,
        dataset_note="Full DDXPlus",
        confusions=[ConfusionPair("Bronchitis", "Pneumonia", 42, 0.12, 4, 3)],
        lowest_recall=[("Bronchitis", 0.31)],
        confusion_image="figures/confusion_matrix.png",
        recall_image="figures/per_class_recall.png",
    )


def test_report_contents() -> None:
    md = _render(with_test=True)
    assert "decision support, not diagnosis" in md
    assert "| xgboost |" in md and "0.800" in md and "0.790" in md
    assert "![Confusion matrix](figures/confusion_matrix.png)" in md
    assert "Bronchitis" in md and "Pneumonia" in md
    assert "**xgboost**" in md  # champion called out
    assert "\u2014" not in md  # no em dashes


def test_headline_ddx_metric_is_normalized_recall_at3() -> None:
    md = _render(with_test=True)
    header = next(line for line in md.splitlines() if line.startswith("| Model | Top-1"))
    assert header.rstrip(" |").endswith("DDx nR@3")
    assert "0.720" in md and "0.710" in md


def test_differential_table_lists_all_three_overlap_metrics() -> None:
    md = _render(with_test=True)
    assert "| Model | k | Precision@k | Recall@k (raw) | Recall@k (normalized) |" in md
    assert "| xgboost | 3 | 0.610 | 0.330 | 0.710 |" in md  # test split values


def test_validation_only_report_says_test_pending() -> None:
    md = _render(with_test=False)
    assert "Test split not evaluated yet" in md
    assert "| xgboost | 3 | 0.610 | 0.330 | 0.720 |" in md  # validation values


def test_report_names_tie_breakers() -> None:
    md = _render(with_test=True, tie_breakers=("top1_accuracy", "macro_f1"))
    assert "Exact ties are broken by `top1_accuracy`, then `macro_f1`." in md
    assert "Exact ties" not in _render(with_test=True)

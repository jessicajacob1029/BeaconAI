import pytest

from beaconai.models.tuning import TrialResult, expand_grid, flatten_params, select_best


def test_expand_grid_product_and_order() -> None:
    trials = expand_grid({"a": 1}, {"z": [1, 2], "b": ["x", "y"]})
    assert trials == [
        {"a": 1, "b": "x", "z": 1},
        {"a": 1, "b": "x", "z": 2},
        {"a": 1, "b": "y", "z": 1},
        {"a": 1, "b": "y", "z": 2},
    ]


def test_expand_grid_without_grid_is_one_trial() -> None:
    assert expand_grid({"a": 1}, {}) == [{"a": 1}]


def test_flatten_params_stringifies_lists() -> None:
    assert flatten_params({"h": [256, 128], "lr": 0.1}) == {"h": "[256, 128]", "lr": 0.1}


def _r(name: str, top3: float) -> TrialResult:
    return TrialResult(name, {}, f"run-{name}", "uri", {"top3_accuracy": top3}, 1.0, 1.0)


def test_select_best_and_ties() -> None:
    assert select_best([_r("a", 0.5), _r("b", 0.9), _r("c", 0.9)], "top3_accuracy").model == "b"


def test_select_best_rejects_empty_and_unknown_metric() -> None:
    with pytest.raises(ValueError):
        select_best([], "top3_accuracy")
    with pytest.raises(KeyError):
        select_best([_r("a", 0.5)], "auc")


def _m(name: str, **metrics: float) -> TrialResult:
    return TrialResult(name, {}, f"run-{name}", "uri", metrics, 1.0, 1.0)


def test_select_best_breaks_ties_with_secondary_metrics() -> None:
    trials = [
        _m("a", top3_accuracy=1.0, top1_accuracy=0.990, macro_f1=0.99),
        _m("b", top3_accuracy=1.0, top1_accuracy=0.996, macro_f1=0.98),
        _m("c", top3_accuracy=1.0, top1_accuracy=0.996, macro_f1=0.99),
        _m("d", top3_accuracy=0.9, top1_accuracy=0.999, macro_f1=0.999),
    ]
    best = select_best(trials, "top3_accuracy", ("top1_accuracy", "macro_f1"))
    assert best.model == "c"  # primary metric still decides first: d loses on top-3

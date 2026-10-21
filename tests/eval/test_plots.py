from pathlib import Path

import numpy as np

from beaconai.eval.plots import plot_confusion, plot_per_class_recall


def test_plots_write_png(tmp_path: Path) -> None:
    cm = np.array([[3, 1, 0], [0, 4, 0], [0, 0, 0]])  # class C has no examples
    names = ["A", "B", "C"]
    p1 = plot_confusion(cm, names, tmp_path / "out" / "cm.png", "Confusion")
    p2 = plot_per_class_recall(np.array([0.75, 1.0, np.nan]), names, tmp_path / "r.png", "Recall")
    for p in (p1, p2):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_calibration_plots_write_png(tmp_path: Path) -> None:
    from beaconai.eval.calibration import coverage_curve, reliability_bins
    from beaconai.eval.plots import plot_coverage, plot_reliability

    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1, 1])
    bins = reliability_bins(y, proba, 5)
    p1 = plot_reliability(
        {"Full evidence": {"none": bins}, "Partial evidence": {"none": bins}},
        tmp_path / "rel.png",
        "Reliability",
    )
    curve = coverage_curve(y, proba, np.linspace(0, 1, 11), k=1)
    p2 = plot_coverage({"Full evidence": curve}, 0.5, tmp_path / "cov.png", "Coverage")
    for p in (p1, p2):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

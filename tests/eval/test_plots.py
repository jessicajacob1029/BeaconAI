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

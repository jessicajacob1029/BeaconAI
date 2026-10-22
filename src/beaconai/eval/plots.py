"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beaconai.eval.calibration import CoverageCurve, ReliabilityBins


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> Path:
    """Row-normalised confusion matrix: each row shows where that true condition went."""
    support = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, support, out=np.zeros(cm.shape, dtype=float), where=support > 0)
    size = max(6.0, 0.28 * len(class_names))
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=90, fontsize=6)
    ax.set_yticks(ticks, class_names, fontsize=6)
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true condition")
    return _save(fig, path)


def plot_per_class_recall(
    recall: np.ndarray, class_names: list[str], path: Path, title: str
) -> Path:
    """Horizontal bars, lowest recall at the bottom. Classes with no examples show as 0."""
    order = np.argsort(np.nan_to_num(recall, nan=-1.0))
    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.2 * len(class_names))))
    ax.barh(np.arange(len(order)), np.nan_to_num(recall[order]), color="#4c72b0")
    ax.set_yticks(np.arange(len(order)), [class_names[i] for i in order], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_title(title)
    return _save(fig, path)


def plot_reliability(panels: dict[str, dict[str, ReliabilityBins]], path: Path, title: str) -> Path:
    """One panel per evidence setting; one line per calibration method; diagonal = perfect."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
    for ax, (panel, curves) in zip(axes[0], panels.items(), strict=True):
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
        for label, bins in curves.items():
            used = bins.counts > 0
            ax.plot(bins.confidence[used], bins.accuracy[used], marker="o", label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Probability of the top condition")
        ax.set_ylabel("Share where the top condition was right")
        ax.set_title(panel)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_coverage(
    curves: dict[str, CoverageCurve], threshold: float, path: Path, title: str
) -> Path:
    """Accuracy on answered patients as the model answers fewer of them."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, curve in curves.items():
        line = ax.plot(curve.coverage, curve.topk_accuracy, label=f"{label}, top-{curve.k}")[0]
        ax.plot(
            curve.coverage,
            curve.top1_accuracy,
            linestyle=":",
            color=line.get_color(),
            label=f"{label}, top-1",
        )
        i = int(np.argmin(np.abs(curve.thresholds - threshold)))
        ax.scatter(curve.coverage[i], curve.topk_accuracy[i], color=line.get_color(), zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Coverage (share of patients answered)")
    ax.set_ylabel("Accuracy on answered patients")
    ax.set_title(f"{title} (dots: threshold {threshold:.2f})")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_evidence_curve(series: dict[str, dict], path: Path, title: str) -> Path:
    """Top-3 accuracy and ECE against the number of known evidences besides the initial one."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, curve in series.items():
        ax1.plot(curve["extra"], curve["top3_accuracy"], marker="o", label=label)
        ax2.plot(curve["extra"], curve["ece"], marker="o", label=label)
    ax1.set_ylabel("Top-3 accuracy")
    ax2.set_ylabel("ECE (lower is better)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Other evidences known, besides the initial one")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_top_features(features: list, condition: str, path: Path) -> Path:
    """Top coefficients for one condition; hatched bars carry an automatic sanity flag.

    `features` are objects with `label`, `coef` and `flags` (explain.global_view.TopFeature).
    """
    items = list(reversed(features))  # largest at the top
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.45 * len(items) + 1)))
    labels = [f.label if len(f.label) <= 70 else f.label[:67] + "..." for f in items]
    bars = ax.barh(range(len(items)), [f.coef for f in items], color="#4c72b0")
    for bar, f in zip(bars, items, strict=True):
        if f.flags:
            bar.set_hatch("//")
            bar.set_facecolor("#dd8452")
    ax.set_yticks(range(len(items)), labels, fontsize=7)
    ax.set_xlabel("Coefficient (logit, relative to the average condition)")
    ax.set_title(f"{condition}: top features (hatched = flagged for review)")
    return _save(fig, path)

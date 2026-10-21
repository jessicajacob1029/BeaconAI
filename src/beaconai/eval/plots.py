"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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

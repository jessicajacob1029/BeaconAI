"""Exploratory plots for the processed DDXPlus data.

Saves three PNGs to reports/ (from the split set in config `eda.split`):
- eda_class_balance.png         patients per condition
- eda_evidences_per_patient.png how many evidences each patient has
- eda_age_by_sex.png            age distribution, female vs male

Run: uv run python -m beaconai.data.eda  (or `make eda`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from beaconai.config import PROJECT_ROOT, load_config
from beaconai.data.preprocess import processed_dir_from_config

# Chart colors and ink (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2, validated for CVD separation


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _title(ax: plt.Axes, title: str, subtitle: str) -> None:
    # Offsets in points (not axes fractions) so spacing is the same for any figure height.
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22)
    ax.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
    )


def plot_class_balance(y: np.ndarray, class_names: list[str], subtitle: str) -> plt.Figure:
    counts = np.bincount(y, minlength=len(class_names))
    order = np.argsort(counts)
    fig, ax = _new_figure(8, 11)
    ax.barh(
        [class_names[i] for i in order],
        counts[order],
        height=0.7,
        color=SERIES[0],
        edgecolor=SURFACE,
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Patients", color=INK_SECONDARY)
    ax.margins(y=0.01)
    # Label only the smallest and largest classes, so the spread is readable at a glance.
    for i in (order[0], order[-1]):
        ax.annotate(
            f"{counts[i]:,}",
            (counts[i], class_names[i]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    _title(ax, "Class balance", subtitle)
    fig.tight_layout()
    return fig


def plot_evidences_per_patient(n_evidences: np.ndarray, subtitle: str) -> plt.Figure:
    fig, ax = _new_figure(8, 4.5)
    lo, hi = int(n_evidences.min()), int(n_evidences.max())
    ax.hist(n_evidences, bins=np.arange(lo, hi + 2) - 0.5, color=SERIES[0], edgecolor=SURFACE, lw=1)
    ax.grid(axis="x", visible=False)
    mean = float(n_evidences.mean())
    ax.axvline(mean, color=INK_SECONDARY, lw=1, ls="--")
    ax.text(mean, ax.get_ylim()[1] * 0.97, f"  mean {mean:.1f}", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Evidences per patient", color=INK_SECONDARY)
    ax.set_ylabel("Patients", color=INK_SECONDARY)
    _title(ax, "Evidences per patient", subtitle)
    fig.tight_layout()
    return fig


def plot_age_by_sex(age: np.ndarray, sex: np.ndarray, subtitle: str) -> plt.Figure:
    fig, ax = _new_figure(8, 4.5)
    bins = np.arange(0, int(age.max()) + 6, 5)
    # The two curves run close together, so one label sits above its peak and one below.
    series = (("F", "Female", 6, "bottom"), ("M", "Male", -6, "top"))
    for (code, name, dy, va), color in zip(series, SERIES, strict=True):
        counts, edges = np.histogram(age[sex == code], bins=bins)
        ax.stairs(counts, edges, color=color, lw=2, label=f"{name} ({(sex == code).sum():,})")
        peak = int(np.argmax(counts))
        ax.annotate(
            name,
            (edges[peak] + 2.5, counts[peak]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Age (years, 5-year bins)", color=INK_SECONDARY)
    ax.set_ylabel("Patients", color=INK_SECONDARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    _title(ax, "Age distribution by sex", subtitle)
    fig.tight_layout()
    return fig


def run_eda(processed_dir: Path, split: str, out_dir: Path) -> list[Path]:
    y = np.load(processed_dir / f"{split}_y.npy")
    meta = np.load(processed_dir / f"{split}_meta.npz")
    labels = json.loads((processed_dir / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    class_names = [c["name"] for c in sorted(labels, key=lambda c: c["index"])]

    dev = manifest["settings"]["dev_sample"]
    sample_note = f"dev subsample of {dev:,} per split" if dev else "full split"
    subtitle = f"DDXPlus {split} split, {len(y):,} patients ({sample_note}). Synthetic data."

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "eda_class_balance.png": plot_class_balance(y, class_names, subtitle),
        "eda_evidences_per_patient.png": plot_evidences_per_patient(meta["n_evidences"], subtitle),
        "eda_age_by_sex.png": plot_age_by_sex(meta["age"], meta["sex"], subtitle),
    }
    paths = []
    for name, fig in figures.items():
        path = out_dir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    cfg = load_config()
    out_dir = cfg.eda.output_dir
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    for path in run_eda(
        processed_dir_from_config(cfg.data.paths.processed), cfg.eda.split, out_dir
    ):
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

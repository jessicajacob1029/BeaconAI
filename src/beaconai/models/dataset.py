"""Load the Phase 1 processed arrays (see beaconai.data.preprocess for the format)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SplitData:
    X: sparse.csr_matrix
    y: np.ndarray
    ddx: sparse.csr_matrix | None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def load_split(processed_dir: Path, split: str) -> SplitData:
    x_path = processed_dir / f"{split}_X.npz"
    if not x_path.is_file():
        raise FileNotFoundError(f"{x_path} not found. Run `make data` first.")
    ddx_path = processed_dir / f"{split}_ddx.npz"
    return SplitData(
        X=sparse.load_npz(x_path).tocsr(),
        y=np.load(processed_dir / f"{split}_y.npy").astype(np.int64),
        ddx=sparse.load_npz(ddx_path).tocsr() if ddx_path.is_file() else None,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(processed_dir: Path) -> list[dict[str, Any]]:
    return _read_json(processed_dir / "labels.json")


def load_manifest(processed_dir: Path) -> dict[str, Any]:
    return _read_json(processed_dir / "manifest.json")


def subset(data: SplitData, idx: np.ndarray) -> SplitData:
    """Rows `idx` of a split, with X, y and the reference differential kept aligned."""
    return SplitData(
        X=data.X[idx],
        y=data.y[idx],
        ddx=None if data.ddx is None else data.ddx[idx],
    )


def load_meta(processed_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Raw age, sex, initial evidence code and evidence count per patient."""
    path = processed_dir / f"{split}_meta.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    with np.load(path) as meta:
        return {key: meta[key] for key in meta.files}


def load_features(processed_dir: Path) -> list[dict[str, Any]]:
    """features.json entries in column order."""
    return sorted(_read_json(processed_dir / "features.json"), key=lambda f: f["index"])


def load_feature_names(processed_dir: Path) -> list[str]:
    """Column names in matrix order, e.g. `E_55_@_V_14`."""
    return [f["name"] for f in load_features(processed_dir)]

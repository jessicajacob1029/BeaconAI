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

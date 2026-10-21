"""Turn raw DDXPlus files into model-ready arrays in data/processed/ddxplus/.

Outputs per split (train, validate, test), keeping the official split:
- `{split}_X.npz`    float32 CSR feature matrix (patients x features)
- `{split}_y.npy`    int16 class id of the true condition (index into labels.json)
- `{split}_ddx.npz`  float32 CSR ground-truth differential (patients x classes)
- `{split}_meta.npz` raw age, sex, initial evidence code, and evidence count per patient
Shared:
- `features.json`  one entry per column: name, kind, readable label, evidence, value
- `labels.json`    one entry per class: English and French name, ICD-10, severity
- `manifest.json`  row counts, feature counts, and the settings used

Run: uv run python -m beaconai.data.preprocess
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from beaconai.config import PROJECT_ROOT, Config, load_config
from beaconai.data.download import DATASET_SUBDIR, raw_dir_from_config
from beaconai.data.encode import (
    FeatureSpace,
    encode_differentials,
    parse_differential,
    parse_evidence_list,
)
from beaconai.data.sample import subsample
from beaconai.data.schema import Catalog, load_catalog

SPLITS = ("train", "validate", "test")
EVIDENCES_FILE = "release_evidences.json"
CONDITIONS_FILE = "release_conditions.json"
REQUIRED_COLUMNS = ("AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE")


def split_csv_name(split: str) -> str:
    return f"release_{split}_patients.csv"


def processed_dir_from_config(processed: Path) -> Path:
    processed = processed if processed.is_absolute() else PROJECT_ROOT / processed
    return processed / DATASET_SUBDIR


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return df


def encode_split(
    df: pd.DataFrame, space: FeatureSpace, class_index: dict[str, int]
) -> dict[str, Any]:
    """Encode one split's rows. Returns arrays ready to save."""
    evidence_lists = [parse_evidence_list(s) for s in df["EVIDENCES"]]
    ages = df["AGE"].astype(int).to_numpy()
    sexes = df["SEX"].to_numpy()

    unknown = set(df["PATHOLOGY"]) - set(class_index)
    if unknown:
        raise KeyError(f"Unknown PATHOLOGY values: {sorted(unknown)[:5]}")

    out: dict[str, Any] = {
        "X": space.encode_many(evidence_lists, ages, sexes),
        "y": df["PATHOLOGY"].map(class_index).to_numpy(dtype=np.int16),
        "meta": {
            "age": ages.astype(np.int16),
            "sex": sexes.astype("U1"),
            "initial_evidence": df["INITIAL_EVIDENCE"].to_numpy().astype(str),
            "n_evidences": np.array([len(e) for e in evidence_lists], dtype=np.int16),
        },
    }
    if "DIFFERENTIAL_DIAGNOSIS" in df.columns:
        out["ddx"] = encode_differentials(
            (parse_differential(s) for s in df["DIFFERENTIAL_DIAGNOSIS"]), class_index
        )
    return out


def labels_payload(catalog: Catalog) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "name": name,
            "name_fr": catalog.conditions[name].name_fr,
            "icd10": catalog.conditions[name].icd10,
            "severity": catalog.conditions[name].severity,
        }
        for i, name in enumerate(catalog.condition_names)
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_preprocess(
    raw_dir: Path,
    out_dir: Path,
    *,
    age_bins: list[int],
    age_scale: float,
    scale_max: float,
    dev_sample: int | None,
    seed: int,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Run the full raw -> processed pipeline. Returns the manifest."""
    catalog = load_catalog(raw_dir / EVIDENCES_FILE, raw_dir / CONDITIONS_FILE)
    space = FeatureSpace(catalog, age_bins, age_scale=age_scale, scale_max=scale_max)
    class_index = {name: i for i, name in enumerate(catalog.condition_names)}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    split_rows: dict[str, int] = {}
    for split in splits:
        df = read_split(raw_dir / split_csv_name(split))
        total = len(df)
        df = subsample(df, dev_sample, seed).reset_index(drop=True)
        print(f"{split}: encoding {len(df):,} of {total:,} patients ...", flush=True)
        enc = encode_split(df, space, class_index)
        sparse.save_npz(out_dir / f"{split}_X.npz", enc["X"])
        np.save(out_dir / f"{split}_y.npy", enc["y"])
        np.savez_compressed(out_dir / f"{split}_meta.npz", **enc["meta"])
        if "ddx" in enc:
            sparse.save_npz(out_dir / f"{split}_ddx.npz", enc["ddx"])
        split_rows[split] = len(df)

    _write_json(out_dir / "features.json", [f.to_dict() for f in space.features])
    _write_json(out_dir / "labels.json", labels_payload(catalog))
    manifest = {
        "dataset": "DDXPlus",
        "splits": split_rows,
        "n_features": space.n_features,
        "n_classes": len(class_index),
        "features_by_kind": dict(Counter(f.kind for f in space.features)),
        "settings": {
            "dev_sample": dev_sample,
            "seed": seed,
            "age_bins": age_bins,
            "age_scale": age_scale,
            "scale_max": scale_max,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def run_from_config(cfg: Config) -> dict[str, Any]:
    enc = cfg.data.encoding
    return run_preprocess(
        raw_dir_from_config(cfg.data.paths.raw),
        processed_dir_from_config(cfg.data.paths.processed),
        age_bins=enc.age_bins,
        age_scale=enc.age_scale,
        scale_max=enc.scale_max,
        dev_sample=cfg.data.dev_sample,
        seed=cfg.seed,
    )


def main() -> int:
    manifest = run_from_config(load_config())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

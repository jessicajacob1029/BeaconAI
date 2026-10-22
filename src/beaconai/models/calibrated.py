"""One prediction interface for everything downstream: CalibratedClassifier.predict_differential.

It returns the top-k conditions with calibrated probabilities, or an Abstain result with a reason
when the most likely condition is below the confidence threshold. BeaconAI is decision support,
not diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from beaconai.config import Config
from beaconai.data.download import raw_dir_from_config
from beaconai.data.encode import FeatureSpace
from beaconai.data.preprocess import CONDITIONS_FILE, EVIDENCES_FILE
from beaconai.data.schema import load_catalog
from beaconai.eval.metrics import top_k_indices
from beaconai.models.base import Matrix, ModelWrapper
from beaconai.models.calibrators import Calibrator, calibrator_from_dict

META_FILE = "calibrated.json"
BASE_DIR = "base"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PatientEvidence(_Frozen):
    """What we know about a patient: DDXPlus evidence tokens plus age and sex."""

    evidence: list[str]  # e.g. "E_91", "E_55_@_V_14", "E_56_@_7"
    age: int = Field(ge=0, le=130)
    sex: Literal["M", "F"]


class DifferentialItem(_Frozen):
    condition: str
    icd10: str = ""
    severity: int | None = None
    probability: float = Field(ge=0.0, le=1.0)


class Differential(_Frozen):
    kind: Literal["differential"] = "differential"
    items: list[DifferentialItem]


class Abstain(_Frozen):
    kind: Literal["abstain"] = "abstain"
    reason: str
    top_probability: float = Field(ge=0.0, le=1.0)


Prediction = Differential | Abstain


def to_prediction(
    proba_row: np.ndarray,
    labels: list[dict[str, Any]],
    k: int,
    threshold: float,
    tie_break: np.ndarray | None = None,
) -> Prediction:
    """Top-k differential, or Abstain when the top-1 probability is below the threshold.

    Calibration can make probabilities exactly equal; `tie_break` (the model's uncalibrated
    scores) then decides the order, instead of the condition's position in labels.json.
    """
    row = np.asarray(proba_row, dtype=np.float64)
    tb = None if tie_break is None else np.asarray(tie_break, dtype=np.float64)[None, :]
    top = top_k_indices(row[None, :], k, tie_break=tb)[0]
    best = float(np.clip(row[top[0]], 0.0, 1.0))
    if best < threshold:
        return Abstain(
            reason=(
                "Not confident enough to rank conditions: the most likely one has probability "
                f"{best:.3f}, below the threshold of {threshold:.3f}. More information is needed."
            ),
            top_probability=best,
        )
    return Differential(
        items=[
            DifferentialItem(
                condition=labels[i]["name"],
                icd10=labels[i].get("icd10") or "",
                severity=labels[i].get("severity"),
                probability=float(np.clip(row[i], 0.0, 1.0)),
            )
            for i in top
        ]
    )


class CalibratedClassifier:
    def __init__(
        self,
        model: ModelWrapper,
        calibrator: Calibrator,
        labels: list[dict[str, Any]],
        feature_names: list[str],
        threshold: float,
        k: int,
        encoder: FeatureSpace | None = None,
    ) -> None:
        if model.n_features is None:
            raise ValueError("base model is not fitted")
        if len(labels) != model.n_classes:
            raise ValueError(f"{len(labels)} labels for a {model.n_classes}-class model")
        if len(feature_names) != model.n_features:
            raise ValueError(f"{len(feature_names)} feature names for {model.n_features} columns")
        if not 1 <= k <= model.n_classes:
            raise ValueError(f"k must be in [1, {model.n_classes}], got {k}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if encoder is not None and encoder.names != list(feature_names):
            raise ValueError("encoder columns do not match the model's feature columns")
        self.model = model
        self.calibrator = calibrator
        self.labels = list(labels)
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.k = int(k)
        self.encoder = encoder

    def predict_proba(self, X: Matrix) -> np.ndarray:
        """Calibrated probabilities, shape (n_patients, n_classes)."""
        return self.calibrator.transform(self.model.predict_proba(X))

    def predictions(self, X: Matrix) -> list[Prediction]:
        raw = self.model.predict_proba(X)
        calibrated = self.calibrator.transform(raw)
        return [
            to_prediction(q, self.labels, self.k, self.threshold, tie_break=r)
            for q, r in zip(calibrated, raw, strict=True)
        ]

    def encode(self, evidence: PatientEvidence) -> sparse.csr_matrix:
        if self.encoder is None:
            raise RuntimeError("no encoder attached; use with_encoder(build_feature_space(cfg))")
        try:
            values = self.encoder.encode(evidence.evidence, evidence.age, evidence.sex)
        except KeyError as err:
            # The catalog's KeyError already reads "Unknown evidence code: 'E_...'".
            raise ValueError(err.args[0]) from err
        cols = sorted(values)
        return sparse.csr_matrix(
            (
                np.array([values[c] for c in cols], dtype=np.float32),
                (np.zeros(len(cols), dtype=np.int64), np.array(cols, dtype=np.int64)),
            ),
            shape=(1, len(self.feature_names)),
        )

    def predict_differential(self, evidence: PatientEvidence) -> Prediction:
        """Top-k conditions with calibrated probabilities, or Abstain with a reason."""
        return self.predictions(self.encode(evidence))[0]

    def with_encoder(self, encoder: FeatureSpace) -> CalibratedClassifier:
        return CalibratedClassifier(
            self.model,
            self.calibrator,
            self.labels,
            self.feature_names,
            self.threshold,
            self.k,
            encoder,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / BASE_DIR)
        meta = {
            "calibrator": self.calibrator.to_dict(),
            "labels": self.labels,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "k": self.k,
        }
        (path / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, encoder: FeatureSpace | None = None) -> CalibratedClassifier:
        path = Path(path)
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        return cls(
            ModelWrapper.load(path / BASE_DIR),
            calibrator_from_dict(meta["calibrator"]),
            meta["labels"],
            meta["feature_names"],
            meta["threshold"],
            meta["k"],
            encoder,
        )


def build_feature_space(cfg: Config) -> FeatureSpace:
    """The Phase 1 encoder (evidence tokens, age, sex -> feature columns) from config."""
    raw = raw_dir_from_config(cfg.data.paths.raw)
    enc = cfg.data.encoding
    catalog = load_catalog(raw / EVIDENCES_FILE, raw / CONDITIONS_FILE)
    return FeatureSpace(catalog, enc.age_bins, age_scale=enc.age_scale, scale_max=enc.scale_max)

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from beaconai.data.encode import FeatureSpace
from beaconai.data.schema import Catalog, parse_conditions, parse_evidences
from beaconai.models.calibrated import (
    Abstain,
    CalibratedClassifier,
    Differential,
    PatientEvidence,
    to_prediction,
)
from beaconai.models.calibrators import IdentityCalibrator, TemperatureCalibrator
from beaconai.models.registry import build_model
from tests.data.conftest import AGE_BINS, RAW_CONDITIONS, RAW_EVIDENCES

LABELS = [
    {"name": "Influenza", "icd10": "J10", "severity": 3},
    {"name": "Unstable angina", "icd10": "I20.0", "severity": 1},
    {"name": "Croup", "icd10": "J05.0", "severity": 2},
]


def _space() -> FeatureSpace:
    catalog = Catalog(
        evidences=parse_evidences(RAW_EVIDENCES), conditions=parse_conditions(RAW_CONDITIONS)
    )
    return FeatureSpace(catalog, AGE_BINS, age_scale=110, scale_max=10)


def _classifier(threshold: float = 0.0, k: int = 2) -> CalibratedClassifier:
    """Logistic regression on the Phase 1 toy catalog: fever + pain vs smoker + chest pain."""
    space = _space()
    flu, angina = ["E_1", "E_10_@_7"], ["E_2", "E_30_@_V_6"]
    tokens = [flu] * 30 + [angina] * 30
    X = space.encode_many(tokens, [30] * 60, ["F", "M"] * 30)
    y = np.array([0] * 30 + [1] * 30)
    model = build_model("logreg", {"max_iter": 200}, 2, 0).fit(X, y)
    return CalibratedClassifier(
        model, IdentityCalibrator(), LABELS[:2], space.names, threshold, k, encoder=space
    )


def test_to_prediction_returns_top_k_in_order() -> None:
    pred = to_prediction(np.array([0.2, 0.5, 0.3]), LABELS, k=2, threshold=0.4)
    assert isinstance(pred, Differential)
    assert [i.condition for i in pred.items] == ["Unstable angina", "Croup"]
    assert pred.items[0].probability == pytest.approx(0.5)
    assert pred.items[0].icd10 == "I20.0" and pred.items[0].severity == 1


def test_to_prediction_abstains_below_threshold_with_reason() -> None:
    pred = to_prediction(np.array([0.4, 0.35, 0.25]), LABELS, k=3, threshold=0.5)
    assert isinstance(pred, Abstain)
    assert pred.top_probability == pytest.approx(0.4)
    assert "0.40" in pred.reason and "0.50" in pred.reason


def test_probability_equal_to_threshold_is_answered() -> None:
    assert isinstance(to_prediction(np.array([0.5, 0.3, 0.2]), LABELS, 3, 0.5), Differential)


def test_predict_differential_from_evidence_codes() -> None:
    pred = _classifier().predict_differential(
        PatientEvidence(evidence=["E_1", "E_10_@_6"], age=34, sex="F")
    )
    assert isinstance(pred, Differential)
    assert pred.items[0].condition == "Influenza"
    assert sum(i.probability for i in pred.items) == pytest.approx(1.0)


def test_unknown_evidence_code_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="E_999"):
        _classifier().predict_differential(PatientEvidence(evidence=["E_999"], age=30, sex="M"))


def test_patient_evidence_validation() -> None:
    with pytest.raises(ValidationError):
        PatientEvidence(evidence=[], age=-1, sex="M")
    with pytest.raises(ValidationError):
        PatientEvidence(evidence=[], age=30, sex="X")


def test_encoder_column_mismatch_rejected() -> None:
    clf = _classifier()
    with pytest.raises(ValueError, match="columns"):
        CalibratedClassifier(
            clf.model, clf.calibrator, clf.labels, clf.feature_names[::-1], 0.5, 2, _space()
        )


def test_no_encoder_raises() -> None:
    clf = _classifier()
    bare = CalibratedClassifier(clf.model, clf.calibrator, clf.labels, clf.feature_names, 0.0, 2)
    with pytest.raises(RuntimeError, match="encoder"):
        bare.predict_differential(PatientEvidence(evidence=["E_1"], age=30, sex="M"))


def test_bad_k_label_count_or_threshold_rejected() -> None:
    clf = _classifier()
    with pytest.raises(ValueError):
        CalibratedClassifier(clf.model, clf.calibrator, clf.labels, clf.feature_names, 0.5, 3)
    with pytest.raises(ValueError):
        CalibratedClassifier(clf.model, clf.calibrator, LABELS, clf.feature_names, 0.5, 2)
    with pytest.raises(ValueError):
        CalibratedClassifier(clf.model, clf.calibrator, clf.labels, clf.feature_names, 1.5, 2)


def test_save_load_round_trip(tmp_path: Path) -> None:
    clf = _classifier(threshold=0.3)
    clf.calibrator = TemperatureCalibrator((0.05, 20.0), temperature=2.0)
    clf.save(tmp_path / "c")
    loaded = CalibratedClassifier.load(tmp_path / "c", encoder=_space())
    evidence = PatientEvidence(evidence=["E_2"], age=60, sex="M")
    assert loaded.predict_differential(evidence) == clf.predict_differential(evidence)
    assert loaded.threshold == 0.3 and loaded.k == 2

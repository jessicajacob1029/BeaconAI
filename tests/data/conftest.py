"""Tiny hand-made DDXPlus-shaped catalog and patients. No real data is used in tests.

Feature layout produced by `space` (age_bins [0, 18, 65]):
    0 E_1            binary   fever
    1 E_2            binary   antecedent: smoker
    2 E_10           scale    pain intensity, value / 10
    3-5 E_20_@_V_1..3  categorical (one-hot) rash color
    6-8 E_30_@_V_5..7  multi-choice (multi-hot) pain location
    9-11 AGE_BIN_0/18/65
    12 AGE_SCALED
    13 SEX_M
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from beaconai.data.encode import FeatureSpace
from beaconai.data.schema import Catalog, parse_conditions, parse_evidences

AGE_BINS = [0, 18, 65]


def _ev(name: str, question: str, data_type: str, **extra: Any) -> dict[str, Any]:
    body = {
        "name": name,
        "code_question": name,
        "question_fr": "fr",
        "question_en": question,
        "is_antecedent": False,
        "default_value": 0,
        "value_meaning": {},
        "possible-values": [],
        "data_type": data_type,
    }
    body.update(extra)
    return body


RAW_EVIDENCES: dict[str, Any] = {
    # Deliberately out of numeric order to check that columns are sorted E_1, E_2, E_10, ...
    "E_10": _ev("E_10", "How intense is the pain?", "C", **{"possible-values": list(range(11))}),
    "E_1": _ev("E_1", "Do you have a fever?", "B"),
    "E_30": _ev(
        "E_30",
        "Do you feel pain somewhere?",
        "M",
        default_value="V_7",
        value_meaning={
            "V_5": {"fr": "tete", "en": "head"},
            "V_6": {"fr": "poitrine", "en": "chest"},
            "V_7": {"fr": "nulle part", "en": "nowhere"},
        },
        **{"possible-values": ["V_5", "V_6", "V_7"]},
    ),
    "E_2": _ev("E_2", "Do you smoke?", "B", is_antecedent=True),
    "E_20": _ev(
        "E_20",
        "What color is the rash?",
        "C",
        default_value="V_1",
        value_meaning={
            "V_1": {"fr": "NA", "en": "NA"},
            "V_2": {"fr": "rose", "en": "pink"},
            "V_3": {"fr": "rouge", "en": "red"},
        },
        **{"possible-values": ["V_1", "V_2", "V_3"]},
    ),
}

RAW_CONDITIONS: dict[str, Any] = {
    # Keyed by French name, like the real file.
    "Grippe": {
        "condition_name": "Grippe",
        "cond-name-fr": "Grippe",
        "cond-name-eng": "Influenza",
        "icd10-id": "J10",
        "symptoms": {},
        "antecedents": {},
        "severity": 3,
    },
    "Angine instable": {
        "condition_name": "Angine instable",
        "cond-name-fr": "Angine instable",
        "cond-name-eng": "Unstable angina",
        "icd10-id": "I20.0",
        "symptoms": {},
        "antecedents": {},
        "severity": 1,
    },
}

# AGE, DIFFERENTIAL_DIAGNOSIS, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE
PATIENT_ROWS = [
    (
        "34",
        "[['Influenza', 0.7], ['Unstable angina', 0.3]]",
        "F",
        "Influenza",
        "['E_1', 'E_10_@_6', 'E_20_@_V_2', 'E_30_@_V_5', 'E_30_@_V_6']",
        "E_1",
    ),
    ("70", "[['Unstable angina', 1.0]]", "M", "Unstable angina", "['E_2', 'E_30_@_V_6']", "E_2"),
    ("5", "[['Influenza', 0.9], ['Unstable angina', 0.1]]", "M", "Influenza", "['E_1']", "E_1"),
]


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(
        evidences=parse_evidences(RAW_EVIDENCES), conditions=parse_conditions(RAW_CONDITIONS)
    )


@pytest.fixture
def space(catalog: Catalog) -> FeatureSpace:
    return FeatureSpace(catalog, AGE_BINS, age_scale=110, scale_max=10)


def write_patients_csv(path: Path, rows: list[tuple[str, ...]] = PATIENT_ROWS) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["AGE", "DIFFERENTIAL_DIAGNOSIS", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"]
        )
        w.writerows(rows)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A raw folder laid out like data/raw/ddxplus after download."""
    d = tmp_path / "raw"
    d.mkdir()
    (d / "release_evidences.json").write_text(json.dumps(RAW_EVIDENCES), encoding="utf-8")
    (d / "release_conditions.json").write_text(json.dumps(RAW_CONDITIONS), encoding="utf-8")
    for split in ("train", "validate", "test"):
        write_patients_csv(d / f"release_{split}_patients.csv")
    return d

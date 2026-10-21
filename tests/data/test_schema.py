from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from beaconai.data.schema import (
    Catalog,
    EvidenceType,
    load_catalog,
    parse_conditions,
    parse_evidences,
)
from tests.data.conftest import RAW_CONDITIONS, RAW_EVIDENCES


def test_evidence_types_and_scale_detection(catalog: Catalog) -> None:
    assert catalog.evidence("E_1").data_type is EvidenceType.BINARY
    assert catalog.evidence("E_10").is_scale
    assert not catalog.evidence("E_20").is_scale  # categorical with V_ codes
    assert not catalog.evidence("E_30").is_scale  # multi-choice
    assert catalog.evidence("E_2").is_antecedent


def test_lookups(catalog: Catalog) -> None:
    assert catalog.question("E_1") == "Do you have a fever?"
    assert catalog.value_label("E_20", "V_3") == "red"
    assert catalog.value_label("E_10", "7") == "7"  # scale values have no meaning text
    with pytest.raises(KeyError, match="E_404"):
        catalog.question("E_404")


def test_conditions_keyed_by_english_name(catalog: Catalog) -> None:
    assert catalog.condition_names == ["Influenza", "Unstable angina"]
    assert catalog.severity("Unstable angina") == 1
    assert catalog.conditions["Influenza"].name_fr == "Grippe"
    assert catalog.conditions["Influenza"].icd10 == "J10"
    with pytest.raises(KeyError):
        catalog.severity("Grippe")


def test_binary_with_possible_values_rejected() -> None:
    raw = copy.deepcopy(RAW_EVIDENCES)
    raw["E_1"]["possible-values"] = ["V_1"]
    with pytest.raises(ValidationError):
        parse_evidences(raw)


def test_categorical_without_values_rejected() -> None:
    raw = copy.deepcopy(RAW_EVIDENCES)
    raw["E_20"]["possible-values"] = []
    with pytest.raises(ValidationError):
        parse_evidences(raw)


def test_unknown_data_type_rejected() -> None:
    raw = copy.deepcopy(RAW_EVIDENCES)
    raw["E_1"]["data_type"] = "X"
    with pytest.raises(ValidationError):
        parse_evidences(raw)


def test_key_name_mismatch_rejected() -> None:
    raw = copy.deepcopy(RAW_EVIDENCES)
    raw["E_99"] = raw.pop("E_1")
    with pytest.raises(ValueError, match="does not match"):
        parse_evidences(raw)


def test_duplicate_english_condition_rejected() -> None:
    raw = copy.deepcopy(RAW_CONDITIONS)
    raw["Copie"] = dict(raw["Grippe"], **{"cond-name-fr": "Copie"})
    with pytest.raises(ValueError, match="Duplicate"):
        parse_conditions(raw)


def test_severity_out_of_range_rejected() -> None:
    raw = copy.deepcopy(RAW_CONDITIONS)
    raw["Grippe"]["severity"] = 9
    with pytest.raises(ValidationError):
        parse_conditions(raw)


def test_load_catalog_from_files(tmp_path: Path) -> None:
    ev, cond = tmp_path / "ev.json", tmp_path / "cond.json"
    ev.write_text(json.dumps(RAW_EVIDENCES), encoding="utf-8")
    cond.write_text(json.dumps(RAW_CONDITIONS), encoding="utf-8")
    catalog = load_catalog(ev, cond)
    assert len(catalog.evidences) == 5
    assert len(catalog.conditions) == 2

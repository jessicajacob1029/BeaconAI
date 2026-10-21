"""Typed views of the DDXPlus catalog files.

`release_evidences.json` defines every symptom/antecedent ("evidence") a patient can have.
`release_conditions.json` defines the 49 conditions. This module validates both files into
Pydantic models and provides the lookups the rest of the project needs: evidence code to
English question, value code to English value, and condition to severity.

Notes on the source format:
- Evidence data types are "B" (binary), "C" (categorical, one value) and "M" (multi-choice).
- Some categorical evidences are 0-10 scales: their possible values are integers and they
  have no value meanings. We call these "scale" evidences and treat them as numeric.
- The conditions file is keyed by the French name. Patient records use the English name
  (`cond-name-eng`), so the catalog is keyed by English name.
- Severity runs from 1 (most severe) to 5 (least severe).
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceType(StrEnum):
    BINARY = "B"
    CATEGORICAL = "C"
    MULTI = "M"


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    name: str
    question_en: str
    data_type: EvidenceType
    is_antecedent: bool = False
    default_value: int | str | None = None
    possible_values: list[int | str] = Field(default_factory=list, alias="possible-values")
    value_meaning: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_values(self) -> Evidence:
        if self.data_type is EvidenceType.BINARY:
            if self.possible_values:
                raise ValueError(f"{self.name}: binary evidence cannot have possible values")
        elif not self.possible_values:
            raise ValueError(f"{self.name}: {self.data_type} evidence needs possible values")
        return self

    @property
    def is_scale(self) -> bool:
        """True for categorical evidences whose values are plain integers (0-10 scales)."""
        return self.data_type is EvidenceType.CATEGORICAL and all(
            isinstance(v, int) for v in self.possible_values
        )

    @property
    def value_codes(self) -> list[str]:
        """Possible values as the strings that appear in patient records."""
        return [str(v) for v in self.possible_values]

    def value_label(self, value: str) -> str:
        """English text for a value code, or the raw value if it has no meaning entry."""
        meaning = self.value_meaning.get(value)
        if meaning and meaning.get("en"):
            return meaning["en"].strip()
        return value


class Condition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = Field(alias="cond-name-eng")
    name_fr: str = Field(alias="cond-name-fr")
    icd10: str | None = Field(default=None, alias="icd10-id")
    severity: int = Field(ge=1, le=5)


class Catalog(BaseModel):
    """All evidences and conditions, with lookup helpers."""

    model_config = ConfigDict(frozen=True)

    evidences: dict[str, Evidence]
    conditions: dict[str, Condition]

    def evidence(self, code: str) -> Evidence:
        try:
            return self.evidences[code]
        except KeyError:
            raise KeyError(f"Unknown evidence code: {code!r}") from None

    def question(self, code: str) -> str:
        return self.evidence(code).question_en

    def value_label(self, code: str, value: str) -> str:
        return self.evidence(code).value_label(value)

    def severity(self, condition: str) -> int:
        try:
            return self.conditions[condition].severity
        except KeyError:
            raise KeyError(f"Unknown condition: {condition!r}") from None

    @property
    def condition_names(self) -> list[str]:
        """Condition names in a fixed, sorted order. Index in this list = class id."""
        return sorted(self.conditions)


def parse_evidences(raw: dict[str, Any]) -> dict[str, Evidence]:
    evidences = {code: Evidence.model_validate(body) for code, body in raw.items()}
    for code, ev in evidences.items():
        if code != ev.name:
            raise ValueError(f"Evidence key {code!r} does not match its name {ev.name!r}")
    return evidences


def parse_conditions(raw: dict[str, Any]) -> dict[str, Condition]:
    conditions: dict[str, Condition] = {}
    for body in raw.values():
        cond = Condition.model_validate(body)
        if cond.name in conditions:
            raise ValueError(f"Duplicate English condition name: {cond.name!r}")
        conditions[cond.name] = cond
    return conditions


def load_catalog(evidences_path: str | Path, conditions_path: str | Path) -> Catalog:
    with Path(evidences_path).open(encoding="utf-8") as f:
        evidences = parse_evidences(json.load(f))
    with Path(conditions_path).open(encoding="utf-8") as f:
        conditions = parse_conditions(json.load(f))
    return Catalog(evidences=evidences, conditions=conditions)

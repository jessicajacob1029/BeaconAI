"""Encode DDXPlus patients into fixed-length sparse feature vectors.

Column layout (in this order):
- Binary evidence: one 0/1 column, e.g. `E_91`.
- Scale evidence (categorical with 0-10 integer values): one numeric column, value / scale_max,
  0 when absent, e.g. `E_56`.
- Categorical evidence with value codes: one-hot, one column per value, e.g. `E_130_@_V_86`.
- Multi-choice evidence: multi-hot, one column per value, e.g. `E_55_@_V_14`.
- Age buckets (one-hot), one scaled age column (age / age_scale, clipped to 1), and `SEX_M`.

Every column has a human-readable label so later explanations (SHAP) can say
"How intense is the pain? (0-10 scale)" instead of `E_56`.
"""

from __future__ import annotations

import ast
import bisect
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse

from beaconai.data.schema import Catalog, Evidence, EvidenceType

VALUE_SEP = "_@_"


class FeatureKind:
    BINARY = "binary"
    SCALE = "scale"
    CATEGORICAL = "categorical"
    MULTI = "multi"
    AGE_BUCKET = "age_bucket"
    AGE_SCALED = "age_scaled"
    SEX = "sex"


@dataclass(frozen=True)
class Feature:
    index: int
    name: str
    kind: str
    label: str
    evidence: str | None = None
    value: str | None = None
    is_antecedent: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _code_sort_key(code: str) -> tuple[int, str]:
    """Sort E_2 before E_10."""
    prefix, _, num = code.rpartition("_")
    return (int(num), prefix) if num.isdigit() else (10**9, code)


def age_bucket_labels(bins: Sequence[int]) -> list[str]:
    labels = []
    for i, lo in enumerate(bins):
        if i + 1 == len(bins):
            labels.append(f"Age {lo}+")
        else:
            hi = bins[i + 1] - 1
            labels.append(f"Age {lo}" if hi == lo else f"Age {lo}-{hi}")
    return labels


def age_bucket_index(age: int, bins: Sequence[int]) -> int:
    if age < 0:
        raise ValueError(f"Age must be non-negative, got {age}")
    return bisect.bisect_right(bins, age) - 1


class FeatureSpace:
    """Maps evidence tokens, age and sex to column indices and values."""

    def __init__(
        self,
        catalog: Catalog,
        age_bins: Sequence[int],
        age_scale: float = 110,
        scale_max: float = 10,
    ) -> None:
        self.catalog = catalog
        self.age_bins = list(age_bins)
        self.age_scale = float(age_scale)
        self.scale_max = float(scale_max)
        self.features: list[Feature] = []
        # Exact token (e.g. "E_91" or "E_55_@_V_14") -> column, for 0/1 columns.
        self._token_col: dict[str, int] = {}
        # Scale evidence code -> its single numeric column.
        self._scale_col: dict[str, int] = {}
        self._build()

    def _add(self, **kwargs: object) -> int:
        idx = len(self.features)
        self.features.append(Feature(index=idx, **kwargs))  # type: ignore[arg-type]
        return idx

    def _build(self) -> None:
        for code in sorted(self.catalog.evidences, key=_code_sort_key):
            ev = self.catalog.evidences[code]
            question = ev.question_en.strip()
            common = {"evidence": code, "is_antecedent": ev.is_antecedent}
            if ev.data_type is EvidenceType.BINARY:
                self._token_col[code] = self._add(
                    name=code, kind=FeatureKind.BINARY, label=question, **common
                )
            elif ev.is_scale:
                self._scale_col[code] = self._add(
                    name=code,
                    kind=FeatureKind.SCALE,
                    label=f"{question} (0-{self.scale_max:g} scale)",
                    **common,
                )
            else:
                kind = (
                    FeatureKind.MULTI
                    if ev.data_type is EvidenceType.MULTI
                    else FeatureKind.CATEGORICAL
                )
                for value in ev.value_codes:
                    token = f"{code}{VALUE_SEP}{value}"
                    self._token_col[token] = self._add(
                        name=token,
                        kind=kind,
                        label=f"{question} = {ev.value_label(value)}",
                        value=value,
                        **common,
                    )

        self._age_bucket_start = len(self.features)
        for lo, label in zip(self.age_bins, age_bucket_labels(self.age_bins), strict=True):
            self._add(name=f"AGE_BIN_{lo}", kind=FeatureKind.AGE_BUCKET, label=label)
        self._age_scaled_col = self._add(
            name="AGE_SCALED",
            kind=FeatureKind.AGE_SCALED,
            label=f"Age (years / {self.age_scale:g})",
        )
        self._sex_col = self._add(name="SEX_M", kind=FeatureKind.SEX, label="Sex: male")

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def labels(self) -> list[str]:
        return [f.label for f in self.features]

    def _evidence_values(self, tokens: Iterable[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        seen_single: set[str] = set()  # categorical evidences already given a value
        for token in tokens:
            code, sep, value = token.partition(VALUE_SEP)
            ev: Evidence = self.catalog.evidence(code)  # raises KeyError on unknown code

            if ev.data_type is EvidenceType.BINARY:
                if sep:
                    raise ValueError(f"Binary evidence {code} must not have a value: {token!r}")
                out[self._token_col[code]] = 1.0
                continue

            if not sep or not value:
                raise ValueError(f"{ev.data_type} evidence {code} needs a value: {token!r}")
            if value not in ev.value_codes:
                raise ValueError(f"Value {value!r} is not allowed for {code}: {token!r}")
            if ev.data_type is EvidenceType.CATEGORICAL:
                if code in seen_single:
                    raise ValueError(f"Categorical evidence {code} has more than one value")
                seen_single.add(code)

            if ev.is_scale:
                out[self._scale_col[code]] = int(value) / self.scale_max
            else:
                out[self._token_col[token]] = 1.0
        return out

    def encode(self, tokens: Iterable[str], age: int, sex: str) -> dict[int, float]:
        """Encode one patient as {column: value}. Zero values are left out."""
        if sex not in {"M", "F"}:
            raise ValueError(f"Sex must be 'M' or 'F', got {sex!r}")
        out = self._evidence_values(tokens)
        out[self._age_bucket_start + age_bucket_index(age, self.age_bins)] = 1.0
        out[self._age_scaled_col] = min(age / self.age_scale, 1.0)
        if sex == "M":
            out[self._sex_col] = 1.0
        return {col: val for col, val in out.items() if val != 0.0}

    def encode_many(
        self,
        evidence_lists: Iterable[Sequence[str]],
        ages: Iterable[int],
        sexes: Iterable[str],
    ) -> sparse.csr_matrix:
        """Encode many patients into a float32 CSR matrix (rows = patients)."""
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for tokens, age, sex in zip(evidence_lists, ages, sexes, strict=True):
            row = self.encode(tokens, int(age), str(sex))
            cols = sorted(row)
            indices.extend(cols)
            data.extend(row[c] for c in cols)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(indptr) - 1, self.n_features),
        )


def parse_evidence_list(raw: str) -> list[str]:
    """Parse a patient's EVIDENCES cell, stored as a Python list literal of strings."""
    value = ast.literal_eval(raw)
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise ValueError(f"EVIDENCES must be a list of strings, got {raw[:80]!r}")
    return value


def parse_differential(raw: str) -> list[tuple[str, float]]:
    """Parse a DIFFERENTIAL_DIAGNOSIS cell: a list literal of [condition, probability] pairs."""
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError(f"DIFFERENTIAL_DIAGNOSIS must be a list, got {raw[:80]!r}")
    return [(str(name), float(prob)) for name, prob in value]


def encode_differentials(
    differentials: Iterable[Sequence[tuple[str, float]]], class_index: dict[str, int]
) -> sparse.csr_matrix:
    """Turn ground-truth differentials into a float32 CSR matrix (patients x classes)."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n = 0
    for n, ddx in enumerate(differentials, start=1):
        for name, prob in ddx:
            if name not in class_index:
                raise KeyError(f"Unknown condition in differential: {name!r}")
            rows.append(n - 1)
            cols.append(class_index[name])
            vals.append(prob)
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(n, len(class_index)),
    )

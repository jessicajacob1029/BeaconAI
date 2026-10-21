"""Encoding tests, one section per evidence type, on the tiny catalog in conftest.py."""

from __future__ import annotations

import numpy as np
import pytest

from beaconai.data.encode import (
    FeatureKind,
    FeatureSpace,
    age_bucket_index,
    age_bucket_labels,
    encode_differentials,
    parse_differential,
    parse_evidence_list,
)

# Column indices from the layout documented in conftest.py.
FEVER, SMOKER, PAIN_SCALE = 0, 1, 2
RASH = {"V_1": 3, "V_2": 4, "V_3": 5}
LOC = {"V_5": 6, "V_6": 7, "V_7": 8}
AGE_BIN = {0: 9, 18: 10, 65: 11}
AGE_SCALED, SEX_M = 12, 13


def evidence_cols(space: FeatureSpace, tokens: list[str]) -> dict[int, float]:
    """Encode with a fixed age/sex and drop the age/sex columns."""
    row = space.encode(tokens, age=30, sex="F")
    return {c: v for c, v in row.items() if c < AGE_BIN[0]}


# ---------- layout and labels ----------


def test_layout_and_order(space: FeatureSpace) -> None:
    assert space.n_features == 14
    assert space.names[:3] == ["E_1", "E_2", "E_10"]  # numeric order, not string order
    assert space.names[RASH["V_2"]] == "E_20_@_V_2"
    assert space.names[LOC["V_6"]] == "E_30_@_V_6"
    assert space.names[-3:] == ["AGE_BIN_65", "AGE_SCALED", "SEX_M"]


def test_every_feature_has_readable_unique_label(space: FeatureSpace) -> None:
    labels = space.labels
    assert all(label.strip() for label in labels)
    assert len(set(labels)) == len(labels)
    for f in space.features:
        if f.evidence is not None:
            assert f.label != f.name  # never just the raw code
    assert space.features[FEVER].label == "Do you have a fever?"
    assert space.features[RASH["V_2"]].label == "What color is the rash? = pink"
    assert space.features[LOC["V_5"]].label == "Do you feel pain somewhere? = head"
    assert space.features[PAIN_SCALE].label == "How intense is the pain? (0-10 scale)"


def test_feature_kinds_and_metadata(space: FeatureSpace) -> None:
    f = space.features
    assert f[FEVER].kind == FeatureKind.BINARY
    assert f[SMOKER].is_antecedent is True
    assert f[PAIN_SCALE].kind == FeatureKind.SCALE
    assert f[RASH["V_3"]].kind == FeatureKind.CATEGORICAL
    assert f[RASH["V_3"]].evidence == "E_20" and f[RASH["V_3"]].value == "V_3"
    assert f[LOC["V_7"]].kind == FeatureKind.MULTI


def test_unknown_evidence_code_raises(space: FeatureSpace) -> None:
    with pytest.raises(KeyError, match="E_999"):
        space.encode(["E_999"], age=30, sex="F")


# ---------- binary ----------


def test_binary_present_sets_one(space: FeatureSpace) -> None:
    assert evidence_cols(space, ["E_1"]) == {FEVER: 1.0}


def test_binary_absent_is_zero(space: FeatureSpace) -> None:
    assert evidence_cols(space, []) == {}


def test_binary_with_value_raises(space: FeatureSpace) -> None:
    with pytest.raises(ValueError, match="must not have a value"):
        space.encode(["E_1_@_V_2"], age=30, sex="F")


# ---------- numeric scale (categorical 0-10) ----------


def test_scale_has_single_column(space: FeatureSpace) -> None:
    assert [f.name for f in space.features if f.evidence == "E_10"] == ["E_10"]


@pytest.mark.parametrize(("value", "expected"), [("6", 0.6), ("10", 1.0), ("1", 0.1)])
def test_scale_value_is_divided_by_max(space: FeatureSpace, value: str, expected: float) -> None:
    cols = evidence_cols(space, [f"E_10_@_{value}"])
    assert cols.keys() == {PAIN_SCALE}
    assert cols[PAIN_SCALE] == pytest.approx(expected)


def test_scale_absent_and_zero_both_encode_as_zero(space: FeatureSpace) -> None:
    assert evidence_cols(space, []) == {}
    assert evidence_cols(space, ["E_10_@_0"]) == {}


def test_scale_respects_configured_max(catalog) -> None:
    space = FeatureSpace(catalog, [0, 18, 65], scale_max=5)
    assert space.encode(["E_10_@_5"], age=30, sex="F")[PAIN_SCALE] == pytest.approx(1.0)


@pytest.mark.parametrize("token", ["E_10_@_11", "E_10_@_-1", "E_10_@_V_2", "E_10"])
def test_scale_invalid_value_raises(space: FeatureSpace, token: str) -> None:
    with pytest.raises(ValueError):
        space.encode([token], age=30, sex="F")


def test_scale_given_twice_raises(space: FeatureSpace) -> None:
    with pytest.raises(ValueError, match="more than one value"):
        space.encode(["E_10_@_3", "E_10_@_4"], age=30, sex="F")


def test_scale_stored_as_float32_in_matrix(space: FeatureSpace) -> None:
    X = space.encode_many([["E_10_@_6"]], [30], ["F"])
    assert X.dtype == np.float32
    assert X[0, PAIN_SCALE] == pytest.approx(0.6)


# ---------- categorical with value codes (one-hot) ----------


def test_categorical_one_hot(space: FeatureSpace) -> None:
    assert evidence_cols(space, ["E_20_@_V_2"]) == {RASH["V_2"]: 1.0}


def test_categorical_default_value_gets_its_own_column(space: FeatureSpace) -> None:
    # Answering the default ("NA") is different from not being asked at all.
    assert evidence_cols(space, ["E_20_@_V_1"]) == {RASH["V_1"]: 1.0}


def test_categorical_two_values_raises(space: FeatureSpace) -> None:
    with pytest.raises(ValueError, match="more than one value"):
        space.encode(["E_20_@_V_2", "E_20_@_V_3"], age=30, sex="F")


@pytest.mark.parametrize("token", ["E_20", "E_20_@_", "E_20_@_V_99"])
def test_categorical_missing_or_unknown_value_raises(space: FeatureSpace, token: str) -> None:
    with pytest.raises(ValueError):
        space.encode([token], age=30, sex="F")


# ---------- multi-choice (multi-hot) ----------


def test_multi_choice_multi_hot(space: FeatureSpace) -> None:
    cols = evidence_cols(space, ["E_30_@_V_5", "E_30_@_V_6"])
    assert cols == {LOC["V_5"]: 1.0, LOC["V_6"]: 1.0}


def test_multi_choice_duplicate_value_is_idempotent(space: FeatureSpace) -> None:
    assert evidence_cols(space, ["E_30_@_V_6", "E_30_@_V_6"]) == {LOC["V_6"]: 1.0}


@pytest.mark.parametrize("token", ["E_30", "E_30_@_V_1"])
def test_multi_choice_missing_or_unknown_value_raises(space: FeatureSpace, token: str) -> None:
    with pytest.raises(ValueError):
        space.encode([token], age=30, sex="F")


# ---------- age and sex ----------


def test_age_bucket_labels() -> None:
    assert age_bucket_labels([0, 1, 5, 75]) == ["Age 0", "Age 1-4", "Age 5-74", "Age 75+"]


@pytest.mark.parametrize(("age", "bucket"), [(0, 0), (17, 0), (18, 18), (64, 18), (65, 65)])
def test_age_bucket_boundaries(space: FeatureSpace, age: int, bucket: int) -> None:
    row = space.encode([], age=age, sex="F")
    assert [c for c in AGE_BIN.values() if c in row] == [AGE_BIN[bucket]]


def test_age_bucket_index_rejects_negative() -> None:
    with pytest.raises(ValueError):
        age_bucket_index(-1, [0, 18])


@pytest.mark.parametrize(("age", "scaled"), [(55, 0.5), (110, 1.0), (120, 1.0)])
def test_age_scaled_column(space: FeatureSpace, age: int, scaled: float) -> None:
    assert space.encode([], age=age, sex="F")[AGE_SCALED] == pytest.approx(scaled)


def test_age_zero_has_no_scaled_entry(space: FeatureSpace) -> None:
    # 0 / 110 == 0, which is not stored in the sparse row.
    assert AGE_SCALED not in space.encode([], age=0, sex="F")


def test_sex_encoding(space: FeatureSpace) -> None:
    assert space.encode([], age=30, sex="M")[SEX_M] == 1.0
    assert SEX_M not in space.encode([], age=30, sex="F")
    with pytest.raises(ValueError):
        space.encode([], age=30, sex="X")


# ---------- whole matrix ----------


def test_encode_many_builds_expected_matrix(space: FeatureSpace) -> None:
    X = space.encode_many(
        [["E_1", "E_10_@_6", "E_20_@_V_2", "E_30_@_V_5", "E_30_@_V_6"], ["E_2"]],
        [34, 70],
        ["F", "M"],
    )
    assert X.shape == (2, space.n_features)
    row0 = X[0].toarray().ravel()
    expected0 = np.zeros(space.n_features, dtype=np.float32)
    expected0[[FEVER, RASH["V_2"], LOC["V_5"], LOC["V_6"], AGE_BIN[18]]] = 1.0
    expected0[PAIN_SCALE] = 0.6
    expected0[AGE_SCALED] = 34 / 110
    np.testing.assert_allclose(row0, expected0, rtol=1e-6)
    row1 = X[1].toarray().ravel()
    assert set(np.flatnonzero(row1)) == {SMOKER, AGE_BIN[65], AGE_SCALED, SEX_M}


def test_encode_many_length_mismatch_raises(space: FeatureSpace) -> None:
    with pytest.raises(ValueError):
        space.encode_many([["E_1"]], [30, 40], ["F"])


# ---------- parsing raw cells ----------


def test_parse_evidence_list() -> None:
    assert parse_evidence_list("['E_1', 'E_30_@_V_5']") == ["E_1", "E_30_@_V_5"]
    assert parse_evidence_list("[]") == []


@pytest.mark.parametrize("raw", ["'E_1'", "[1, 2]", "{'a': 1}"])
def test_parse_evidence_list_rejects_non_string_lists(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_evidence_list(raw)


def test_parse_evidence_list_does_not_execute_code() -> None:
    with pytest.raises(ValueError):
        parse_evidence_list("__import__('os').getcwd()")


def test_differentials_matrix() -> None:
    ddx = [parse_differential("[['Influenza', 0.7], ['Unstable angina', 0.3]]"), []]
    M = encode_differentials(ddx, {"Influenza": 0, "Unstable angina": 1})
    assert M.shape == (2, 2)
    np.testing.assert_allclose(M.toarray(), [[0.7, 0.3], [0.0, 0.0]], rtol=1e-6)


def test_differentials_unknown_condition_raises() -> None:
    with pytest.raises(KeyError):
        encode_differentials([[("Nope", 1.0)]], {"Influenza": 0})

import pytest

from beaconai.explain.labels import FeatureLabeler

FEATURES = [
    {"index": 0, "name": "E_91", "kind": "binary", "label": "Do you have a fever?"},
    {"index": 1, "name": "E_56", "kind": "scale", "label": "How intense is the pain? (0-10 scale)"},
    {"index": 2, "name": "E_55_@_V_14", "kind": "multi", "label": "Where is the pain? = chest"},
    {"index": 3, "name": "E_4_@_V_2", "kind": "categorical", "label": "Rash color = pink"},
    {"index": 4, "name": "AGE_BIN_30", "kind": "age_bucket", "label": "Age 30-44"},
    {"index": 5, "name": "AGE_SCALED", "kind": "age_scaled", "label": "Age (scaled)"},
    {"index": 6, "name": "SEX_M", "kind": "sex", "label": "Sex: male"},
]
L = FeatureLabeler(FEATURES, scale_max=10, age_scale=110)


@pytest.mark.parametrize(
    ("j", "value", "expected"),
    [
        (0, 1.0, "Do you have a fever?: yes"),
        (1, 0.6, "How intense is the pain? (0-10 scale): 6/10"),
        (2, 1.0, "Where is the pain? = chest"),
        (3, 1.0, "Rash color = pink"),
        (4, 1.0, "Age 30-44"),
        (5, 52 / 110, "Age: 52"),
        (6, 1.0, "Sex: male"),
    ],
)
def test_describe(j: int, value: float, expected: str) -> None:
    assert L.describe(j, value) == expected


def test_kind_name_label() -> None:
    assert (L.kind(6), L.name(6), L.label(0)) == ("sex", "SEX_M", "Do you have a fever?")


def test_unknown_kind_falls_back_to_label() -> None:
    odd = FeatureLabeler([{"index": 0, "name": "X", "kind": "new", "label": "Thing"}], 10, 110)
    assert odd.describe(0, 1.0) == "Thing"

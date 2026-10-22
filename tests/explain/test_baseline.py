import json
from pathlib import Path

import numpy as np
import pytest

from beaconai.data.encode import FeatureSpace
from beaconai.data.schema import Catalog, parse_conditions, parse_evidences
from beaconai.explain.baseline import (
    baseline_vector,
    demographic_means,
    load_baseline,
    save_baseline,
)
from tests.data.conftest import AGE_BINS, RAW_CONDITIONS, RAW_EVIDENCES


def _space() -> FeatureSpace:
    catalog = Catalog(
        evidences=parse_evidences(RAW_EVIDENCES), conditions=parse_conditions(RAW_CONDITIONS)
    )
    return FeatureSpace(catalog, AGE_BINS, age_scale=110, scale_max=10)


def test_means_match_the_phase1_encoder() -> None:
    space = _space()
    rng = np.random.default_rng(0)
    ages = rng.integers(0, 120, size=500)  # includes ages past age_scale (clipped to 1)
    sexes = rng.choice(["F", "M"], size=500)
    X = space.encode_many([[] for _ in ages], ages, sexes).toarray()
    features = [f.to_dict() for f in space.features]
    means = demographic_means(ages, sexes, features, AGE_BINS, 110)
    for f in features:
        if f["kind"] in {"age_bucket", "age_scaled", "sex"}:
            assert means[f["name"]] == pytest.approx(X[:, f["index"]].mean())
    assert len(means) == len(AGE_BINS) + 2


def test_baseline_vector_zero_for_evidence() -> None:
    features = [f.to_dict() for f in _space().features]
    means = demographic_means([30, 70], ["F", "M"], features, AGE_BINS, 110)
    m = baseline_vector(features, means)
    for f in features:
        expected = means.get(f["name"], 0.0)
        assert m[f["index"]] == pytest.approx(expected)
    assert all(
        m[f["index"]] == 0 for f in features if f["kind"] not in {"age_bucket", "age_scaled", "sex"}
    )


def test_baseline_vector_requires_every_demographic_column() -> None:
    features = [f.to_dict() for f in _space().features]
    with pytest.raises(ValueError, match="missing"):
        baseline_vector(features, {"SEX_M": 0.5})


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    save_baseline(path, {"SEX_M": 0.48}, "raw train CSV (full training set)", 1025602)
    means, meta = load_baseline(path)
    assert means == {"SEX_M": 0.48}
    assert meta["source"].startswith("raw train CSV") and meta["n_patients"] == 1025602
    json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict JSON {c}"))

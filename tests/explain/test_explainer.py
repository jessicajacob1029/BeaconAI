import time

import numpy as np
import pytest
from scipy import sparse

from beaconai.explain.baseline import baseline_vector
from beaconai.explain.explainer import SPACE_NOTE, ChampionExplainer
from beaconai.explain.labels import FeatureLabeler
from beaconai.explain.linear import baseline_logits, logits
from beaconai.models.calibrated import Abstain, CalibratedClassifier, Differential
from beaconai.models.calibrators import IdentityCalibrator, calibrator_from_dict
from beaconai.models.registry import build_model

N_CLASSES = 4
LABELS = [{"name": f"Cond {i}", "icd10": "", "severity": 3} for i in range(N_CLASSES)]
FEATURES = [
    *[{"index": j, "name": f"E_{j}", "kind": "binary", "label": f"Symptom {j}?"} for j in range(8)],
    {"index": 8, "name": "AGE_BIN_0", "kind": "age_bucket", "label": "Age 0-39"},
    {"index": 9, "name": "AGE_BIN_40", "kind": "age_bucket", "label": "Age 40+"},
    {"index": 10, "name": "AGE_SCALED", "kind": "age_scaled", "label": "Age (scaled)"},
    {"index": 11, "name": "SEX_M", "kind": "sex", "label": "Sex: male"},
]
N_FEATURES = len(FEATURES)
MEANS = {"AGE_BIN_0": 0.6, "AGE_BIN_40": 0.4, "AGE_SCALED": 0.35, "SEX_M": 0.48}
M = baseline_vector(FEATURES, MEANS)


def _model(n_classes=N_CLASSES, n_features=N_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(400) % n_classes
    X = (rng.random((400, n_features)) < 0.15).astype(np.float32)
    X[np.arange(400), y % min(8, n_features)] = 1.0
    return build_model("logreg", {"max_iter": 300}, n_classes, 0).fit(sparse.csr_matrix(X), y)


def _explainer(threshold=0.0, calibrator=None, baseline=M):
    clf = CalibratedClassifier(
        _model(),
        calibrator or IdentityCalibrator(),
        LABELS,
        [f["name"] for f in FEATURES],
        threshold,
        3,
    )
    return ChampionExplainer(clf, FeatureLabeler(FEATURES, 10, 110), baseline, 5, 3)


def _row(cols: dict[int, float]) -> sparse.csr_matrix:
    x = np.zeros((1, N_FEATURES), dtype=np.float32)
    for j, v in cols.items():
        x[0, j] = v
    return sparse.csr_matrix(x)


# Symptoms 1 and 5, age 44 (bucket 40+), female (SEX_M = 0).
ROW = _row({1: 1.0, 5: 1.0, 9: 1.0, 10: 44 / 110})


def test_explains_each_item_and_reproduces_the_logit() -> None:
    exp = _explainer()
    out = exp.explain_row(ROW)
    assert isinstance(out.prediction, Differential)
    assert [e.condition for e in out.explanations] == [i.condition for i in out.prediction.items]
    first = out.explanations[0]
    c = [lab["name"] for lab in LABELS].index(first.condition)
    assert first.baseline_logit == pytest.approx(baseline_logits(exp.W, exp.b, M)[c])
    assert first.logit == pytest.approx(logits(exp.W, exp.b, ROW)[c])
    total = sum(x.contribution for x in first.supporting + first.opposing)
    assert first.baseline_logit + total == pytest.approx(first.logit)  # 4 groups <= 5 + 3
    assert out.space == SPACE_NOTE
    assert "before" in SPACE_NOTE and "calibration" in SPACE_NOTE and "average" in SPACE_NOTE


def test_demographics_are_grouped_and_a_female_patient_gets_a_sex_contribution() -> None:
    out = _explainer().explain_row(ROW)
    items = {x.feature: x for x in out.explanations[0].supporting + out.explanations[0].opposing}
    assert {"AGE", "SEX"} <= set(items)
    assert not {"AGE_BIN_0", "AGE_BIN_40", "AGE_SCALED", "SEX_M"} & set(items)
    assert items["SEX"].description.startswith("Sex: female")
    assert items["AGE"].description.startswith("Age: 44")
    assert "Symptom 1?: yes" in {x.description for x in items.values()}


def test_male_and_female_contributions_have_the_same_meaning() -> None:
    # Measured against the training mix (48% male), female and male move the score in
    # opposite directions for every condition, and neither is the silent baseline.
    exp = _explainer()
    male_row = _row({1: 1.0, 5: 1.0, 9: 1.0, 10: 44 / 110, 11: 1.0})
    for c in range(N_CLASSES):
        female = {x.feature: x.contribution for x in exp.grouped_contributions(ROW, c)}
        male = {x.feature: x.contribution for x in exp.grouped_contributions(male_row, c)}
        assert female["SEX"] == pytest.approx(exp.W[c, 11] * (0 - 0.48))
        assert male["SEX"] == pytest.approx(exp.W[c, 11] * (1 - 0.48))


def test_grouped_contributions_sum_to_the_logit() -> None:
    exp = _explainer()
    for c in range(N_CLASSES):
        total = sum(x.contribution for x in exp.grouped_contributions(ROW, c))
        z = logits(exp.W, exp.b, ROW)[c]
        assert baseline_logits(exp.W, exp.b, M)[c] + total == pytest.approx(z)


def test_without_baseline_only_present_features_contribute() -> None:
    out = _explainer(baseline=None).explain_row(_row({1: 1.0, 5: 1.0}))
    first = out.explanations[0]
    assert {x.feature for x in first.supporting + first.opposing} <= {"E_1", "E_5"}


def test_contrast_top1_vs_top2_is_exact() -> None:
    out = _explainer().explain_row(ROW)
    con = out.contrast
    assert con is not None
    assert (con.condition, con.versus) == (
        out.prediction.items[0].condition,
        out.prediction.items[1].condition,
    )
    assert con.logit_gap >= 0 and not con.ranking_changed_by_calibration
    total = sum(x.contribution for x in con.favors_condition + con.favors_versus)
    assert con.baseline_gap + total == pytest.approx(con.logit_gap)


def test_flags_when_calibration_flips_the_top_two() -> None:
    # Isotonic maps that shrink class 0 to 1% and classes 2-3 to 0.01%: when the model's own
    # top-1 is class 0 (row with symptom 0), calibration shows class 1 first and class 0 second.
    flip = {
        "method": "isotonic",
        "x": [[0.0, 1.0]] * N_CLASSES,
        "y": [[0.0, 0.01], [0.0, 1.0], [0.0, 0.0001], [0.0, 0.0001]],
    }
    exp = _explainer(calibrator=calibrator_from_dict(flip))
    row = _row({0: 1.0})
    assert int(np.argmax(exp.clf.model.predict_proba(row))) == 0  # precondition
    con = exp.explain_row(row).contrast
    assert (con.condition, con.versus) == ("Cond 1", "Cond 0")
    assert con.logit_gap < 0
    assert con.ranking_changed_by_calibration
    assert "calibration" in con.note.lower()


def test_abstain_still_explained() -> None:
    out = _explainer(threshold=1.0).explain_row(ROW)
    assert isinstance(out.prediction, Abstain)
    assert len(out.explanations) == 3 and out.contrast is not None


def test_refuses_non_linear_base() -> None:
    X = np.tile(ROW.toarray(), (8, 1))
    xgb = build_model("xgboost", {"n_estimators": 5}, N_CLASSES, 0).fit(X, np.arange(8) % 4)
    clf = CalibratedClassifier(
        xgb, IdentityCalibrator(), LABELS, [f["name"] for f in FEATURES], 0.0, 3
    )
    with pytest.raises(TypeError, match="logistic regression"):
        ChampionExplainer(clf, FeatureLabeler(FEATURES, 10, 110))


def test_baseline_length_must_match_features() -> None:
    clf = _explainer().clf
    with pytest.raises(ValueError, match="baseline"):
        ChampionExplainer(clf, FeatureLabeler(FEATURES, 10, 110), np.zeros(3))


def test_latency_on_a_realistically_sized_model() -> None:
    n_classes, n_features = 49, 922
    feats = [
        {"index": j, "name": f"E_{j}", "kind": "binary", "label": f"S{j}"}
        for j in range(n_features)
    ]
    labels = [{"name": f"C{i}", "icd10": "", "severity": 3} for i in range(n_classes)]
    clf = CalibratedClassifier(
        _model(n_classes, n_features, seed=1),
        IdentityCalibrator(),
        labels,
        [f["name"] for f in feats],
        0.5,
        3,
    )
    exp = ChampionExplainer(clf, FeatureLabeler(feats, 10, 110))
    rng = np.random.default_rng(2)
    rows = [
        sparse.csr_matrix((rng.random((1, n_features)) < 0.025).astype(np.float32))
        for _ in range(100)
    ]
    times = []
    for r in rows:
        start = time.perf_counter()
        exp.explain_row(r)
        times.append((time.perf_counter() - start) * 1000)
    assert np.percentile(times, 95) < 50

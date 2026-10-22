"""Why the champion ranks conditions the way it does, in plain words.

Explanations are exact for the logistic regression base model, in its logit space, before
isotonic calibration. Calibration changes the probabilities shown to the user (and occasionally
their order), not these scores. Each contribution is measured against a baseline patient: no
evidence, and the training-average age and sex mix. All age columns form one "Age" contribution
and the sex column one "Sex" contribution. The contrast between the top two conditions is the
most useful explanation for a triage user, and the only one free of the multinomial
"relative to the average condition" ambiguity.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import sparse

from beaconai.eval.metrics import top_k_indices
from beaconai.explain.baseline import DEMOGRAPHIC_KINDS
from beaconai.explain.labels import FeatureLabeler
from beaconai.explain.linear import (
    Pairs,
    baseline_logits,
    contrast_contributions,
    contributions,
    logits,
)
from beaconai.models.calibrated import (
    Abstain,
    CalibratedClassifier,
    Differential,
    PatientEvidence,
    to_prediction,
)
from beaconai.models.logreg import LogRegModel

SPACE_NOTE = (
    "Contributions are in the model's logit space before isotonic calibration. Each one is "
    "measured against a baseline patient with no evidence and the training-average age and sex "
    "mix. Calibration changes the probabilities shown, not these scores."
)
AGE, SEX = "AGE", "SEX"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Contribution(_Frozen):
    feature: str  # column name, or AGE / SEX for the grouped demographic contributions
    description: str
    contribution: float


class ConditionExplanation(_Frozen):
    """Support for one condition relative to the average condition (centered coefficients)."""

    condition: str
    logit: float
    baseline_logit: float
    supporting: list[Contribution]
    opposing: list[Contribution]


class ContrastExplanation(_Frozen):
    """Why `condition` rather than `versus`: positive contributions favor `condition`."""

    condition: str
    versus: str
    logit_gap: float
    baseline_gap: float
    favors_condition: list[Contribution]
    favors_versus: list[Contribution]
    ranking_changed_by_calibration: bool
    note: str | None = None


class ExplainedPrediction(_Frozen):
    prediction: Differential | Abstain
    explanations: list[ConditionExplanation]
    contrast: ContrastExplanation | None
    space: str = SPACE_NOTE


class ChampionExplainer:
    def __init__(
        self,
        clf: CalibratedClassifier,
        labeler: FeatureLabeler,
        baseline: np.ndarray | None = None,
        top_positive: int = 5,
        top_negative: int = 3,
    ) -> None:
        if not isinstance(clf.model, LogRegModel):
            raise TypeError(
                "Exact explanations are implemented for a logistic regression champion only; "
                f"got {type(clf.model).__name__}."
            )
        self.clf = clf
        self.labeler = labeler
        self.W, self.b = clf.model.linear_parameters()
        if baseline is not None and len(baseline) != self.W.shape[1]:
            raise ValueError(f"baseline has {len(baseline)} values for {self.W.shape[1]} columns")
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=np.float64)
        self.base_logits = baseline_logits(self.W, self.b, self.baseline)
        self.top_positive = top_positive
        self.top_negative = top_negative
        self.names = [lab["name"] for lab in clf.labels]
        kinds = [labeler.kind(j) for j in range(self.W.shape[1])]
        self._group = {
            j: (SEX if k == "sex" else AGE) for j, k in enumerate(kinds) if k in DEMOGRAPHIC_KINDS
        }
        self._age_scaled = next((j for j, k in enumerate(kinds) if k == "age_scaled"), None)
        self._sex = next((j for j, k in enumerate(kinds) if k == "sex"), None)

    def _describe_group(self, group: str, values: dict[int, float]) -> str:
        m = self.baseline
        if group == AGE:
            j = self._age_scaled
            age = round(values.get(j, 0.0) * self.labeler.age_scale) if j is not None else None
            text = f"Age: {age}" if age is not None else "Age"
            if m is not None and j is not None:
                text += f" (training average {m[j] * self.labeler.age_scale:.0f})"
            return text
        j = self._sex
        text = "Sex: male" if j is not None and values.get(j, 0.0) else "Sex: female"
        if m is not None and j is not None:
            text += f" (training mix {m[j]:.0%} male)"
        return text

    def _group_pairs(self, pairs: Pairs, values: dict[int, float]) -> list[Contribution]:
        """Sum age and sex columns into one contribution each; evidence columns stay as they are."""
        grouped: dict[str, float] = defaultdict(float)
        out = []
        for j, v in pairs:
            if j in self._group:
                grouped[self._group[j]] += v
            else:
                out.append(
                    Contribution(
                        feature=self.labeler.name(j),
                        description=self.labeler.describe(j, values.get(j, 0.0)),
                        contribution=v,
                    )
                )
        for group, v in grouped.items():
            out.append(
                Contribution(
                    feature=group, description=self._describe_group(group, values), contribution=v
                )
            )
        return out

    def _top(self, items: list[Contribution]) -> tuple[list[Contribution], list[Contribution]]:
        """Largest positive and most negative; zeros excluded; ties by feature name."""
        pos = sorted(
            (x for x in items if x.contribution > 0), key=lambda x: (-x.contribution, x.feature)
        )
        neg = sorted(
            (x for x in items if x.contribution < 0), key=lambda x: (x.contribution, x.feature)
        )
        return pos[: self.top_positive], neg[: self.top_negative]

    @staticmethod
    def _values(row: sparse.csr_matrix) -> dict[int, float]:
        return dict(zip(row.indices.tolist(), row.data.astype(float).tolist(), strict=True))

    def grouped_contributions(self, x_row: sparse.csr_matrix, c: int) -> list[Contribution]:
        """Every grouped contribution to condition c, untruncated (sums to logit - baseline)."""
        row = sparse.csr_matrix(x_row)
        return self._group_pairs(contributions(self.W, row, c, self.baseline), self._values(row))

    def explain_row(self, x_row: sparse.csr_matrix) -> ExplainedPrediction:
        row = sparse.csr_matrix(x_row)
        values = self._values(row)
        raw = self.clf.model.predict_proba(row)
        proba = self.clf.calibrator.transform(raw)[0]
        prediction = to_prediction(
            proba, self.clf.labels, self.clf.k, self.clf.threshold, tie_break=raw[0]
        )
        # Same order as the prediction: calibrated probability, ties by the model's own score.
        shown = [int(c) for c in top_k_indices(proba[None, :], self.clf.k, tie_break=raw)[0]]
        z = logits(self.W, self.b, row)

        explanations = []
        for c in shown:
            pairs = contributions(self.W, row, c, self.baseline)
            supporting, opposing = self._top(self._group_pairs(pairs, values))
            explanations.append(
                ConditionExplanation(
                    condition=self.names[c],
                    logit=float(z[c]),
                    baseline_logit=float(self.base_logits[c]),
                    supporting=supporting,
                    opposing=opposing,
                )
            )

        contrast = None
        if len(shown) >= 2:
            a, b = shown[0], shown[1]
            pairs = contrast_contributions(self.W, row, a, b, self.baseline)
            favors_a, favors_b = self._top(self._group_pairs(pairs, values))
            gap = float(z[a] - z[b])
            flipped = gap < 0
            contrast = ContrastExplanation(
                condition=self.names[a],
                versus=self.names[b],
                logit_gap=gap,
                baseline_gap=float(self.base_logits[a] - self.base_logits[b]),
                favors_condition=favors_a,
                favors_versus=favors_b,
                ranking_changed_by_calibration=flipped,
                note=(
                    f"Calibration ranks {self.names[a]} above {self.names[b]}, but the model's "
                    "own scores rank them the other way; the reasons below explain the scores."
                    if flipped
                    else None
                ),
            )
        return ExplainedPrediction(
            prediction=prediction, explanations=explanations, contrast=contrast
        )

    def explain_evidence(self, evidence: PatientEvidence) -> ExplainedPrediction:
        return self.explain_row(self.clf.encode(evidence))

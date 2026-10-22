"""Plain-English wording for a feature column and its value (from features.json)."""

from __future__ import annotations

from typing import Any


class FeatureLabeler:
    def __init__(self, features: list[dict[str, Any]], scale_max: float, age_scale: float) -> None:
        self._features = sorted(features, key=lambda f: f["index"])
        self.scale_max = scale_max
        self.age_scale = age_scale

    def name(self, j: int) -> str:
        return self._features[j]["name"]

    def label(self, j: int) -> str:
        return self._features[j]["label"]

    def kind(self, j: int) -> str:
        return self._features[j].get("kind") or ""

    def describe(self, j: int, value: float) -> str:
        kind, label = self.kind(j), self.label(j)
        if kind == "binary":
            return f"{label}: yes"
        if kind == "scale":
            return f"{label}: {round(value * self.scale_max)}/{round(self.scale_max)}"
        if kind == "age_scaled":
            return f"Age: {round(value * self.age_scale)}"
        return label  # categorical / multi values, age buckets and sex already read as text

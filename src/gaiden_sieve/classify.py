"""Probability-based classification for the Phase 1 model."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from gaiden_sieve.data import build_input_text
from gaiden_sieve.profile import ProfileConfig


@dataclass(frozen=True, slots=True)
class Prediction:
    """One local classification result."""

    probability_relevant: float
    classification: str


def classification_from_probability(
    probability_relevant: float, profile: ProfileConfig
) -> str:
    """Map P(relevant) to the publication's three operational states."""

    if probability_relevant >= profile.relevant_threshold:
        return "relevant"
    if probability_relevant < profile.not_relevant_threshold:
        return "not_relevant"
    return "uncertain"


def predict_one(
    model: Pipeline,
    profile: ProfileConfig,
    *,
    title: str,
    summary: str,
) -> Prediction:
    """Classify one title + summary with predict_proba()."""

    text = build_input_text(
        title=title,
        summary=summary,
        text_fields=profile.text_fields,
    )
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index(profile.positive_label)
    probability = float(model.predict_proba([text])[0][positive_index])

    # uncertain は学習クラスではなく、P(relevant) の中間帯を運用上そう呼んでいる。
    return Prediction(
        probability_relevant=probability,
        classification=classification_from_probability(probability, profile),
    )

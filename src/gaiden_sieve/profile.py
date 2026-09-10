"""Profile configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ProfileValidationError(ValueError):
    """Raised when a profile config does not satisfy the v0 contract."""


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Validated settings that separate one publication from another."""

    profile: str
    positive_label: str
    precision_gate: float
    recall_gate: float
    relevant_threshold: float
    not_relevant_threshold: float
    text_fields: tuple[str, ...]


_REQUIRED_KEYS = {
    "profile",
    "positive_label",
    "precision_gate",
    "recall_gate",
    "relevant_threshold",
    "not_relevant_threshold",
    "text_fields",
}
_ALLOWED_TEXT_FIELDS = {"title", "summary"}


def _as_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{name} must be a number between 0 and 1")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ProfileValidationError(f"{name} must be between 0 and 1")
    return probability


def load_profile(path: str | Path) -> ProfileConfig:
    """Load and validate one YAML profile."""

    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ProfileValidationError("profile config must be a YAML mapping")

    missing = sorted(_REQUIRED_KEYS - raw.keys())
    if missing:
        raise ProfileValidationError(f"missing profile keys: {', '.join(missing)}")

    profile_name = raw["profile"]
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ProfileValidationError("profile must be a non-empty string")

    positive_label = raw["positive_label"]
    if positive_label != "relevant":
        raise ProfileValidationError("v0 positive_label must be 'relevant'")

    precision_gate = _as_probability("precision_gate", raw["precision_gate"])
    recall_gate = _as_probability("recall_gate", raw["recall_gate"])
    relevant_threshold = _as_probability("relevant_threshold", raw["relevant_threshold"])
    not_relevant_threshold = _as_probability(
        "not_relevant_threshold", raw["not_relevant_threshold"]
    )

    if not_relevant_threshold >= relevant_threshold:
        raise ProfileValidationError(
            "not_relevant_threshold must be lower than relevant_threshold"
        )

    text_fields = raw["text_fields"]
    if not isinstance(text_fields, list) or not text_fields:
        raise ProfileValidationError("text_fields must be a non-empty list")
    if any(not isinstance(field, str) for field in text_fields):
        raise ProfileValidationError("text_fields must contain only strings")
    if len(text_fields) != len(set(text_fields)):
        raise ProfileValidationError("text_fields must not contain duplicates")

    unexpected_fields = sorted(set(text_fields) - _ALLOWED_TEXT_FIELDS)
    if unexpected_fields:
        raise ProfileValidationError(
            "v0 text_fields may contain only title and summary: "
            + ", ".join(unexpected_fields)
        )

    # label や label_reason を特徴量へ混ぜないため、v0 の入力契約をここで固定する。
    if set(text_fields) != _ALLOWED_TEXT_FIELDS:
        raise ProfileValidationError("v0 text_fields must contain title and summary")

    return ProfileConfig(
        profile=profile_name.strip(),
        positive_label=positive_label,
        precision_gate=precision_gate,
        recall_gate=recall_gate,
        relevant_threshold=relevant_threshold,
        not_relevant_threshold=not_relevant_threshold,
        text_fields=tuple(text_fields),
    )

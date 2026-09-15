"""Evaluation and quality-gate helpers for GAIDEN SIEVE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.metrics import precision_recall_fscore_support

from gaiden_sieve.profile import ProfileConfig


@dataclass(frozen=True, slots=True)
class Metrics:
    """Metrics used to inspect the positive class."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class QualityGate:
    """Result of comparing candidate metrics with the profile's promotion gates."""

    precision_gate: float
    recall_gate: float
    passed: bool


def calculate_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], *, positive_label: str
) -> Metrics:
    """Calculate binary Precision / Recall / F1 for the publication's positive label."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=positive_label,
        zero_division=0,
    )
    return Metrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def evaluate_quality_gate(metrics: Metrics, profile: ProfileConfig) -> QualityGate:
    """Apply the profile's minimum Precision / Recall requirements."""

    return QualityGate(
        precision_gate=profile.precision_gate,
        recall_gate=profile.recall_gate,
        passed=(
            metrics.precision >= profile.precision_gate
            and metrics.recall >= profile.recall_gate
        ),
    )

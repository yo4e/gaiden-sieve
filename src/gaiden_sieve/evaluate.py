"""Evaluation helpers for the Phase 1 binary classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.metrics import precision_recall_fscore_support


@dataclass(frozen=True, slots=True)
class Metrics:
    """The three Phase 1 metrics used to inspect the positive class."""

    precision: float
    recall: float
    f1: float


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

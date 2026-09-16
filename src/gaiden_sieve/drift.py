"""Small observational drift report for Phase 3."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Sequence

from sklearn.pipeline import Pipeline

from gaiden_sieve.batch import InputItem
from gaiden_sieve.data import build_input_text
from gaiden_sieve.profile import ProfileConfig


def _oov_feature_rate(
    model: Pipeline,
    profile: ProfileConfig,
    items: Sequence[InputItem],
) -> float:
    """Estimate how often analyzer features are absent from the fitted TF-IDF vocabulary."""

    vectorizer = model.named_steps["tfidf"]
    vocabulary = getattr(vectorizer, "vocabulary_", {})
    analyzer = vectorizer.build_analyzer()
    total = 0
    unknown = 0
    for item in items:
        text = build_input_text(
            title=item.title,
            summary=item.summary,
            text_fields=profile.text_fields,
        )
        for feature in analyzer(text):
            total += 1
            if feature not in vocabulary:
                unknown += 1
    return unknown / total if total else 0.0


def build_drift_report(
    model: Pipeline,
    profile: ProfileConfig,
    items: Sequence[InputItem],
    classified_rows: Sequence[dict[str, object]],
    *,
    model_version: str,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Aggregate simple signals that can reveal input-distribution changes."""

    if len(items) != len(classified_rows):
        raise ValueError("items and classified_rows must have the same length")
    if not items:
        raise ValueError("drift report requires at least one item")

    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must include a timezone")

    probabilities = [float(row["probability_relevant"]) for row in classified_rows]
    counts = {"relevant": 0, "uncertain": 0, "not_relevant": 0}
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in classified_rows:
        classification = str(row["classification"])
        counts[classification] += 1
        by_source[str(row["source"])].append(row)

    source_distribution: dict[str, object] = {}
    for source, rows in sorted(by_source.items()):
        source_counts = {"relevant": 0, "uncertain": 0, "not_relevant": 0}
        for row in rows:
            source_counts[str(row["classification"])] += 1
        source_distribution[source] = {
            "count": len(rows),
            "mean_probability_relevant": fmean(
                float(row["probability_relevant"]) for row in rows
            ),
            "classification_distribution": source_counts,
        }

    total = len(classified_rows)
    return {
        "schema_version": 1,
        "profile": profile.profile,
        "model_version": model_version,
        "generated_at": timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "input_count": total,
        "mean_probability_relevant": fmean(probabilities),
        "uncertain_rate": counts["uncertain"] / total,
        "oov_feature_rate": _oov_feature_rate(model, profile, items),
        "classification_distribution": counts,
        "source_distribution": source_distribution,
        "thresholds": {
            "relevant": profile.relevant_threshold,
            "not_relevant": profile.not_relevant_threshold,
        },
    }


def write_drift_report(path: str | Path, report: dict[str, object]) -> Path:
    """Persist one drift report atomically."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path

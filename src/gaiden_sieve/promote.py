"""Explicit candidate-to-production promotion."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from gaiden_sieve.artifacts import (
    candidate_model_path,
    production_model_path,
    save_production_metadata,
    sha256_file,
    verify_candidate_integrity,
)
from gaiden_sieve.evaluate import Metrics, evaluate_quality_gate
from gaiden_sieve.profile import ProfileConfig


class PromotionRejected(RuntimeError):
    """Raised when a candidate does not meet the current profile gate."""


def promote_candidate(
    *,
    profile: ProfileConfig,
    artifacts_dir: str | Path,
    model_version: str,
    promoted_at: datetime | None = None,
) -> dict[str, object]:
    """Explicitly replace production with one gate-passing candidate."""

    metadata = verify_candidate_integrity(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    raw_metrics = metadata.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise PromotionRejected("candidate metadata does not contain metrics")
    metrics = Metrics(
        precision=float(raw_metrics["precision"]),
        recall=float(raw_metrics["recall"]),
        f1=float(raw_metrics["f1"]),
    )
    gate = evaluate_quality_gate(metrics, profile)

    # 新しく学習できたことと、本番採用してよいことは別。gate を越えない candidate は production を触らない。
    if not gate.passed:
        raise PromotionRejected(
            "candidate failed quality gate: "
            f"precision={metrics.precision:.3f} (required {profile.precision_gate:.3f}), "
            f"recall={metrics.recall:.3f} (required {profile.recall_gate:.3f})"
        )

    source = candidate_model_path(artifacts_dir, profile.profile, model_version)
    destination = production_model_path(artifacts_dir, profile.profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)

    timestamp = promoted_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise PromotionRejected("promoted_at must include a timezone")

    production_metadata = dict(metadata)
    production_metadata.update(
        {
            "stage": "production",
            "source_candidate": model_version,
            "quality_gate": asdict(gate),
            "promoted_at": timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "model_artifact_hash": sha256_file(destination),
        }
    )
    save_production_metadata(
        production_metadata,
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
    )
    return production_metadata

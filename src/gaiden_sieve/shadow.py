"""Shadow-mode comparison helpers for real publication inputs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sklearn.pipeline import Pipeline

from gaiden_sieve.artifacts import load_candidate_model, verify_candidate_integrity
from gaiden_sieve.batch import (
    classify_batch,
    load_input_jsonl,
    uncertain_rows,
    write_classified_jsonl,
    write_uncertain_queue,
)
from gaiden_sieve.drift import build_drift_report, write_drift_report
from gaiden_sieve.evaluate import Metrics, evaluate_quality_gate
from gaiden_sieve.profile import ProfileConfig


class ShadowError(RuntimeError):
    """Raised when shadow observation cannot be performed safely."""


def _utc_timestamp(value: datetime | None = None) -> tuple[datetime, str]:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ShadowError("generated_at must include a timezone")
    utc = timestamp.astimezone(timezone.utc)
    return utc, utc.isoformat().replace("+00:00", "Z")


def load_shadow_records(path: str | Path) -> list[dict[str, Any]]:
    """Load AI外電 shadow rows while preserving the current admission observation."""

    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ShadowError(
                    f"{input_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise ShadowError(
                    f"{input_path}:{line_number}: each JSONL row must be an object"
                )

            for field in ("id", "source", "title"):
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ShadowError(
                        f"{input_path}:{line_number}: {field} must be a non-empty string"
                    )
            summary = raw.get("summary", "")
            if not isinstance(summary, str):
                raise ShadowError(
                    f"{input_path}:{line_number}: summary must be a string"
                )
            mode = raw.get("admission_mode")
            if mode not in {"all", "filtered", "none"}:
                raise ShadowError(
                    f"{input_path}:{line_number}: admission_mode must be all/filtered/none"
                )
            accepted = raw.get("admission_accepted")
            if not isinstance(accepted, bool):
                raise ShadowError(
                    f"{input_path}:{line_number}: admission_accepted must be a boolean"
                )
            policy = raw.get("admission_policy")
            if policy is not None and not isinstance(policy, str):
                raise ShadowError(
                    f"{input_path}:{line_number}: admission_policy must be a string or null"
                )
            item_id = raw["id"]
            if item_id in seen_ids:
                raise ShadowError(f"{input_path}:{line_number}: duplicate id: {item_id}")
            seen_ids.add(item_id)
            normalized = dict(raw)
            normalized["summary"] = summary
            rows.append(normalized)

    if not rows:
        raise ShadowError(f"{input_path}: no shadow input items found")
    return rows


def load_shadow_candidate(
    profile: ProfileConfig,
    *,
    artifacts_dir: str | Path,
    model_version: str,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Load a gate-passing candidate without promoting it to production."""

    metadata = verify_candidate_integrity(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    raw_metrics = metadata.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ShadowError("candidate metadata does not contain metrics")
    try:
        metrics = Metrics(
            precision=float(raw_metrics["precision"]),
            recall=float(raw_metrics["recall"]),
            f1=float(raw_metrics["f1"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ShadowError("candidate metrics are incomplete") from exc

    # shadow でも gate を迂回しない。production へ昇格はしないが、観察対象の品質条件は現在設定で再確認する。
    gate = evaluate_quality_gate(metrics, profile)
    if not gate.passed:
        raise ShadowError("candidate failed current quality gate; shadow run was stopped")

    model = load_candidate_model(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    return model, metadata, asdict(gate)


def build_comparison_report(
    shadow_records: Sequence[dict[str, Any]],
    classified_rows: Sequence[dict[str, object]],
    *,
    profile: ProfileConfig,
    model_version: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Compare existing admission with SIEVE predictions without declaring a winner."""

    _, generated_at_text = _utc_timestamp(generated_at)
    predictions = {str(row["id"]): row for row in classified_rows}
    input_ids = {str(row["id"]) for row in shadow_records}
    if set(predictions) != input_ids:
        raise ShadowError("shadow input and classified output ids do not match")

    distribution = {"relevant": 0, "uncertain": 0, "not_relevant": 0}
    admission_accept_count = 0
    agreement = 0
    disagreement = 0
    confident_compared = 0
    disagreements: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, Any]] = {}

    for record in shadow_records:
        item_id = str(record["id"])
        source = str(record["source"])
        prediction = predictions[item_id]
        classification = str(prediction["classification"])
        if classification not in distribution:
            raise ShadowError(f"unexpected classification: {classification}")
        distribution[classification] += 1

        admission_accepted = bool(record["admission_accepted"])
        admission_accept_count += int(admission_accepted)

        source_stats = by_source.setdefault(
            source,
            {
                "input_count": 0,
                "admission_accept_count": 0,
                "sieve_distribution": {
                    "relevant": 0,
                    "uncertain": 0,
                    "not_relevant": 0,
                },
                "confident_disagreement_count": 0,
            },
        )
        source_stats["input_count"] += 1
        source_stats["admission_accept_count"] += int(admission_accepted)
        source_stats["sieve_distribution"][classification] += 1

        if classification == "uncertain":
            continue

        confident_compared += 1
        sieve_accept = classification == "relevant"
        if sieve_accept == admission_accepted:
            agreement += 1
            continue

        disagreement += 1
        source_stats["confident_disagreement_count"] += 1
        disagreements.append(
            {
                "id": item_id,
                "source": source,
                "title": record["title"],
                "admission_mode": record["admission_mode"],
                "admission_policy": record.get("admission_policy"),
                "admission_accepted": admission_accepted,
                "sieve_classification": classification,
                "probability_relevant": prediction["probability_relevant"],
            }
        )

    agreement_rate = agreement / confident_compared if confident_compared else None
    revisions = {
        str(record["ai_gaiden_commit"])
        for record in shadow_records
        if record.get("ai_gaiden_commit")
    }
    source_revision = next(iter(revisions)) if len(revisions) == 1 else None
    return {
        "schema_version": 1,
        "kind": "ai-gaiden-admission-vs-sieve-shadow",
        "profile": profile.profile,
        "candidate_model_version": model_version,
        "ai_gaiden_commit": source_revision,
        "generated_at": generated_at_text,
        "input_count": len(shadow_records),
        "existing_admission": {
            "accepted": admission_accept_count,
            "rejected": len(shadow_records) - admission_accept_count,
        },
        "sieve_distribution": distribution,
        "uncertain_count": distribution["uncertain"],
        "confident_comparison": {
            "compared": confident_compared,
            "agreement": agreement,
            "disagreement": disagreement,
            "agreement_rate": agreement_rate,
        },
        "by_source": {key: by_source[key] for key in sorted(by_source)},
        "disagreements": disagreements,
        "interpretation": (
            "A disagreement is an observation, not a ground-truth error. "
            "Human review is required before changing publication policy."
        ),
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


def run_shadow_candidate(
    *,
    profile: ProfileConfig,
    artifacts_dir: str | Path,
    model_version: str,
    input_path: str | Path,
    classified_output: str | Path,
    drift_output: str | Path,
    uncertain_output: str | Path,
    comparison_output: str | Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Run one candidate-only shadow observation and persist inspectable reports."""

    shadow_records = load_shadow_records(input_path)
    items = load_input_jsonl(input_path)
    model, metadata, gate = load_shadow_candidate(
        profile,
        artifacts_dir=artifacts_dir,
        model_version=model_version,
    )
    timestamp, _ = _utc_timestamp(generated_at)
    classified_rows = classify_batch(
        model,
        profile,
        items,
        model_version=model_version,
        classified_at=timestamp,
    )
    write_classified_jsonl(classified_output, classified_rows)
    write_uncertain_queue(uncertain_output, classified_rows)

    drift_report = build_drift_report(
        model,
        profile,
        items,
        classified_rows,
        model_version=model_version,
        generated_at=timestamp,
    )
    write_drift_report(drift_output, drift_report)

    comparison_report = build_comparison_report(
        shadow_records,
        classified_rows,
        profile=profile,
        model_version=model_version,
        generated_at=timestamp,
    )
    _write_json(comparison_output, comparison_report)

    return {
        "profile": profile.profile,
        "stage": "candidate-shadow",
        "candidate_model_version": metadata["model_version"],
        "quality_gate": gate,
        "input_count": len(classified_rows),
        "uncertain_count": len(uncertain_rows(classified_rows)),
        "confident_disagreement_count": comparison_report["confident_comparison"][
            "disagreement"
        ],
        "classified_output": str(Path(classified_output)),
        "drift_report_path": str(Path(drift_output)),
        "uncertain_queue_path": str(Path(uncertain_output)),
        "comparison_report_path": str(Path(comparison_output)),
        "production_promoted": False,
    }

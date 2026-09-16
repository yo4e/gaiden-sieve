"""Batch input, classification, and uncertain-queue helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from sklearn.pipeline import Pipeline

from gaiden_sieve.classify import predict_one
from gaiden_sieve.profile import ProfileConfig


class BatchValidationError(ValueError):
    """Raised when an incoming JSONL batch violates the v0 contract."""


@dataclass(frozen=True, slots=True)
class InputItem:
    """One unlabeled RSS-like item waiting for production classification."""

    id: str
    source: str
    title: str
    summary: str


_REQUIRED_FIELDS = {"id", "source", "title"}


def _required_string(record: dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise BatchValidationError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise BatchValidationError(f"{field} must not be empty")
    return value


def validate_input_item(record: dict[str, Any]) -> InputItem:
    """Validate one unlabeled input row."""

    missing = sorted(_REQUIRED_FIELDS - record.keys())
    if missing:
        raise BatchValidationError(f"missing input fields: {', '.join(missing)}")
    summary = record.get("summary", "")
    if not isinstance(summary, str):
        raise BatchValidationError("summary must be a string")

    return InputItem(
        id=_required_string(record, "id"),
        source=_required_string(record, "source"),
        title=_required_string(record, "title"),
        summary=summary,
    )


def load_input_jsonl(path: str | Path) -> list[InputItem]:
    """Load an unlabeled JSONL batch while rejecting duplicate ids."""

    input_path = Path(path)
    items: list[InputItem] = []
    seen_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchValidationError(
                    f"{input_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise BatchValidationError(
                    f"{input_path}:{line_number}: each JSONL row must be an object"
                )
            try:
                item = validate_input_item(raw)
            except BatchValidationError as exc:
                raise BatchValidationError(
                    f"{input_path}:{line_number}: {exc}"
                ) from exc
            if item.id in seen_ids:
                raise BatchValidationError(
                    f"{input_path}:{line_number}: duplicate id: {item.id}"
                )
            seen_ids.add(item.id)
            items.append(item)

    if not items:
        raise BatchValidationError(f"{input_path}: no input items found")
    return items


def _utc_timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise BatchValidationError("classified_at must include a timezone")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_batch(
    model: Pipeline,
    profile: ProfileConfig,
    items: Sequence[InputItem],
    *,
    model_version: str,
    classified_at: datetime | None = None,
) -> list[dict[str, object]]:
    """Classify one batch with the exact same profile thresholds as single-item inference."""

    timestamp = _utc_timestamp(classified_at)
    rows: list[dict[str, object]] = []
    for item in items:
        prediction = predict_one(
            model,
            profile,
            title=item.title,
            summary=item.summary,
        )
        rows.append(
            {
                "id": item.id,
                "source": item.source,
                "title": item.title,
                "summary": item.summary,
                "probability_relevant": prediction.probability_relevant,
                "classification": prediction.classification,
                "classified_at": timestamp,
                "model_version": model_version,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_classified_jsonl(path: str | Path, rows: Sequence[dict[str, object]]) -> Path:
    """Persist all batch predictions."""

    output_path = Path(path)
    _write_jsonl(output_path, rows)
    return output_path


def uncertain_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Return only uncertain items in the compact active-learning queue schema."""

    queue: list[dict[str, object]] = []
    for row in rows:
        if row.get("classification") != "uncertain":
            continue
        queue.append(
            {
                "id": row["id"],
                "source": row["source"],
                "title": row["title"],
                "summary": row["summary"],
                "probability": row["probability_relevant"],
                "classified_at": row["classified_at"],
                "model_version": row["model_version"],
            }
        )
    return queue


def write_uncertain_queue(path: str | Path, rows: Sequence[dict[str, object]]) -> Path:
    """Persist uncertain items as the v0 human/LLM review handoff queue."""

    output_path = Path(path)
    _write_jsonl(output_path, uncertain_rows(rows))
    return output_path

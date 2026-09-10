"""Training-data schema and JSONL loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


class DataValidationError(ValueError):
    """Raised when labeled training data violates the v0 contract."""


@dataclass(frozen=True, slots=True)
class LabeledItem:
    """One labeled RSS-like item used as training or evaluation data."""

    id: str
    source: str
    title: str
    summary: str
    label: str
    label_source: str
    label_reason: str
    labeled_at: str


_REQUIRED_FIELDS = {
    "id",
    "source",
    "title",
    "summary",
    "label",
    "label_source",
    "label_reason",
    "labeled_at",
}
_ALLOWED_LABELS = {"relevant", "not_relevant"}
_ALLOWED_LABEL_SOURCES = {"llm", "human", "rule", "imported"}


def _required_string(record: dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = record[field]
    if not isinstance(value, str):
        raise DataValidationError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise DataValidationError(f"{field} must not be empty")
    return value


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataValidationError("labeled_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise DataValidationError("labeled_at must include a timezone")


def validate_labeled_item(record: dict[str, Any]) -> LabeledItem:
    """Validate one raw mapping and return the typed training-data record."""

    missing = sorted(_REQUIRED_FIELDS - record.keys())
    if missing:
        raise DataValidationError(f"missing data fields: {', '.join(missing)}")

    item_id = _required_string(record, "id")
    source = _required_string(record, "source")
    title = _required_string(record, "title")
    summary = _required_string(record, "summary", allow_empty=True)
    label = _required_string(record, "label")
    label_source = _required_string(record, "label_source")
    label_reason = _required_string(record, "label_reason")
    labeled_at = _required_string(record, "labeled_at")

    if label not in _ALLOWED_LABELS:
        raise DataValidationError(
            "label must be either 'relevant' or 'not_relevant' in v0"
        )
    if label_source not in _ALLOWED_LABEL_SOURCES:
        raise DataValidationError(
            "label_source must be one of: " + ", ".join(sorted(_ALLOWED_LABEL_SOURCES))
        )
    _validate_timestamp(labeled_at)

    return LabeledItem(
        id=item_id,
        source=source,
        title=title,
        summary=summary,
        label=label,
        label_source=label_source,
        label_reason=label_reason,
        labeled_at=labeled_at,
    )


def load_labeled_jsonl(path: str | Path) -> list[LabeledItem]:
    """Load a JSONL training fixture and validate every row."""

    data_path = Path(path)
    items: list[LabeledItem] = []
    seen_ids: set[str] = set()

    with data_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{data_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise DataValidationError(
                    f"{data_path}:{line_number}: each JSONL row must be an object"
                )
            try:
                item = validate_labeled_item(raw)
            except DataValidationError as exc:
                raise DataValidationError(
                    f"{data_path}:{line_number}: {exc}"
                ) from exc

            # 同一 id の重複は、後の train/test 分割で同じ記事が両側へ漏れる原因になるため早めに止める。
            if item.id in seen_ids:
                raise DataValidationError(
                    f"{data_path}:{line_number}: duplicate id: {item.id}"
                )
            seen_ids.add(item.id)
            items.append(item)

    if not items:
        raise DataValidationError(f"{data_path}: no labeled items found")
    return items


def build_text(item: LabeledItem, text_fields: Iterable[str]) -> str:
    """Build the future model input from profile-approved text fields only."""

    fields = tuple(text_fields)
    if not fields:
        raise DataValidationError("text_fields must not be empty")

    parts: list[str] = []
    for field in fields:
        if field not in {"title", "summary"}:
            raise DataValidationError(f"unsupported text field: {field}")
        parts.append(getattr(item, field).strip())

    # 教師ラベルや理由文をここへ入れないこと自体が、data leakage 防止の境界になる。
    return "\n\n".join(part for part in parts if part)

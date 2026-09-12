from pathlib import Path

import pytest

from gaiden_sieve.data import (
    DataValidationError,
    build_input_text,
    build_text,
    load_labeled_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_loads_with_binary_labels() -> None:
    items = load_labeled_jsonl(ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl")

    assert len(items) == 20
    assert {item.label for item in items} == {"relevant", "not_relevant"}


def test_build_text_uses_only_profile_text_fields() -> None:
    item = load_labeled_jsonl(
        ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl"
    )[0]

    model_input = build_text(item, ("title", "summary"))

    assert item.title in model_input
    assert item.summary in model_input
    assert item.label not in model_input
    assert item.label_reason not in model_input


def test_build_input_text_rejects_empty_input() -> None:
    with pytest.raises(DataValidationError, match="must not both be empty"):
        build_input_text(title="", summary="", text_fields=("title", "summary"))


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    row = (
        '{"id":"same","source":"x","title":"t","summary":"s",'
        '"label":"relevant","label_source":"human",'
        '"label_reason":"reason","labeled_at":"2026-09-10T00:00:00Z"}'
    )
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="duplicate id"):
        load_labeled_jsonl(path)

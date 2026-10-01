from pathlib import Path

import pytest

from gaiden_sieve.data import (
    DataValidationError,
    build_input_text,
    build_text,
    labeled_dataset_hash,
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


def test_sharded_dataset_loads_in_filename_order_and_hashes_content(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    row_a = (
        '{"id":"a","source":"x","title":"A","summary":"",'
        '"label":"relevant","label_source":"llm",'
        '"label_reason":"reason","labeled_at":"2026-10-01T00:00:00Z"}'
    )
    row_b = (
        '{"id":"b","source":"y","title":"B","summary":"",'
        '"label":"not_relevant","label_source":"llm",'
        '"label_reason":"reason","labeled_at":"2026-10-01T00:01:00Z"}'
    )
    (dataset / "02.jsonl").write_text(row_b + "\n", encoding="utf-8")
    (dataset / "01.jsonl").write_text(row_a + "\n", encoding="utf-8")

    items = load_labeled_jsonl(dataset)
    first_hash = labeled_dataset_hash(dataset)

    assert [item.id for item in items] == ["a", "b"]
    assert first_hash.startswith("sha256:")

    (dataset / "02.jsonl").write_text(row_b.replace('"title":"B"', '"title":"B2"') + "\n", encoding="utf-8")
    assert labeled_dataset_hash(dataset) != first_hash


def test_duplicate_id_across_shards_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    row = (
        '{"id":"same","source":"x","title":"t","summary":"s",'
        '"label":"relevant","label_source":"llm",'
        '"label_reason":"reason","labeled_at":"2026-10-01T00:00:00Z"}'
    )
    (dataset / "01.jsonl").write_text(row + "\n", encoding="utf-8")
    (dataset / "02.jsonl").write_text(row + "\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="duplicate id"):
        load_labeled_jsonl(dataset)

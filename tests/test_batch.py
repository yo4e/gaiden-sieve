import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from gaiden_sieve.batch import (
    classify_batch,
    load_input_jsonl,
    uncertain_rows,
    write_uncertain_queue,
)
from gaiden_sieve.profile import load_profile


ROOT = Path(__file__).resolve().parents[1]


class SequencedModel:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.named_steps = {
            "classifier": SimpleNamespace(classes_=["not_relevant", "relevant"])
        }

    def predict_proba(self, _texts):
        probability = next(self.probabilities)
        return [[1.0 - probability, probability]]


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "incoming.jsonl"
    rows = [
        {"id": "a", "source": "one", "title": "AI model", "summary": "release"},
        {"id": "b", "source": "one", "title": "Borderline", "summary": "topic"},
        {"id": "c", "source": "two", "title": "Sports", "summary": "result"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_batch_uses_profile_thresholds_and_queue_keeps_only_uncertain(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    items = load_input_jsonl(_input_file(tmp_path))
    rows = classify_batch(
        SequencedModel([0.90, 0.60, 0.20]),
        profile,
        items,
        model_version="model-v1",
    )

    assert [row["classification"] for row in rows] == [
        "relevant",
        "uncertain",
        "not_relevant",
    ]
    queue = uncertain_rows(rows)
    assert [row["id"] for row in queue] == ["b"]
    assert set(queue[0]) == {
        "id",
        "source",
        "title",
        "summary",
        "probability",
        "classified_at",
        "model_version",
    }

    queue_path = tmp_path / "uncertain.jsonl"
    write_uncertain_queue(queue_path, rows)
    saved = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert saved == queue

    stricter = replace(
        profile,
        relevant_threshold=0.95,
        not_relevant_threshold=0.70,
    )
    changed = classify_batch(
        SequencedModel([0.90]),
        stricter,
        items[:1],
        model_version="model-v1",
    )
    assert changed[0]["classification"] == "uncertain"

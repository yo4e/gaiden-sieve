from datetime import datetime, timezone
from pathlib import Path

import pytest

from gaiden_sieve.artifacts import load_candidate_model, sha256_file
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.train import (
    RANDOM_SEED,
    TEST_SIZE,
    TrainingError,
    evaluate_candidate,
    train_and_evaluate,
    train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_train_and_evaluate_uses_reproducible_stratified_holdout() -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    items = load_labeled_jsonl(ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl")
    result = train_and_evaluate(items, profile)
    repeated = train_and_evaluate(items, profile)
    assert result.train_size == 16
    assert result.test_size == 4
    assert result.random_seed == RANDOM_SEED
    assert TEST_SIZE == 0.20
    assert result.metrics == repeated.metrics
    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
    assert 0.0 <= result.metrics.f1 <= 1.0
    assert list(result.model.named_steps) == ["tfidf", "classifier"]


def test_train_candidate_saves_reloadable_model_and_lineage(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    data_path = ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl"
    items = load_labeled_jsonl(data_path)
    metadata = train_candidate(
        items=items,
        data_path=data_path,
        profile=profile,
        artifacts_dir=tmp_path,
        code_commit="abc123",
        created_at=datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc),
    )
    version = str(metadata["model_version"])
    model = load_candidate_model(artifacts_dir=tmp_path, profile=profile.profile, model_version=version)
    assert list(model.named_steps) == ["tfidf", "classifier"]
    assert metadata["profile"] == "ai-gaiden"
    assert metadata["training_data_hash"] == sha256_file(data_path)
    assert metadata["code_commit"] == "abc123"
    assert metadata["random_seed"] == 42
    assert metadata["test_size"] == 0.20
    assert metadata["hyperparameters"]["classifier"]["name"] == "LogisticRegression"
    assert set(metadata["metrics"]) == {"precision", "recall", "f1"}
    assert metadata["quality_gate"]["passed"] is True

    metrics, evaluation = evaluate_candidate(
        data_path=data_path,
        profile=profile,
        artifacts_dir=tmp_path,
        model_version=version,
    )
    assert evaluation["metrics"] == metadata["metrics"]
    assert metrics.precision == 1.0


def test_train_candidate_rejects_items_that_do_not_match_hashed_data(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    data_path = ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl"
    items = load_labeled_jsonl(data_path)
    with pytest.raises(TrainingError, match="do not match"):
        train_candidate(
            items=items[:-1],
            data_path=data_path,
            profile=profile,
            artifacts_dir=tmp_path,
        )

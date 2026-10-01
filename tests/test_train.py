from datetime import datetime, timezone
from pathlib import Path

import pytest

from gaiden_sieve.artifacts import load_candidate_model, sha256_file
from gaiden_sieve.data import labeled_dataset_hash, load_labeled_jsonl
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


def test_real_bootstrap_uses_explicit_disjoint_holdout(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    training_path = ROOT / "profiles" / "ai-gaiden" / "bootstrap"
    evaluation_path = ROOT / "profiles" / "ai-gaiden" / "real-holdout"
    training_items = load_labeled_jsonl(training_path)
    evaluation_items = load_labeled_jsonl(evaluation_path)

    metadata = train_candidate(
        items=training_items,
        data_path=training_path,
        evaluation_items=evaluation_items,
        evaluation_data_path=evaluation_path,
        profile=profile,
        artifacts_dir=tmp_path,
        code_commit="phase4b1-test",
        created_at=datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc),
    )

    assert len(training_items) == 36
    assert len(evaluation_items) == 10
    assert {item.id for item in training_items}.isdisjoint(
        {item.id for item in evaluation_items}
    )
    assert metadata["evaluation_mode"] == "external_holdout"
    assert metadata["training_data_hash"] == labeled_dataset_hash(training_path)
    assert metadata["evaluation_data_hash"] == labeled_dataset_hash(evaluation_path)
    assert metadata["train_size"] == 36
    assert metadata["holdout_size"] == 10
    assert metadata["test_size"] is None
    assert metadata["hyperparameters"]["classifier"]["name"] == "LogisticRegression"
    assert metadata["quality_gate"]["passed"] is True

    metrics, evaluation = evaluate_candidate(
        data_path=training_path,
        evaluation_data_path=evaluation_path,
        profile=profile,
        artifacts_dir=tmp_path,
        model_version=str(metadata["model_version"]),
    )
    assert evaluation["evaluation_mode"] == "external_holdout"
    assert evaluation["metrics"] == metadata["metrics"]
    assert metrics.precision >= profile.precision_gate
    assert metrics.recall >= profile.recall_gate


def test_external_holdout_rejects_training_overlap(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    data_path = ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl"
    items = load_labeled_jsonl(data_path)

    with pytest.raises(TrainingError, match="must be disjoint"):
        train_candidate(
            items=items,
            data_path=data_path,
            evaluation_items=items,
            evaluation_data_path=data_path,
            profile=profile,
            artifacts_dir=tmp_path,
        )

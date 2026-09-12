from pathlib import Path

from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.train import RANDOM_SEED, TEST_SIZE, train_and_evaluate


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

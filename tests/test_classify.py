from pathlib import Path

from gaiden_sieve.classify import classification_from_probability, predict_one
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.train import train_and_evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_thresholds_create_uncertain_middle_band() -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")

    assert classification_from_probability(0.90, profile) == "relevant"
    assert classification_from_probability(0.60, profile) == "uncertain"
    assert classification_from_probability(0.20, profile) == "not_relevant"
    assert classification_from_probability(0.80, profile) == "relevant"
    assert classification_from_probability(0.40, profile) == "uncertain"


def test_predict_one_returns_probability_and_operational_class() -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    items = load_labeled_jsonl(ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl")
    result = train_and_evaluate(items, profile)

    prediction = predict_one(
        result.model,
        profile,
        title="AI lab releases a new language model",
        summary="The model improves reasoning and tool use.",
    )

    assert 0.0 <= prediction.probability_relevant <= 1.0
    assert prediction.classification in {"relevant", "uncertain", "not_relevant"}

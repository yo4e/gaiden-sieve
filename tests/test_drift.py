from pathlib import Path

import pytest

from gaiden_sieve.batch import InputItem
from gaiden_sieve.drift import build_drift_report
from gaiden_sieve.profile import load_profile


ROOT = Path(__file__).resolve().parents[1]


class FakeVectorizer:
    vocabulary_ = {"known": 0}

    def build_analyzer(self):
        return lambda text: text.lower().split()


class FakeModel:
    named_steps = {"tfidf": FakeVectorizer()}


def test_drift_report_aggregates_observational_signals() -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    items = [
        InputItem("a", "one", "known", ""),
        InputItem("b", "one", "known new", ""),
        InputItem("c", "two", "new", ""),
    ]
    rows = [
        {
            "source": "one",
            "probability_relevant": 0.90,
            "classification": "relevant",
        },
        {
            "source": "one",
            "probability_relevant": 0.60,
            "classification": "uncertain",
        },
        {
            "source": "two",
            "probability_relevant": 0.20,
            "classification": "not_relevant",
        },
    ]

    report = build_drift_report(
        FakeModel(),
        profile,
        items,
        rows,
        model_version="model-v1",
    )

    assert report["input_count"] == 3
    assert report["mean_probability_relevant"] == pytest.approx((0.9 + 0.6 + 0.2) / 3)
    assert report["uncertain_rate"] == pytest.approx(1 / 3)
    assert report["oov_feature_rate"] == pytest.approx(0.5)
    assert report["classification_distribution"] == {
        "relevant": 1,
        "uncertain": 1,
        "not_relevant": 1,
    }
    assert report["source_distribution"]["one"]["count"] == 2
    assert report["thresholds"] == {"relevant": 0.80, "not_relevant": 0.40}

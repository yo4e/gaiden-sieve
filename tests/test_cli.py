import json
from pathlib import Path

from gaiden_sieve.__main__ import main


ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_prints_metrics(capsys) -> None:
    exit_code = main(
        [
            "train",
            "--profile",
            "ai-gaiden",
            "--profiles-dir",
            str(ROOT / "profiles"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["train_size"] == 16
    assert payload["test_size"] == 4
    assert set(payload["metrics"]) == {"precision", "recall", "f1"}


def test_classify_cli_prints_probability(capsys) -> None:
    exit_code = main(
        [
            "classify",
            "--profile",
            "ai-gaiden",
            "--profiles-dir",
            str(ROOT / "profiles"),
            "--title",
            "New AI agent model announced",
            "--summary",
            "The system plans tasks and uses external tools.",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert 0.0 <= payload["prediction"]["probability_relevant"] <= 1.0
    assert payload["prediction"]["classification"] in {
        "relevant",
        "uncertain",
        "not_relevant",
    }

from pathlib import Path

import pytest

from gaiden_sieve.profile import ProfileValidationError, load_profile


ROOT = Path(__file__).resolve().parents[1]


def test_ai_gaiden_profile_loads() -> None:
    config = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")

    assert config.profile == "ai-gaiden"
    assert config.precision_gate == 0.95
    assert config.recall_gate == 0.80
    assert config.text_fields == ("title", "summary")


def test_profile_rejects_overlapping_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        """\
profile: ai-gaiden
positive_label: relevant
precision_gate: 0.95
recall_gate: 0.80
relevant_threshold: 0.40
not_relevant_threshold: 0.40
text_fields: [title, summary]
""",
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError, match="must be lower"):
        load_profile(path)

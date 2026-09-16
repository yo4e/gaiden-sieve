from datetime import datetime, timezone
from pathlib import Path

from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.reproducibility import verify_candidate_reproducibility
from gaiden_sieve.train import train_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_reproducibility_check_tracks_data_code_config_dependencies_and_metrics(
    tmp_path: Path,
) -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    data_path = ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl"
    metadata = train_candidate(
        items=load_labeled_jsonl(data_path),
        data_path=data_path,
        profile=profile,
        artifacts_dir=tmp_path,
        code_commit="abc123",
        created_at=datetime(2026, 9, 16, 1, 0, tzinfo=timezone.utc),
    )

    report = verify_candidate_reproducibility(
        data_path=data_path,
        profile=profile,
        artifacts_dir=tmp_path,
        model_version=str(metadata["model_version"]),
    )

    assert report["passed"] is True
    assert all(report["checks"].values())
    lineage = report["lineage"]
    assert lineage["code_commit"] == "abc123"
    assert lineage["random_seed"] == 42
    assert lineage["test_size"] == 0.20
    assert lineage["hyperparameters"]["classifier"]["name"] == "LogisticRegression"
    assert {"python", "scikit_learn", "joblib"} <= set(lineage["dependencies"])
    assert report["evaluation"]["metrics"] == lineage["metrics"]

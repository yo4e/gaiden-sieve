from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest

from gaiden_sieve.artifacts import (
    load_production,
    production_metadata_path,
    production_model_path,
)
from gaiden_sieve.classify import predict_one
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.promote import PromotionRejected, promote_candidate
from gaiden_sieve.train import train_candidate

ROOT = Path(__file__).resolve().parents[1]


def _candidate(tmp_path: Path):
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    data_path = ROOT / "profiles" / "ai-gaiden" / "labeled.jsonl"
    metadata = train_candidate(
        items=load_labeled_jsonl(data_path),
        data_path=data_path,
        profile=profile,
        artifacts_dir=tmp_path,
        code_commit="abc123",
        created_at=datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc),
    )
    return profile, metadata


def test_failed_candidate_cannot_replace_production(tmp_path: Path) -> None:
    profile, metadata = _candidate(tmp_path)
    failing_profile = replace(profile, precision_gate=1.0, recall_gate=1.0)
    # model hash は保ったまま成績だけ下げ、整合性はあるが gate を通らない candidate metadata を再現する。
    metadata_path = tmp_path / profile.profile / "candidate" / f"{metadata['model_version']}.metadata.json"
    import json
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["metrics"]["precision"] = 0.50
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PromotionRejected, match="failed quality gate"):
        promote_candidate(
            profile=failing_profile,
            artifacts_dir=tmp_path,
            model_version=str(metadata["model_version"]),
        )
    assert not production_model_path(tmp_path, profile.profile).exists()


def test_passing_candidate_promotes_and_production_classifies(tmp_path: Path) -> None:
    profile, metadata = _candidate(tmp_path)
    production = promote_candidate(
        profile=profile,
        artifacts_dir=tmp_path,
        model_version=str(metadata["model_version"]),
        promoted_at=datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc),
    )
    model, loaded_metadata = load_production(artifacts_dir=tmp_path, profile=profile.profile)
    prediction = predict_one(
        model,
        profile,
        title="New AI agent model announced",
        summary="The system plans tasks and uses external tools.",
    )
    assert production["stage"] == "production"
    assert loaded_metadata["source_candidate"] == metadata["model_version"]
    assert 0.0 <= prediction.probability_relevant <= 1.0
    assert prediction.classification in {"relevant", "uncertain", "not_relevant"}


class _NoOffsetTimezone(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize("existing_production", [False, True])
@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 9, 15, 9, 0), datetime(2026, 9, 15, 9, 0, tzinfo=_NoOffsetTimezone())],
    ids=["no-timezone", "no-offset"],
)
def test_invalid_promotion_time_leaves_production_untouched(
    tmp_path: Path, existing_production: bool, timestamp: datetime
) -> None:
    profile, metadata = _candidate(tmp_path)
    model_path = production_model_path(tmp_path, profile.profile)
    metadata_path = production_metadata_path(tmp_path, profile.profile)
    if existing_production:
        # 本番資産の代わりに、変更検出用のダミーファイルを一時ディレクトリへ置く。
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"previous production model")
        metadata_path.write_bytes(b'{"model_version": "previous"}\n')
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    with pytest.raises(PromotionRejected, match="promoted_at must include a timezone"):
        promote_candidate(
            profile=profile,
            artifacts_dir=tmp_path,
            model_version=str(metadata["model_version"]),
            promoted_at=timestamp,
        )

    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    if not existing_production:
        assert not model_path.parent.exists()


def test_promotion_time_is_normalized_to_utc(tmp_path: Path) -> None:
    profile, metadata = _candidate(tmp_path)
    production = promote_candidate(
        profile=profile,
        artifacts_dir=tmp_path,
        model_version=str(metadata["model_version"]),
        promoted_at=datetime(2026, 9, 15, 9, 0, tzinfo=timezone(timedelta(hours=9))),
    )
    _, loaded_metadata = load_production(artifacts_dir=tmp_path, profile=profile.profile)
    assert production["promoted_at"] == "2026-09-15T00:00:00Z"
    assert loaded_metadata == production

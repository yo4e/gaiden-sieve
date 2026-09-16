from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from gaiden_sieve.ai_gaiden_shadow import build_shadow_rows
from gaiden_sieve.data import load_labeled_jsonl
from gaiden_sieve.profile import load_profile
from gaiden_sieve.shadow import build_comparison_report, run_shadow_candidate
from gaiden_sieve.train import train_candidate


ROOT = Path(__file__).resolve().parents[1]


def _fake_item(
    item_id: str,
    *,
    title: str,
    source: str = "source-a",
    summary: str = "",
):
    return SimpleNamespace(
        source_id=source,
        title=title,
        summary=summary,
        canonical_url=f"https://example.com/{item_id}",
        dedupe_key=f"url:{item_id}",
        published_at=datetime(2026, 9, 16, 0, 0, tzinfo=UTC),
    )


def test_ai_gaiden_shadow_rows_preserve_existing_admission_outcome() -> None:
    config = SimpleNamespace(id="source-a")
    accepted = _fake_item("accepted", title="AI agent release")
    rejected = _fake_item("rejected", title="Storage maintenance")
    raw = [
        SimpleNamespace(
            config=config,
            success=True,
            not_modified=False,
            items=(accepted, rejected),
        )
    ]
    admitted = [
        SimpleNamespace(
            config=config,
            success=True,
            not_modified=False,
            items=(accepted,),
        )
    ]
    rules = {"source-a": SimpleNamespace(mode="filtered", policy="ai_terms_v1")}

    rows = build_shadow_rows(
        raw,
        admitted,
        rules,
        ai_gaiden_commit="abc123",
    )

    assert [row["admission_accepted"] for row in rows] == [True, False]
    assert all(row["admission_mode"] == "filtered" for row in rows)
    assert all(row["admission_policy"] == "ai_terms_v1" for row in rows)
    assert all(row["ai_gaiden_commit"] == "abc123" for row in rows)


def test_comparison_report_separates_uncertain_from_confident_disagreement() -> None:
    profile = load_profile(ROOT / "profiles" / "ai-gaiden" / "config.yml")
    shadow_rows = [
        {
            "id": "a",
            "source": "source-a",
            "title": "A",
            "summary": "",
            "admission_mode": "filtered",
            "admission_policy": "ai_terms_v1",
            "admission_accepted": True,
            "ai_gaiden_commit": "abc123",
        },
        {
            "id": "b",
            "source": "source-a",
            "title": "B",
            "summary": "",
            "admission_mode": "filtered",
            "admission_policy": "ai_terms_v1",
            "admission_accepted": True,
            "ai_gaiden_commit": "abc123",
        },
        {
            "id": "c",
            "source": "source-b",
            "title": "C",
            "summary": "",
            "admission_mode": "all",
            "admission_policy": None,
            "admission_accepted": True,
            "ai_gaiden_commit": "abc123",
        },
    ]
    classified = [
        {
            "id": "a",
            "classification": "relevant",
            "probability_relevant": 0.9,
        },
        {
            "id": "b",
            "classification": "not_relevant",
            "probability_relevant": 0.1,
        },
        {
            "id": "c",
            "classification": "uncertain",
            "probability_relevant": 0.6,
        },
    ]

    report = build_comparison_report(
        shadow_rows,
        classified,
        profile=profile,
        model_version="candidate-1",
        generated_at=datetime(2026, 9, 16, 1, 0, tzinfo=UTC),
    )

    assert report["ai_gaiden_commit"] == "abc123"
    assert report["confident_comparison"] == {
        "compared": 2,
        "agreement": 1,
        "disagreement": 1,
        "agreement_rate": 0.5,
    }
    assert report["uncertain_count"] == 1
    assert len(report["disagreements"]) == 1
    assert report["disagreements"][0]["id"] == "b"


def test_candidate_shadow_writes_reports_without_promoting_production(
    tmp_path: Path,
) -> None:
    profile_dir = ROOT / "profiles" / "ai-gaiden"
    profile = load_profile(profile_dir / "config.yml")
    data_path = profile_dir / "labeled.jsonl"
    artifacts = tmp_path / "artifacts"
    metadata = train_candidate(
        items=load_labeled_jsonl(data_path),
        data_path=data_path,
        profile=profile,
        artifacts_dir=artifacts,
        code_commit="test-shadow",
        created_at=datetime(2026, 9, 16, 2, 0, tzinfo=UTC),
    )

    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "live-1",
                        "source": "microsoft-cloud",
                        "title": "Azure AI adds new model inference tools",
                        "summary": "Developers can deploy generative AI workloads.",
                        "admission_mode": "filtered",
                        "admission_policy": "ai_terms_v1",
                        "admission_accepted": True,
                        "ai_gaiden_commit": "feedface",
                    }
                ),
                json.dumps(
                    {
                        "id": "live-2",
                        "source": "sourcegraph-changelog",
                        "title": "Sourcegraph self-hosted patch release",
                        "summary": "General maintenance and security fixes.",
                        "admission_mode": "filtered",
                        "admission_policy": "ai_terms_v1",
                        "admission_accepted": False,
                        "ai_gaiden_commit": "feedface",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_shadow_candidate(
        profile=profile,
        artifacts_dir=artifacts,
        model_version=str(metadata["model_version"]),
        input_path=incoming,
        classified_output=tmp_path / "classified.jsonl",
        drift_output=tmp_path / "drift.json",
        uncertain_output=tmp_path / "uncertain.jsonl",
        comparison_output=tmp_path / "comparison.json",
        generated_at=datetime(2026, 9, 16, 3, 0, tzinfo=UTC),
    )

    assert payload["production_promoted"] is False
    assert payload["input_count"] == 2
    assert (tmp_path / "classified.jsonl").exists()
    assert (tmp_path / "drift.json").exists()
    assert (tmp_path / "uncertain.jsonl").exists()
    comparison = json.loads(
        (tmp_path / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["ai_gaiden_commit"] == "feedface"
    assert not (artifacts / "ai-gaiden" / "production" / "model.joblib").exists()

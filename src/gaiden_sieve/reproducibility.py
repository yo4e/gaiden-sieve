"""Traceability checks for a saved candidate model."""

from __future__ import annotations

from pathlib import Path

from gaiden_sieve.artifacts import sha256_file, verify_candidate_integrity
from gaiden_sieve.profile import ProfileConfig
from gaiden_sieve.train import evaluate_candidate


def verify_candidate_reproducibility(
    *,
    data_path: str | Path,
    profile: ProfileConfig,
    artifacts_dir: str | Path,
    model_version: str,
) -> dict[str, object]:
    """Check that candidate lineage is complete and its recorded evaluation can be replayed."""

    metadata = verify_candidate_integrity(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    checks: dict[str, bool] = {
        "training_data_hash_matches": metadata.get("training_data_hash")
        == sha256_file(data_path),
        "code_commit_recorded": isinstance(metadata.get("code_commit"), str)
        and bool(metadata.get("code_commit")),
        "random_seed_recorded": isinstance(metadata.get("random_seed"), int),
        "test_size_recorded": isinstance(metadata.get("test_size"), (int, float)),
        "hyperparameters_recorded": isinstance(metadata.get("hyperparameters"), dict)
        and bool(metadata.get("hyperparameters")),
        "dependencies_recorded": isinstance(metadata.get("dependencies"), dict)
        and bool(metadata.get("dependencies")),
        "metrics_recorded": isinstance(metadata.get("metrics"), dict)
        and bool(metadata.get("metrics")),
    }

    evaluation: dict[str, object] | None = None
    if checks["training_data_hash_matches"]:
        _, evaluation = evaluate_candidate(
            data_path=data_path,
            profile=profile,
            artifacts_dir=artifacts_dir,
            model_version=model_version,
        )
        checks["evaluation_metrics_match"] = (
            evaluation.get("metrics") == metadata.get("metrics")
        )
    else:
        checks["evaluation_metrics_match"] = False

    return {
        "profile": profile.profile,
        "model_version": model_version,
        "passed": all(checks.values()),
        "checks": checks,
        "lineage": {
            "training_data_hash": metadata.get("training_data_hash"),
            "code_commit": metadata.get("code_commit"),
            "random_seed": metadata.get("random_seed"),
            "test_size": metadata.get("test_size"),
            "hyperparameters": metadata.get("hyperparameters"),
            "dependencies": metadata.get("dependencies"),
            "metrics": metadata.get("metrics"),
        },
        "evaluation": evaluation,
    }

"""Traceability checks for a saved candidate model."""

from __future__ import annotations

from pathlib import Path

from gaiden_sieve.artifacts import verify_candidate_integrity\nfrom gaiden_sieve.data import labeled_dataset_hash
from gaiden_sieve.profile import ProfileConfig
from gaiden_sieve.train import evaluate_candidate


def verify_candidate_reproducibility(
    *,
    data_path: str | Path,
    profile: ProfileConfig,
    artifacts_dir: str | Path,
    model_version: str,
    evaluation_data_path: str | Path | None = None,
) -> dict[str, object]:
    """Check lineage completeness and replay the candidate's recorded evaluation."""

    metadata = verify_candidate_integrity(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    evaluation_mode = str(metadata.get("evaluation_mode") or "random_holdout")
    checks: dict[str, bool] = {
        "training_data_hash_matches": metadata.get("training_data_hash")
        == labeled_dataset_hash(data_path),
        "code_commit_recorded": isinstance(metadata.get("code_commit"), str)
        and bool(metadata.get("code_commit")),
        "random_seed_recorded": isinstance(metadata.get("random_seed"), int),
        "evaluation_mode_recorded": evaluation_mode
        in {"random_holdout", "external_holdout"},
        "hyperparameters_recorded": isinstance(metadata.get("hyperparameters"), dict)
        and bool(metadata.get("hyperparameters")),
        "dependencies_recorded": isinstance(metadata.get("dependencies"), dict)
        and bool(metadata.get("dependencies")),
        "metrics_recorded": isinstance(metadata.get("metrics"), dict)
        and bool(metadata.get("metrics")),
    }

    if evaluation_mode == "external_holdout":
        checks["evaluation_data_hash_matches"] = (
            evaluation_data_path is not None
            and metadata.get("evaluation_data_hash")
            == labeled_dataset_hash(evaluation_data_path)
        )
        checks["test_size_recorded"] = metadata.get("test_size") is None
    else:
        checks["evaluation_data_hash_matches"] = metadata.get(
            "evaluation_data_hash"
        ) is None
        checks["test_size_recorded"] = isinstance(
            metadata.get("test_size"), (int, float)
        )

    evaluation: dict[str, object] | None = None
    if checks["training_data_hash_matches"] and checks["evaluation_data_hash_matches"]:
        _, evaluation = evaluate_candidate(
            data_path=data_path,
            evaluation_data_path=evaluation_data_path,
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
            "evaluation_mode": evaluation_mode,
            "evaluation_data_hash": metadata.get("evaluation_data_hash"),
            "code_commit": metadata.get("code_commit"),
            "random_seed": metadata.get("random_seed"),
            "test_size": metadata.get("test_size"),
            "hyperparameters": metadata.get("hyperparameters"),
            "dependencies": metadata.get("dependencies"),
            "metrics": metadata.get("metrics"),
        },
        "evaluation": evaluation,
    }

"""Model artifact and metadata storage for the local v0 lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


class ArtifactError(RuntimeError):
    """Raised when a model artifact or its metadata is missing or inconsistent."""


def sha256_file(path: str | Path) -> str:
    """Return a prefixed SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_code_commit(repo_root: str | Path = ".") -> str | None:
    """Resolve the code revision when running inside a Git checkout."""

    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _safe_version(model_version: str) -> str:
    if not model_version or model_version in {".", ".."}:
        raise ArtifactError("model_version must not be empty")
    if Path(model_version).name != model_version:
        raise ArtifactError("model_version must not contain path separators")
    return model_version


def candidate_dir(artifacts_dir: str | Path, profile: str) -> Path:
    return Path(artifacts_dir) / profile / "candidate"


def production_dir(artifacts_dir: str | Path, profile: str) -> Path:
    return Path(artifacts_dir) / profile / "production"


def candidate_model_path(
    artifacts_dir: str | Path, profile: str, model_version: str
) -> Path:
    return candidate_dir(artifacts_dir, profile) / f"{_safe_version(model_version)}.joblib"


def candidate_metadata_path(
    artifacts_dir: str | Path, profile: str, model_version: str
) -> Path:
    return candidate_dir(artifacts_dir, profile) / f"{_safe_version(model_version)}.metadata.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_candidate_model(
    model: Pipeline,
    *,
    artifacts_dir: str | Path,
    profile: str,
    model_version: str,
) -> Path:
    """Persist a candidate model without changing the production slot."""

    path = candidate_model_path(artifacts_dir, profile, model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, path)
    return path


def save_candidate_metadata(
    metadata: dict[str, Any],
    *,
    artifacts_dir: str | Path,
    profile: str,
    model_version: str,
) -> Path:
    """Persist metadata next to the candidate model it describes."""

    path = candidate_metadata_path(artifacts_dir, profile, model_version)
    _write_json(path, metadata)
    return path


def load_candidate_metadata(
    *, artifacts_dir: str | Path, profile: str, model_version: str
) -> dict[str, Any]:
    path = candidate_metadata_path(artifacts_dir, profile, model_version)
    if not path.exists():
        raise ArtifactError(f"candidate metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"candidate metadata must be a JSON object: {path}")
    if payload.get("profile") != profile or payload.get("model_version") != model_version:
        raise ArtifactError("candidate metadata does not match requested profile/version")
    return payload


def load_candidate_model(
    *, artifacts_dir: str | Path, profile: str, model_version: str
) -> Pipeline:
    path = candidate_model_path(artifacts_dir, profile, model_version)
    if not path.exists():
        raise ArtifactError(f"candidate model not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise ArtifactError(f"candidate artifact is not a scikit-learn Pipeline: {path}")
    return model


def verify_candidate_integrity(
    *, artifacts_dir: str | Path, profile: str, model_version: str
) -> dict[str, Any]:
    """Verify that candidate metadata still points to the exact saved model bytes."""

    metadata = load_candidate_metadata(
        artifacts_dir=artifacts_dir, profile=profile, model_version=model_version
    )
    expected = metadata.get("model_artifact_hash")
    actual = sha256_file(candidate_model_path(artifacts_dir, profile, model_version))
    if expected != actual:
        raise ArtifactError("candidate model hash does not match metadata")
    return metadata


def latest_candidate_version(*, artifacts_dir: str | Path, profile: str) -> str:
    """Return the newest candidate according to metadata created_at."""

    directory = candidate_dir(artifacts_dir, profile)
    candidates: list[tuple[str, str]] = []
    if directory.exists():
        for path in directory.glob("*.metadata.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = payload.get("model_version")
            created_at = payload.get("created_at")
            if isinstance(version, str) and isinstance(created_at, str):
                candidates.append((created_at, version))
    if not candidates:
        raise ArtifactError(f"no candidate metadata found for profile '{profile}'")
    return max(candidates)[1]


def production_model_path(artifacts_dir: str | Path, profile: str) -> Path:
    return production_dir(artifacts_dir, profile) / "model.joblib"


def production_metadata_path(artifacts_dir: str | Path, profile: str) -> Path:
    return production_dir(artifacts_dir, profile) / "metadata.json"


def save_production_metadata(
    metadata: dict[str, Any], *, artifacts_dir: str | Path, profile: str
) -> Path:
    path = production_metadata_path(artifacts_dir, profile)
    _write_json(path, metadata)
    return path


def load_production(
    *, artifacts_dir: str | Path, profile: str
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the currently promoted model and its lineage metadata."""

    model_path = production_model_path(artifacts_dir, profile)
    metadata_path = production_metadata_path(artifacts_dir, profile)
    if not model_path.exists() or not metadata_path.exists():
        raise ArtifactError(f"production model is not available for profile '{profile}'")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("profile") != profile:
        raise ArtifactError("production metadata does not match requested profile")
    expected = metadata.get("model_artifact_hash")
    actual = sha256_file(model_path)
    if expected != actual:
        raise ArtifactError("production model hash does not match metadata")

    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise ArtifactError("production artifact is not a scikit-learn Pipeline")
    return model, metadata

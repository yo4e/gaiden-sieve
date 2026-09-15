"""Training plus candidate creation for the GAIDEN SIEVE model lifecycle."""

from __future__ import annotations

import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Sequence

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from gaiden_sieve.artifacts import (
    ArtifactError,
    candidate_model_path,
    load_candidate_metadata,
    load_candidate_model,
    resolve_code_commit,
    save_candidate_metadata,
    save_candidate_model,
    sha256_file,
    verify_candidate_integrity,
)
from gaiden_sieve.data import LabeledItem, build_text, load_labeled_jsonl
from gaiden_sieve.evaluate import Metrics, calculate_metrics, evaluate_quality_gate
from gaiden_sieve.profile import ProfileConfig


RANDOM_SEED = 42
TEST_SIZE = 0.20


class TrainingError(ValueError):
    """Raised when labeled data cannot support the training contract."""


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """In-memory model plus the holdout score used during training."""

    model: Pipeline
    metrics: Metrics
    train_size: int
    test_size: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class _Split:
    x_train: list[str]
    x_test: list[str]
    y_train: list[str]
    y_test: list[str]


def build_pipeline(*, random_seed: int = RANDOM_SEED) -> Pipeline:
    """Create one preprocessing + classifier pipeline shared by train and inference."""

    # 学習時と推論時で同じ前処理を必ず通すため、TF-IDF と分類器を1本の Pipeline にまとめる。
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_seed),
            ),
        ]
    )


def _validate_training_data(items: Sequence[LabeledItem], *, test_size: float) -> None:
    label_counts = Counter(item.label for item in items)
    if set(label_counts) != {"relevant", "not_relevant"}:
        raise TrainingError("training data must contain relevant and not_relevant labels")
    if min(label_counts.values()) < 2:
        raise TrainingError("each label needs at least two items for a stratified split")

    test_count = ceil(len(items) * test_size)
    train_count = len(items) - test_count
    if test_count < len(label_counts) or train_count < len(label_counts):
        raise TrainingError("train/test split is too small to contain both labels")


def _split_items(
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int,
    test_size: float,
) -> _Split:
    if not 0.0 < test_size < 1.0:
        raise TrainingError("test_size must be between 0 and 1")
    _validate_training_data(items, test_size=test_size)

    texts = [build_text(item, profile.text_fields) for item in items]
    labels = [item.label for item in items]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    return _Split(
        x_train=list(x_train),
        x_test=list(x_test),
        y_train=list(y_train),
        y_test=list(y_test),
    )


def evaluate_trained_model(
    model: Pipeline,
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> Metrics:
    """Recreate the recorded holdout split and score an already trained model."""

    split = _split_items(
        items, profile, random_seed=random_seed, test_size=test_size
    )
    y_pred = model.predict(split.x_test)
    return calculate_metrics(
        split.y_test,
        y_pred,
        positive_label=profile.positive_label,
    )


def train_and_evaluate(
    items: Sequence[LabeledItem],
    profile: ProfileConfig,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> TrainingResult:
    """Train on one stratified split and score only the held-out test rows."""

    split = _split_items(
        items, profile, random_seed=random_seed, test_size=test_size
    )
    model = build_pipeline(random_seed=random_seed)
    model.fit(split.x_train, split.y_train)

    # test データはモデルへ見せず、学習後の成績確認だけに使う。
    metrics = calculate_metrics(
        split.y_test,
        model.predict(split.x_test),
        positive_label=profile.positive_label,
    )
    return TrainingResult(
        model=model,
        metrics=metrics,
        train_size=len(split.x_train),
        test_size=len(split.x_test),
        random_seed=random_seed,
    )


def _model_hyperparameters(model: Pipeline) -> dict[str, object]:
    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    return {
        "tfidf": {
            "ngram_range": list(tfidf.ngram_range),
            "max_features": tfidf.max_features,
        },
        "classifier": {
            "name": classifier.__class__.__name__,
            "C": float(classifier.C),
            "max_iter": int(classifier.max_iter),
        },
    }


def _created_at(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        raise TrainingError("created_at must include a timezone")
    return result.astimezone(timezone.utc)


def _model_version(created_at: datetime, data_hash: str) -> str:
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{data_hash.removeprefix('sha256:')[:8]}"


def train_candidate(
    *,
    items: Sequence[LabeledItem],
    data_path: str | Path,
    profile: ProfileConfig,
    artifacts_dir: str | Path = "artifacts",
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    code_commit: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Train, save, reload, evaluate, and describe one candidate artifact."""

    source_items = load_labeled_jsonl(data_path)
    if list(items) != source_items:
        raise TrainingError("items do not match the hashed training data file")

    data_hash = sha256_file(data_path)
    timestamp = _created_at(created_at)
    version = _model_version(timestamp, data_hash)
    result = train_and_evaluate(
        items,
        profile,
        random_seed=random_seed,
        test_size=test_size,
    )

    # candidate を一度ディスクへ保存し、再ロードした実物を評価する。保存前の in-memory model だけを信用しない。
    model_path = save_candidate_model(
        result.model,
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=version,
    )
    reloaded = load_candidate_model(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=version,
    )
    metrics = evaluate_trained_model(
        reloaded,
        items,
        profile,
        random_seed=random_seed,
        test_size=test_size,
    )
    gate = evaluate_quality_gate(metrics, profile)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "stage": "candidate",
        "profile": profile.profile,
        "model_version": version,
        "training_data_hash": data_hash,
        "model_artifact_hash": sha256_file(model_path),
        "code_commit": code_commit if code_commit is not None else resolve_code_commit(),
        "random_seed": random_seed,
        "test_size": test_size,
        "train_size": result.train_size,
        "holdout_size": result.test_size,
        "hyperparameters": _model_hyperparameters(reloaded),
        "metrics": asdict(metrics),
        "quality_gate": asdict(gate),
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "dependencies": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_candidate_metadata(
        metadata,
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=version,
    )
    return metadata


def evaluate_candidate(
    *,
    data_path: str | Path,
    profile: ProfileConfig,
    artifacts_dir: str | Path,
    model_version: str,
) -> tuple[Metrics, dict[str, object]]:
    """Re-evaluate a saved candidate using its recorded split settings."""

    metadata = verify_candidate_integrity(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    current_hash = sha256_file(data_path)
    if metadata.get("training_data_hash") != current_hash:
        raise ArtifactError(
            "training data hash changed; this candidate cannot be re-evaluated against a different dataset"
        )

    items = load_labeled_jsonl(data_path)
    model = load_candidate_model(
        artifacts_dir=artifacts_dir,
        profile=profile.profile,
        model_version=model_version,
    )
    metrics = evaluate_trained_model(
        model,
        items,
        profile,
        random_seed=int(metadata["random_seed"]),
        test_size=float(metadata["test_size"]),
    )
    gate = evaluate_quality_gate(metrics, profile)
    evaluation = {
        "profile": profile.profile,
        "model_version": model_version,
        "metrics": asdict(metrics),
        "quality_gate": asdict(gate),
    }
    return metrics, evaluation
